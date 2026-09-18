#!/usr/bin/env python3
"""Analyze BioWrap EMG CSV data using a band-pass + 50 Hz notch + RMS envelope pipeline.

Expected CSV format:
    timestamp,emg1,emg2
    56136,-172118,-5736
    ...

Interpretation:
- ADC counts are converted to microvolts using the ADS1299 gain and reference.
- Raw signal is band-pass filtered 20-200 Hz.
- 50 Hz notch is applied to reduce mains hum.
- Signal is rectified and converted to a linear envelope via a 150 ms RMS window.
- A candidate threshold of 29 uV is used, with rest threshold around 3 uV.

The script writes:
- processed_emg.csv
- emg_analysis.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


FS = 500.0
ADC_COUNTS_PER_UV = (4.5 / 8388607.0) * 1e6 / 1.0
THRESHOLD_CANDIDATE_UV = 29.0
THRESHOLD_REST_UV = 3.0
RMS_WINDOW_MS = 150.0
BANDPASS_LOW_HZ = 20.0
BANDPASS_HIGH_HZ = 200.0
NOTCH_HZ = 50.0


def load_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CSV and return timestamp, ch1, ch2 as numpy arrays.

    Supports both of these formats:
    - legacy BioWrap format: timestamp,emg1,emg2
    - raw ESP32 serial format from main.cpp: millis,CH1
    - raw CSV without a header: timestamp,emg1 or millis,CH1

    If only one EMG channel is present, the missing channel is set to zero so the
    downstream analysis still has a valid 2-channel structure.
    """
    with open(path, "r", newline="") as f:
        rows = [row for row in csv.reader(f) if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"No rows found in CSV: {path}")

    def parse_values(row: list[str]) -> list[float]:
        values: list[float] = []
        for cell in row:
            value = cell.strip()
            if value:
                values.append(float(value))
        return values

    header = [cell.strip().lower() for cell in rows[0]]
    if any(name in {"timestamp", "millis", "time", "emg1", "ch1", "channel 1", "emg2", "ch2", "channel 2"} for name in header):
        header_map = {cell.strip().lower(): idx for idx, cell in enumerate(rows[0])}
        ts_idx = header_map.get("timestamp", header_map.get("millis", header_map.get("time")))
        ch1_idx = (
            header_map.get("emg1",
            header_map.get("ch1",
            header_map.get("channel 1")))
        )
        ch2_idx = (
            header_map.get("emg2",
            header_map.get("ch2",
            header_map.get("channel 2")))
        )

        if ts_idx is None:
            raise ValueError(f"CSV header in {path} does not contain a timestamp column")
        if ch1_idx is None:
            raise ValueError(f"CSV header in {path} does not contain an emg1/ch1 column")

        data_rows = rows[1:]
        ts = np.array([float(row[ts_idx]) for row in data_rows], dtype=np.float64)
        ch1 = np.array([float(row[ch1_idx]) for row in data_rows], dtype=np.float64)

        if ch2_idx is None:
            ch2 = np.zeros_like(ch1)
        else:
            ch2 = np.array([float(row[ch2_idx]) for row in data_rows], dtype=np.float64)

        return ts, ch1, ch2

    parsed_rows = [parse_values(row) for row in rows]
    ts_values: list[float] = []
    ch1_values: list[float] = []
    ch2_values: list[float] = []

    for values in parsed_rows:
        if len(values) < 2:
            raise ValueError(f"Row '{values}' in {path} is not valid; expected at least two numeric values.")
        ts_values.append(values[0])
        ch1_values.append(values[1])
        ch2_values.append(0.0 if len(values) < 3 else values[2])

    return np.array(ts_values, dtype=np.float64), np.array(ch1_values, dtype=np.float64), np.array(ch2_values, dtype=np.float64)


def counts_to_microvolts(raw_counts: np.ndarray) -> np.ndarray:
    """Convert ADS1299 raw counts to microvolts for a PGA gain of 1."""
    return raw_counts * ADC_COUNTS_PER_UV


def apply_notch_filter(signal_in: np.ndarray, fs: float, notch_hz: float = NOTCH_HZ) -> np.ndarray:
    """Apply a narrow 50 Hz notch filter to reject mains hum."""
    b_notch, a_notch = signal.iirnotch(notch_hz, Q=35.0, fs=fs)
    return signal.filtfilt(b_notch, a_notch, signal_in)


def apply_bandpass(signal_in: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Band-pass filter 20-200 Hz using a 4th-order Butterworth filter."""
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = signal.butter(4, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, signal_in)


def moving_rms_envelope(signal_in: np.ndarray, fs: float, window_ms: float = RMS_WINDOW_MS) -> np.ndarray:
    """Compute a 150 ms RMS envelope of the rectified EMG.

    This approximates the EMG linear envelope used in many muscle-activity analyses.
    """
    window_samples = max(1, int(round((window_ms / 1000.0) * fs)))
    if window_samples % 2 == 0:
        window_samples += 1

    rectified = np.abs(signal_in)
    # Use a smooth moving RMS window through the full signal.
    rms = np.sqrt(signal.convolve(rectified ** 2, np.ones(window_samples) / window_samples, mode="same"))
    return rms


def classify_activity(envelope_uv: np.ndarray, candidate_threshold_uv: float = THRESHOLD_CANDIDATE_UV,
                      rest_threshold_uv: float = THRESHOLD_REST_UV) -> np.ndarray:
    """Return boolean activity mask, using 29 uV candidate threshold and 3 uV rest baseline."""
    active = envelope_uv >= candidate_threshold_uv
    rest = envelope_uv <= rest_threshold_uv
    result = np.zeros_like(envelope_uv, dtype=bool)
    result[active] = True
    result[rest] = False
    return result


def analyze_channel(raw_counts: np.ndarray) -> dict:
    """Process one EMG channel into microvolts, bandpass, notch, and envelope."""
    raw_uv = counts_to_microvolts(raw_counts)
    notch = apply_notch_filter(raw_uv, FS, NOTCH_HZ)
    band = apply_bandpass(notch, FS, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
    rectified = np.abs(band)
    envelope = moving_rms_envelope(band, FS, RMS_WINDOW_MS)
    activity = classify_activity(envelope)
    return {
        "raw_uv": raw_uv,
        "notch_uv": notch,
        "band_uv": band,
        "rectified_uv": rectified,
        "envelope_uv": envelope,
        "active": activity,
    }


def save_processed_csv(path: str | Path, timestamp: np.ndarray, ch1: dict, ch2: dict) -> None:
    """Write the processed data to CSV with a useful summary format."""
    rows = []
    for i in range(len(timestamp)):
        rows.append({
            "timestamp": float(timestamp[i]),
            "emg1_raw_uv": float(ch1["raw_uv"][i]),
            "emg2_raw_uv": float(ch2["raw_uv"][i]),
            "emg1_filtered_uv": float(ch1["band_uv"][i]),
            "emg2_filtered_uv": float(ch2["band_uv"][i]),
            "emg1_envelope_uv": float(ch1["envelope_uv"][i]),
            "emg2_envelope_uv": float(ch2["envelope_uv"][i]),
            "emg1_active": int(ch1["active"][i]),
            "emg2_active": int(ch2["active"][i]),
        })

    fieldnames = [
        "timestamp",
        "emg1_raw_uv",
        "emg2_raw_uv",
        "emg1_filtered_uv",
        "emg2_filtered_uv",
        "emg1_envelope_uv",
        "emg2_envelope_uv",
        "emg1_active",
        "emg2_active",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(timestamp: np.ndarray, ch1: dict, ch2: dict, output_path: str | Path) -> None:
    """Plot a multi-panel EMG analysis figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    # Raw counts vs microvolts
    ax1.plot(timestamp, ch1["raw_uv"], color="tab:blue", alpha=0.8)
    ax1.set_title("CH1 raw (uV)")
    ax1.set_ylabel("uV")

    ax2.plot(timestamp, ch2["raw_uv"], color="tab:orange", alpha=0.8)
    ax2.set_title("CH2 raw (uV)")
    ax2.set_ylabel("uV")

    # Filtered 20-200 Hz
    ax3.plot(timestamp, ch1["band_uv"], color="tab:blue", alpha=0.9)
    ax3.set_title("CH1 20-200 Hz filtered")
    ax3.set_ylabel("uV")

    ax4.plot(timestamp, ch2["band_uv"], color="tab:orange", alpha=0.9)
    ax4.set_title("CH2 20-200 Hz filtered")
    ax4.set_ylabel("uV")

    # Envelope / RMS
    ax5.plot(timestamp, ch1["envelope_uv"], color="tab:blue", linewidth=2)
    ax5.axhline(THRESHOLD_CANDIDATE_UV, color="red", linestyle="--", label=f"candidate {THRESHOLD_CANDIDATE_UV} uV")
    ax5.axhline(THRESHOLD_REST_UV, color="gray", linestyle=":", label=f"rest {THRESHOLD_REST_UV} uV")
    ax5.set_title("CH1 EMG envelope (150 ms RMS)")
    ax5.set_xlabel("time (ms)")
    ax5.set_ylabel("uV")
    ax5.legend()

    ax6.plot(timestamp, ch2["envelope_uv"], color="tab:orange", linewidth=2)
    ax6.axhline(THRESHOLD_CANDIDATE_UV, color="red", linestyle="--", label=f"candidate {THRESHOLD_CANDIDATE_UV} uV")
    ax6.axhline(THRESHOLD_REST_UV, color="gray", linestyle=":", label=f"rest {THRESHOLD_REST_UV} uV")
    ax6.set_title("CH2 EMG envelope (150 ms RMS)")
    ax6.set_xlabel("time (ms)")
    ax6.set_ylabel("uV")
    ax6.legend()

    for axis in axes.flat:
        axis.grid(True, alpha=0.2)

    plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze raw EMG CSV data.")
    parser.add_argument("input_csv", nargs="?", default=str(project_root / "data1.csv"), help="CSV file to analyze.")
    parser.add_argument("--output", default=str(project_root / "processed_emg.csv"), help="Output CSV path.")
    parser.add_argument("--plot", default=str(project_root / "emg_analysis.png"), help="Output plot path.")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    output_path = Path(args.output)
    plot_path = Path(args.plot)

    ts, ch1_counts, ch2_counts = load_csv(csv_path)
    ch1 = analyze_channel(ch1_counts)
    ch2 = analyze_channel(ch2_counts)

    save_processed_csv(output_path, ts, ch1, ch2)
    plot_results(ts, ch1, ch2, plot_path)

    print(f"Loaded: {csv_path}")
    print(f"Processed CSV saved to: {output_path}")
    print(f"Plot saved to: {plot_path}")
    print(f"Threshold candidate: {THRESHOLD_CANDIDATE_UV} uV")
    print(f"Rest baseline: {THRESHOLD_REST_UV} uV")
    print(f"Sample rate: {FS} Hz")
    print(f"Band-pass: {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz")
    print(f"Notch: {NOTCH_HZ} Hz")
    print(f"RMS window: {RMS_WINDOW_MS} ms")


if __name__ == "__main__":
    main()
