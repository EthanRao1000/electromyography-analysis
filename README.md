# Electromyography Analysis

Analyse raw electromyography data and interpret muscle contractions from BioWrap EMG recordings.

## Overview

This project reads raw EMG CSV data, converts ADC counts to microvolts, filters the signal, and identifies periods of muscle activity based on an RMS envelope.

It is designed for data captured from the ADS1299-based BioWrap system and can also handle simple CSV exports with one or two channels.

## Signal processing pipeline

The analysis script applies the following filters and processing steps:

1. ADC conversion to microvolts
   - Raw counts are converted using the ADS1299 scaling factor.
   - This is implemented in `counts_to_microvolts()`.

2. 50 Hz notch filter
   - A narrow IIR notch filter is applied to suppress mains hum.
   - Filter: `signal.iirnotch(50.0, Q=35.0, fs=500.0)`
   - Purpose: remove 50 Hz interference.

3. Band-pass filter
   - A 4th-order Butterworth band-pass filter is used.
   - Frequency range: 20 Hz to 200 Hz
   - Filter: `signal.butter(4, [20/250, 200/250], btype="bandpass")`
   - Purpose: keep muscle activity while removing slow drift and high-frequency noise.

4. RMS envelope detection
   - The filtered signal is rectified and converted into a moving RMS envelope.
   - Window length: 150 ms
   - Purpose: estimate the linear envelope of the EMG signal and detect activity.

5. Threshold-based activity classification
   - Candidate contraction threshold: 29 uV
   - Rest baseline threshold: 3 uV
   - Samples above the candidate threshold are treated as active.

## Input format

The script accepts CSV files with either:

- Legacy BioWrap format: `timestamp,emg1,emg2`
- Raw serial format: `millis,CH1`
- Simple numeric data without a header

## Output

The script generates:

- `processed_emg.csv` containing processed signal values and activity flags
- `emg_analysis.png` showing the raw, filtered, and envelope plots

## Usage

Run the analysis from the project folder:

```bash
python3 emg_analysis.py
```

Optional arguments:

```bash
python3 emg_analysis.py path/to/input.csv --output processed_emg.csv --plot emg_analysis.png
```

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

## Notes

- The project assumes a sampling rate of 500 Hz.
- If only one EMG channel is present, the second channel is filled with zeros so the output structure remains consistent.
- The current thresholds are set for basic contraction detection and may need tuning for different subjects or sensor placements.
