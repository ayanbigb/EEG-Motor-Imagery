# Quick Start Guide

This guide helps you get the EEG Motor Imagery Classification project running locally.

## Prerequisites

- Python 3.10+ 
- macOS (or Linux/Windows with minor adjustments)
- The EDF data files: `S001R04.edf` and `S001R04.edf.event`

## Setup Instructions

### 1. Create Virtual Environment

```bash
cd /Users/ayanbhakta/EEG-Motor-Imagery
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

Test that MNE-Python can load EEG data:

```bash
python -c "import mne; print(f'MNE version: {mne.__version__}')"
```

## Running the Code

### Step 1: Explore Raw EEG Data

Load and visualize the raw EEG recording:

```bash
python scripts/explore_eeg_data.py
```

**Output:**
- Prints dataset info (64 channels, 160 Hz sampling, ~2 min duration)
- Lists all 30 motor imagery events (T0, T1, T2)
- Generates 3 diagnostic plots:
  - `eeg_raw_signal.png` - Raw EEG traces
  - `eeg_power_spectrum.png` - Frequency analysis
  - `eeg_zoomed_view.png` - Zoomed 10-second view

### Step 2: Preprocess EEG Data

Apply filters, artifact removal, and feature extraction:

```bash
python scripts/preprocess_eeg.py
```

**Output:**
- Applies 8-30 Hz bandpass filter (motor imagery frequency range)
- Removes 60 Hz power line noise
- Creates 29 epochs (4 seconds each)
- Removes bad trials with excessive noise
- Computes baseline correction
- Saves preprocessed data:
  - `data/epochs_preprocessed-epo.fif` (MNE format)
  - `data/preprocessed_data.npz` (NumPy arrays)

### Step 3: Train Classification Model

Use the template training script (requires model implementation):

```bash
python scripts/train.py --config configs/config.yaml
```

## Project Structure

```
EEG-Motor-Imagery/
├── src/                      # Core Python package
│   ├── data/                # Data loading utilities
│   │   ├── dataset.py      # EEGDataset class
│   │   └── __init__.py
│   ├── models/             # Model definitions
│   │   ├── classifier.py   # SimpleCNN model
│   │   └── __init__.py
│   └── utils/              # Preprocessing utilities
│       ├── preprocessing.py
│       └── __init__.py
├── scripts/                 # Runnable scripts
│   ├── explore_eeg_data.py  # Load & visualize raw data ✓
│   ├── preprocess_eeg.py    # Filter & epoch data ✓
│   └── train.py             # Train classification model
├── configs/                 # Configuration files
│   └── config.yaml         # Training config
├── data/                    # Data directory (created after preprocessing)
│   ├── S001R04.edf        # Raw EEG recording
│   ├── S001R04.edf.event  # Event annotations
│   ├── epochs_preprocessed-epo.fif
│   └── preprocessed_data.npz
├── notebooks/               # Jupyter notebooks for analysis
├── tests/                   # Unit tests
│   └── test_dataset.py
├── requirements.txt         # Python dependencies
└── README.md               # Project overview
```

## Data Information

### S001R04.edf Dataset

From the PhysioNet EEG Motor Movement/Imagery Dataset:

| Property | Value |
|----------|-------|
| Channels | 64 (10-20 EEG montage) |
| Sampling Rate | 160 Hz |
| Duration | ~2 minutes |
| Total Samples | 20,000 |
| Events | 30 motor imagery tasks |

### Event Labels

- **T0** (15 trials): Baseline/rest
- **T1** (8 trials): Motor imagery (hand movement 1)
- **T2** (7 trials): Motor imagery (hand movement 2)

### Channel Locations

Full 10-20 system:
- Frontal: Fp1, Fpz, Fp2, F7, F5, F3, F1, Fz, F2, F4, F6, F8
- Central: Ft7, Fc5, Fc3, Fc1, Fcz, Fc2, Fc4, Fc6, Ft8
- Parietal: T7, C5, C3, C1, Cz, C2, C4, C6, T8
- Temporal: Tp7, Cp5, Cp3, Cp1, Cpz, Cp2, Cp4, Cp6, Tp8
- Posterior: P7, P5, P3, P1, Pz, P2, P4, P6, P8
- Occipital: Po7, Po3, Poz, Po4, Po8, O1, Oz, O2, Iz

## Next Steps

1. **Feature Engineering**: Extract CSP (Common Spatial Patterns), band power, spectral entropy
2. **Model Selection**: Try LDA, SVM, CNN, or deep learning approaches
3. **Cross-validation**: Use k-fold CV to assess generalization
4. **Hyperparameter Tuning**: Optimize learning rate, filter frequencies, model architecture
5. **Multiple Subjects**: Extend to all PhysioNet subjects for robust evaluation

## Troubleshooting

### MNE import fails

```bash
# Verify venv is activated
which python  # Should show .../venv/bin/python

# Reinstall MNE
pip install --force-reinstall mne
```

### Plots not saving

- Verify `data/` directory exists: `mkdir -p data`
- Check file permissions: `ls -la *.png`

### Memory issues with large datasets

- Reduce `n_channels` in preprocessing
- Use `raw.filter()` with lower `l_freq` cutoff
- Process subjects one at a time

## References

- MNE-Python: https://mne.tools/
- PhysioNet Dataset: https://physionet.org/content/eegmmidb/1.0.0/
- Motor Imagery Classification: https://arxiv.org/abs/1611.08024

## Author Notes

This scaffold demonstrates a complete BCI preprocessing pipeline. The key insight for motor imagery classification is that movement imagination produces distinct patterns in the mu (8-12 Hz) and beta (15-30 Hz) bands, which can be captured by spatial filters like CSP.
