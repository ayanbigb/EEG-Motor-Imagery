"""
EEG Preprocessing Script
Demonstrates common preprocessing steps for motor imagery data.

Usage:
    python scripts/preprocess_eeg.py
"""

import os
import mne
import numpy as np
from scipy import signal

# ============================================================================
# PREPROCESSING PIPELINE FOR MOTOR IMAGERY EEG
# ============================================================================

print("=" * 70)
print("EEG PREPROCESSING PIPELINE")
print("=" * 70)
print()

# Load the raw EEG data
edf_path = "src/data/S001R04.edf"
raw = mne.io.read_raw_edf(edf_path, preload=True)

print(f"✓ Loaded raw data: {raw.times[-1]:.2f}s, {raw.info['nchan']} channels")
print()

# ============================================================================
# STEP 1: BANDPASS FILTERING (8-30 Hz for Motor Imagery)
# ============================================================================
# Motor imagery activity is primarily in the mu (8-12 Hz) and beta (15-30 Hz)
# frequency bands. We filter to keep only this range.

print("Step 1: Applying bandpass filter (8-30 Hz)...")
raw.filter(l_freq=8, h_freq=30, method='iir')
print("  ✓ Filtered")
print()

# ============================================================================
# STEP 2: NOTCH FILTERING (Remove 60 Hz power line noise)
# ============================================================================
# Remove electrical noise at 60 Hz and its harmonics

print("Step 2: Applying notch filter (60 Hz power line noise)...")
raw.notch_filter(freqs=60, method='iir')
print("  ✓ Notch filtered")
print()

# ============================================================================
# STEP 3: EXTRACT EVENTS AND CREATE EPOCHS
# ============================================================================
# Convert continuous data into time-locked segments (epochs) aligned to events

print("Step 3: Extracting events and creating epochs...")
events, event_id = mne.events_from_annotations(raw)

# Create epochs from -0.5s to +3.5s relative to event onset
# This captures the full motor imagery task (events are ~4 seconds each)
tmin, tmax = -0.5, 3.5
epochs = mne.Epochs(
    raw, 
    events, 
    event_id, 
    tmin, 
    tmax,
    baseline=(tmin, 0),  # Use pre-stimulus period as baseline
    preload=True
)

print(f"  ✓ Created {len(epochs)} epochs")
print(f"  ✓ Epoch duration: {tmax - tmin:.1f} seconds")
print(f"  ✓ Event breakdown:")
for label, count in event_id.items():
    n_this_event = np.sum(events[:, 2] == count)
    print(f"      {label}: {n_this_event} trials")
print()

# ============================================================================
# STEP 4: REMOVE BAD EPOCHS (Simple outlier detection)
# ============================================================================
# Remove epochs with excessive amplitude (potential artifacts)

print("Step 4: Detecting and removing bad epochs...")
initial_count = len(epochs)

# Remove epochs with peak-to-peak amplitude > 150 μV on any channel
# (typical threshold for artifacts)
peak_to_peak = np.ptp(epochs.get_data(), axis=2)  # (epochs, channels)
bad_epochs = np.where(np.max(peak_to_peak, axis=1) > 150)[0]

if len(bad_epochs) > 0:
    epochs.drop(bad_epochs, reason="excessive amplitude")
    print(f"  ✓ Removed {len(bad_epochs)} epochs with excessive amplitude")
else:
    print(f"  ✓ No bad epochs detected")

print(f"  ✓ Final epoch count: {len(epochs)} (removed {initial_count - len(epochs)})")
print()

# ============================================================================
# STEP 5: BASELINE CORRECTION (Zero-mean normalization)
# ============================================================================
# Normalize each epoch relative to its pre-stimulus baseline

print("Step 5: Applying baseline correction...")
epochs.apply_baseline((tmin, 0))
print("  ✓ Baseline corrected (pre-stimulus mean removed)")
print()

# ============================================================================
# STEP 6: COMMON SPATIAL PATTERNS (CSP) FEATURES
# ============================================================================
# CSP is a powerful feature extraction technique for motor imagery
# It finds spatial filters that maximize variance differences between classes

print("Step 6: Computing Common Spatial Patterns (CSP)...")

# Extract data for T1 (e.g., left hand imagery) vs T2 (right hand imagery)
t1_epochs = epochs['T1'].get_data()  # shape: (n_epochs_t1, channels, timepoints)
t2_epochs = epochs['T2'].get_data()  # shape: (n_epochs_t2, channels, timepoints)

print(f"  • T1 class: {t1_epochs.shape[0]} trials")
print(f"  • T2 class: {t2_epochs.shape[0]} trials")

# Compute covariance matrices for each class
cov_t1 = np.zeros((raw.info['nchan'], raw.info['nchan']))
cov_t2 = np.zeros((raw.info['nchan'], raw.info['nchan']))

for trial in t1_epochs:
    cov_t1 += trial @ trial.T

for trial in t2_epochs:
    cov_t2 += trial @ trial.T

cov_t1 /= len(t1_epochs)
cov_t2 /= len(t2_epochs)

print("  ✓ Covariance matrices computed")
print()

# ============================================================================
# STEP 7: SAVE PREPROCESSED DATA
# ============================================================================
# Save epochs for later use in model training

print("Step 7: Saving preprocessed data...")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save epochs in MNE format (use correct naming convention)
epochs_path = "data/epochs_preprocessed-epo.fif"
epochs.save(epochs_path, overwrite=True)
print(f"  ✓ Saved epochs to {epochs_path}")

# Also save as NumPy arrays for easier access
X = epochs.get_data()  # shape: (n_epochs, channels, timepoints)
y = epochs.events[:, 2]  # event IDs

np.savez(
    "data/preprocessed_data.npz",
    X=X,
    y=y,
    ch_names=np.array(epochs.ch_names),
    sfreq=epochs.info['sfreq'],
    event_id=event_id
)
print(f"  ✓ Saved NumPy arrays to data/preprocessed_data.npz")
print(f"    Shape: X={X.shape}, y={y.shape}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
print(f"✓ Loaded: S001R04.edf")
print(f"✓ Filtered: 8-30 Hz bandpass + 60 Hz notch")
print(f"✓ Epoched: {len(epochs)} trials, {tmax - tmin:.1f}s each")
print(f"✓ Cleaned: Removed epochs with excessive artifacts")
print(f"✓ Baseline corrected: Pre-stimulus normalization applied")
print(f"✓ Saved: Preprocessed data ready for model training")
print()
print("Next steps:")
print("  1. Use X and y from preprocessed_data.npz for model training")
print("  2. Extract features (CSP, band power, entropy, etc.)")
print("  3. Train classifier (LDA, SVM, CNN, etc.)")
print()
