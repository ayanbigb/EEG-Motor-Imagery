"""
EEG Data Exploration Script
Loads and analyzes the PhysioNet EEG Motor Movement/Imagery Dataset
using MNE-Python for comprehensive data inspection.

Usage:
    python scripts/explore_eeg_data.py
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Load the EDF file
# ============================================================================
# The EDF file contains the raw EEG recordings. MNE reads it and creates
# an mne.io.Raw object with all channel data, sampling rate, and metadata.

edf_path = "src/data/S001R04.edf"

if not os.path.exists(edf_path):
    raise FileNotFoundError(f"EDF file not found at {edf_path}")

print("=" * 70)
print("STEP 1: LOADING EDF FILE")
print("=" * 70)

# Read the raw EEG data from the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)

print(f"✓ Successfully loaded: {edf_path}")
print()


# ============================================================================
# STEP 2: Print basic dataset information
# ============================================================================
# Display key metadata about the recording: number of channels, sampling rate,
# duration, channel names, and data type.

print("=" * 70)
print("STEP 2: BASIC DATASET INFORMATION")
print("=" * 70)

print(f"Number of channels: {raw.info['nchan']}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Recording duration: {raw.times[-1]:.2f} seconds ({raw.times[-1] / 60:.2f} minutes)")
print(f"Total number of samples: {raw.n_times}")
print(f"Channel names: {raw.ch_names}")
print(f"Data shape: {raw.get_data().shape}  (channels, timepoints)")
print()


# ============================================================================
# STEP 3: Print all annotations and event labels
# ============================================================================
# Annotations are events marked in the EEG file (e.g., motor imagery tasks).
# They have timestamps, durations, and descriptions.

print("=" * 70)
print("STEP 3: ANNOTATIONS AND EVENT LABELS")
print("=" * 70)

if len(raw.annotations) > 0:
    print(f"Total number of annotations: {len(raw.annotations)}")
    print()
    print("Annotation details:")
    print("-" * 70)
    for i, annotation in enumerate(raw.annotations):
        print(f"  Event {i+1}:")
        print(f"    Description: {annotation['description']}")
        print(f"    Onset: {annotation['onset']:.2f} seconds")
        print(f"    Duration: {annotation['duration']:.2f} seconds")
    print()
else:
    print("No annotations found in the raw file.")
    print()


# ============================================================================
# STEP 4: Extract events using mne.events_from_annotations()
# ============================================================================
# This function converts annotations into an events array with shape
# (n_events, 3), where each row is [sample_index, 0, event_id].
# We also get a mapping of event labels to integer IDs.

print("=" * 70)
print("STEP 4: EXTRACT EVENTS FROM ANNOTATIONS")
print("=" * 70)

# Convert annotations to events array and event dictionary
events, event_id = mne.events_from_annotations(raw)

print(f"✓ Events extracted successfully")
print(f"Shape of events array: {events.shape}")
print(f"  (n_events={events.shape[0]}, 3 columns: [sample, 0, event_id])")
print()
print("First 5 events:")
print(events[:5])
print()


# ============================================================================
# STEP 5: Display the event dictionary
# ============================================================================
# The event dictionary maps event labels (strings) to integer IDs.
# This is used when creating epochs for specific event types.

print("=" * 70)
print("STEP 5: EVENT DICTIONARY")
print("=" * 70)

print("Event label to ID mapping:")
print("-" * 70)
for label, event_id_val in sorted(event_id.items()):
    # Count how many times this event occurs
    count = np.sum(events[:, 2] == event_id_val)
    print(f"  '{label:30}' → ID {event_id_val:3d}  (count: {count})")
print()


# ============================================================================
# STEP 6: Create EEG plots to inspect the recording
# ============================================================================
# Plot 1: Raw signal overview (all channels over time)
# Plot 2: Power spectral density (frequency content)
# Plot 3: Zoomed view of first 10 seconds to see individual channels clearly

print("=" * 70)
print("STEP 6: GENERATING EEG PLOTS")
print("=" * 70)

# Suppress MNE warnings for cleaner output
mne.set_log_level("ERROR")

# Plot 1: Raw signal (overview of entire recording)
print("  • Creating raw signal plot...")
fig1 = raw.plot(
    duration=60,  # Show 60 seconds at a time
    n_channels=10,  # Show only first 10 channels for readability
    title="Raw EEG Signal (first 60 seconds, first 10 channels)",
    show=False,
)
fig1.savefig("eeg_raw_signal.png", dpi=100, bbox_inches="tight")
print("    Saved: eeg_raw_signal.png")

# Plot 2: Power spectral density (frequency content)
print("  • Creating power spectral density plot...")
fig2 = raw.compute_psd().plot(
    picks="eeg",
    show=False,
)
fig2.suptitle("Power Spectral Density (Frequency Content)", fontsize=14)
fig2.savefig("eeg_power_spectrum.png", dpi=100, bbox_inches="tight")
print("    Saved: eeg_power_spectrum.png")

# Plot 3: Zoomed view of first 10 seconds
print("  • Creating zoomed signal view...")
fig3, axes = plt.subplots(figsize=(14, 8))
# Get data from first 10 seconds
time_limit = min(10, raw.times[-1])  # Get 10 sec or less if shorter
t_mask = raw.times <= time_limit
data_zoomed = raw.get_data()[:, t_mask]
time_zoomed = raw.times[t_mask]

# Plot each channel
for i, (channel, ch_data) in enumerate(zip(raw.ch_names, data_zoomed)):
    axes.plot(time_zoomed, ch_data + i * 500, label=channel, linewidth=0.8, alpha=0.8)

axes.set_xlabel("Time (seconds)")
axes.set_ylabel("Channel (offset for visibility)")
axes.set_title("Zoomed EEG Signal (first 10 seconds)")
axes.grid(True, alpha=0.3)
axes.legend(loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
fig3.savefig("eeg_zoomed_view.png", dpi=100, bbox_inches="tight")
print("    Saved: eeg_zoomed_view.png")

plt.close("all")
print()


# ============================================================================
# STEP 7: Additional insights
# ============================================================================
# Summary statistics and data quality checks

print("=" * 70)
print("STEP 7: ADDITIONAL INSIGHTS")
print("=" * 70)

# Compute statistics for each channel
print("Channel statistics (mean ± std in μV):")
print("-" * 70)
data = raw.get_data() * 1e6  # Convert to microvolts
for i, ch_name in enumerate(raw.ch_names):
    mean_val = np.mean(data[i])
    std_val = np.std(data[i])
    print(f"  {ch_name:6} → mean: {mean_val:8.2f} μV, std: {std_val:8.2f} μV")
print()

# Check for bad channels
if len(raw.info["bads"]) > 0:
    print(f"⚠ Bad channels marked: {raw.info['bads']}")
else:
    print("✓ No bad channels detected")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Successfully analyzed {edf_path}")
print(f"✓ Generated 3 diagnostic plots: eeg_raw_signal.png, ")
print(f"  eeg_power_spectrum.png, eeg_zoomed_view.png")
print(f"✓ Found {len(events)} motor imagery/movement events")
print(f"✓ Ready for preprocessing and feature extraction!")
print()
