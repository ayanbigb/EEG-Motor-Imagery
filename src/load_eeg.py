"""
load_eeg.py

Load a PhysioNet EDF file with MNE, print recording information,
extract annotations/events, and build epochs and labels for T1/T2.

Usage:
    python3 src/load_eeg.py
"""

import os
import sys
import numpy as np

try:
    import mne
except Exception:
    print("Error: failed to import MNE-Python. Install it with: python -m pip install mne")
    raise


def find_edf_file(filename="S001R04.edf"):
    """Search common project locations for the EDF file."""
    candidates = [
        os.path.join("data", filename),
        os.path.join("src", "data", filename),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    # 1) Locate the EDF file. Search `data/` and `src/data/` for the raw file.
    edf_rel = find_edf_file("S001R04.edf")
    if edf_rel is None:
        print("Could not find S001R04.edf in common locations.")
        print("Please place the EDF file at data/S001R04.edf or src/data/S001R04.edf")
        sys.exit(2)

    # 2) Load the EDF file with MNE
    print(f"Loading EDF file: {edf_rel}")
    try:
        raw = mne.io.read_raw_edf(edf_rel, preload=True, verbose='WARNING')
    except FileNotFoundError:
        print(f"File not found: {edf_rel}")
        sys.exit(3)
    except Exception as e:
        print(f"Failed to read EDF file: {e}")
        sys.exit(4)

    # 3) Print basic recording information
    print("\n=== Basic recording information ===")
    n_channels = raw.info.get('nchan', 'unknown')
    sfreq = raw.info.get('sfreq', 'unknown')
    duration_s = raw.times[-1] if raw.n_times > 0 else 0.0
    print(f"Channels       : {n_channels}")
    print(f"Sampling rate  : {sfreq} Hz")
    print(f"Duration       : {duration_s:.2f} seconds ({duration_s/60:.2f} minutes)")
    print(f"Total samples  : {raw.n_times}")
    print(f"Channel names  : {raw.ch_names}")

    # 4) Extract annotations and convert to events
    print("\n=== Annotations and events ===")
    if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
        print(f"Found {len(raw.annotations)} annotations")
        for i, ann in enumerate(raw.annotations):
            print(f"  {i+1:2d}. desc={ann['description']!s:6s} onset={ann['onset']:.2f}s dur={ann['duration']:.2f}s")
    else:
        print("No annotations found in the EDF file.")

    try:
        events, event_id = mne.events_from_annotations(raw)
    except Exception as e:
        print(f"Failed to extract events from annotations: {e}")
        sys.exit(5)

    # 5) Print the event dictionary
    print("\nEvent dictionary:")
    if event_id:
        for k, v in event_id.items():
            print(f"  '{k}' -> {v}")
    else:
        print("  (empty)")

    # 6) Print the first 10 events
    print("\nFirst 10 events (sample, 0, event_id):")
    if len(events) == 0:
        print("  No events available to display.")
    else:
        n_show = min(10, len(events))
        for row in events[:n_show]:
            print(f"  {row.tolist()}")

    # 7) Filter the raw EEG between 8 and 30 Hz
    #    This keeps the motor imagery-relevant frequency bands while removing
    #    slow drift and high-frequency noise.
    raw_filtered = raw.copy()
    print("\nFiltering EEG between 8 and 30 Hz...")
    raw_filtered.filter(l_freq=8.0, h_freq=30.0, method='iir')

    # 8) Select only T1 and T2 event IDs and create epochs from 0 to 4 seconds
    if 'T1' not in event_id or 'T2' not in event_id:
        print("Required event labels T1 and T2 were not found in the event dictionary.")
        sys.exit(6)

    sel_event_id = {'T1': event_id['T1'], 'T2': event_id['T2']}
    print("Creating epochs for T1 and T2 from 0 to 4 seconds...")
    epochs = mne.Epochs(
        raw_filtered,
        events,
        event_id=sel_event_id,
        tmin=0.0,
        tmax=4.0,
        baseline=None,
        preload=True,
        picks='eeg',
    )

    # 9) Build X and y arrays suitable for machine learning
    #    X has shape (n_epochs, n_channels, n_timepoints)
    #    y contains binary labels: 0 for T1 (left), 1 for T2 (right).
    print("\nExtracting X and y arrays for machine learning...")
    X = epochs.get_data()
    y = np.array([0 if eid == sel_event_id['T1'] else 1 for eid in epochs.events[:, 2]], dtype=np.int64)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    print("Trial counts:")
    print(f"  T1 (left)  : {count_map.get(0, 0)}")
    print(f"  T2 (right) : {count_map.get(1, 0)}")

    # 10) Explanation of each step
    print("\nExplanation:")
    print("- Loaded the EDF file and printed dataset metadata.")
    print("- Extracted annotations and converted them into MNE events.")
    print("- Filtered the EEG between 8 and 30 Hz to isolate motor imagery bands.")
    print("- Created epochs around T1 and T2 events from 0 to 4 seconds.")
    print("- Selected only T1 and T2 trials and built X (epochs) and y (labels).")
    print("- Printed the resulting data shapes and trial counts.")
    print("\nDone.")


if __name__ == '__main__':
    main()
