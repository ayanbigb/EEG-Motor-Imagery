"""
Extract T1/T2 epochs and create left/right labels using MNE-Python.

Steps:
1. Load EDF file
2. Bandpass filter 8-30 Hz (motor imagery bands)
3. Extract events and select T1 and T2
4. Create labels: T1 -> left (0), T2 -> right (1)
5. Print shapes and counts
6. Save example epoch plots

Run:
    python scripts/extract_epochs_T1T2.py
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt


# -------------------------
# Configuration
# -------------------------
EDF_PATH = "src/data/S001R04.edf"
TMIN, TMAX = -0.5, 3.5  # epoch window (s)
LOW_FREQ, HIGH_FREQ = 8.0, 30.0
N_EXAMPLES = 3


def main():
    # 1) Load the raw EDF file into an MNE Raw object
    if not os.path.exists(EDF_PATH):
        raise FileNotFoundError(f"EDF file not found: {EDF_PATH}")
    print("Loading EDF file...")
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True)
    print(f"Loaded: {EDF_PATH} | {raw.info['nchan']} channels | sfreq={raw.info['sfreq']} Hz")

    # 2) Bandpass filter between 8 and 30 Hz
    #    This keeps the mu (8-12 Hz) and beta (15-30 Hz) bands relevant
    #    for motor imagery while removing slow drifts and high-frequency noise.
    print(f"Applying bandpass filter: {LOW_FREQ}-{HIGH_FREQ} Hz...")
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, method='iir')

    # 3) Extract annotations -> events and event_id mapping
    #    mne.events_from_annotations converts annotation descriptions into
    #    integer event IDs and an events array of shape (n_events, 3).
    print("Extracting events from annotations...")
    events, event_id = mne.events_from_annotations(raw_filtered)
    print("Event dictionary:", event_id)

    # 4) Select only T1 and T2 events
    #    Ensure both labels exist in the event_id mapping.
    if 'T1' not in event_id or 'T2' not in event_id:
        raise ValueError("Event labels T1 and/or T2 not found in annotations")
    sel_event_id = {'T1': event_id['T1'], 'T2': event_id['T2']}

    # 5) Create epochs around T1 and T2 (pre-stimulus baseline used)
    #    Epochs shape: (n_epochs, n_channels, n_times)
    print(f"Creating epochs: tmin={TMIN}, tmax={TMAX} (s) for T1/T2)")
    epochs = mne.Epochs(raw_filtered, events, event_id=sel_event_id, tmin=TMIN, tmax=TMAX,
                        baseline=(TMIN, 0), preload=True, picks='eeg')

    # 6) Build labels: map T1 -> 0 (left), T2 -> 1 (right)
    #    epochs.events[:, 2] contains the integer event IDs; map them to 0/1
    id_T1 = event_id['T1']
    id_T2 = event_id['T2']
    event_ids = epochs.events[:, 2]
    labels = np.array([0 if ev == id_T1 else 1 for ev in event_ids], dtype=np.int64)

    # 7) Show shapes and counts
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    print('\nResulting dataset shapes:')
    print(f"  X.shape = {X.shape}  (n_epochs, n_channels, n_times)")
    print(f"  y.shape = {labels.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {0: 'T1 (left)', 1: 'T2 (right)'}
    print('Class counts:')
    for u, c in zip(unique, counts):
        print(f"  {mapping[u]}: {c}")

    # 8) Plot a few example epochs from each class and save figures
    os.makedirs('figures', exist_ok=True)
    picks = [0, 1, 2]  # first three channels for compact plots

    # Helper to plot epochs for a class
    def plot_examples(class_label, outname):
        idxs = np.where(labels == class_label)[0][:N_EXAMPLES]
        if len(idxs) == 0:
            print(f"No examples for class {class_label} to plot")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        times = epochs.times
        offset = 0
        for i, ix in enumerate(idxs):
            data = X[ix, picks, :]
            # plot each picked channel with vertical offset
            for ch_i, ch_data in enumerate(data):
                ax.plot(times, ch_data + offset, label=f"trial{ix}_ch{picks[ch_i]}")
                offset += np.max(np.abs(data)) * 1.5
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Example epochs for {mapping[class_label]} (first {len(idxs)} trials)')
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        plt.tight_layout()
        fig.savefig(outname)
        plt.close(fig)
        print(f"Saved: {outname}")

    plot_examples(0, 'figures/examples_T1_left.png')
    plot_examples(1, 'figures/examples_T2_right.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
