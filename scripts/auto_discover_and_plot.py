"""
auto_discover_and_plot.py

Discover EDF files automatically in src/data, load runs R04/R08/R12,
concatenate epochs, and visualize channel averages and PSD differences for
C3, Cz, and C4.

Usage:
    source venv/bin/activate
    python scripts/auto_discover_and_plot.py
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from mne.time_frequency import psd_array_welch

sns.set(style='whitegrid', font_scale=1.1)


DATA_DIR = 'src/data'
FIG_DIR = 'figures'
VALID_RUNS = {'R04', 'R08', 'R12'}
TARGET_CHANNELS = ['C3', 'Cz', 'C4']


def discover_edf_files(data_dir):
    pattern = os.path.join(data_dir, '*.edf')
    all_paths = glob.glob(pattern)
    matches = []
    subject_ids = []
    file_re = re.compile(r'.*?(S\d{3})R(04|08|12)\.edf$', re.IGNORECASE)
    for path in sorted(all_paths):
        base = os.path.basename(path)
        m = file_re.match(base)
        if m:
            subject_ids.append(m.group(1))
            matches.append(path)
    return sorted(matches), sorted(set(subject_ids))


def find_channel_indices(ch_names, targets):
    indices = {}
    for target in targets:
        for i, name in enumerate(ch_names):
            if name.upper().startswith(target.upper()):
                indices[target] = i
                break
        if target not in indices:
            raise ValueError(f"Channel target not found: {target}")
    return indices


def load_epochs(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    events, event_id = mne.events_from_annotations(raw)
    if 'T1' not in event_id or 'T2' not in event_id:
        print(f"Skipping {path}: missing T1/T2 annotations")
        return None, None, None

    raw_filtered = raw.copy()
    raw_filtered.filter(8.0, 30.0, method='iir', verbose='ERROR')

    sel_id = {'T1': event_id['T1'], 'T2': event_id['T2']}
    epochs = mne.Epochs(raw_filtered, events, event_id=sel_id, tmin=0.0, tmax=4.0,
                        baseline=None, preload=True, picks='eeg', verbose='ERROR')
    X = epochs.get_data()
    y = np.array([0 if eid == sel_id['T1'] else 1 for eid in epochs.events[:, 2]], dtype=np.int64)
    return X, y, epochs.ch_names


def combine_epochs(paths):
    X_list = []
    y_list = []
    all_subjects = set()
    for path in paths:
        subject = re.search(r'(S\d{3})', os.path.basename(path), re.IGNORECASE)
        if subject:
            all_subjects.add(subject.group(1))
        X, y, ch_names = load_epochs(path)
        if X is None:
            continue
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError('No valid EDF files found.')

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all, ch_names, sorted(all_subjects)


def plot_class_balance(y, outpath):
    counts = np.bincount(y)
    labels = ['T1 (left)', 'T2 (right)']
    plt.figure(figsize=(5, 3))
    sns.barplot(x=labels, y=counts, palette=['#4c72b0', '#dd8452'])
    plt.ylabel('Number of trials')
    plt.title('T1 vs T2 class balance')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_mean_signals(X, y, ch_names, outpath):
    # Convert from volts to microvolts for readability
    X_uv = X * 1e6
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    times = np.linspace(0, 4.0, X_uv.shape[2])
    t1_mean = np.mean(X_uv[y == 0], axis=0)
    t2_mean = np.mean(X_uv[y == 1], axis=0)

    plt.figure(figsize=(10, 8))
    for i, ch in enumerate(TARGET_CHANNELS, 1):
        idx = idx_map[ch]
        ax = plt.subplot(3, 1, i)
        ax.plot(times, t1_mean[idx], label='T1 left', color='#4c72b0')
        ax.plot(times, t2_mean[idx], label='T2 right', color='#dd8452')
        ax.set_title(f'{ch} mean EEG signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return t1_mean, t2_mean


def plot_mean_differences(t1_mean, t2_mean, ch_names, outpath):
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    times = np.linspace(0, 4.0, t1_mean.shape[1])

    plt.figure(figsize=(10, 8))
    for i, ch in enumerate(TARGET_CHANNELS, 1):
        idx = idx_map[ch]
        ax = plt.subplot(3, 1, i)
        diff = t1_mean[idx] - t2_mean[idx]
        ax.plot(times, diff, color='#2ca02c')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_title(f'{ch} T1 minus T2 difference')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude diff')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def compute_psd_diff(X, y, channel_idx, sfreq=160):
    psd_t1, freqs = psd_array_welch(X[y == 0, channel_idx, :], sfreq=sfreq,
                                    n_per_seg=256, n_overlap=128, verbose=False)
    psd_t2, _ = psd_array_welch(X[y == 1, channel_idx, :], sfreq=sfreq,
                                n_per_seg=256, n_overlap=128, verbose=False)
    return freqs, np.mean(psd_t1, axis=0), np.mean(psd_t2, axis=0)


def plot_psd_channels(X, y, ch_names, outpath):
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    freqs, _, _ = compute_psd_diff(X, y, idx_map[TARGET_CHANNELS[0]])

    plt.figure(figsize=(10, 8))
    for i, ch in enumerate(TARGET_CHANNELS, 1):
        idx = idx_map[ch]
        freqs, p1, p2 = compute_psd_diff(X, y, idx)
        ax = plt.subplot(3, 1, i)
        ax.plot(freqs, 10 * np.log10(p1), label='T1 left', color='#4c72b0')
        ax.plot(freqs, 10 * np.log10(p2), label='T2 right', color='#dd8452')
        ax.fill_between(freqs, -100, 100, where=(freqs >= 8) & (freqs <= 12), color='gray', alpha=0.15,
                                label='Mu 8-12 Hz' if i == 1 else None)
        ax.fill_between(freqs, -100, 100, where=(freqs >= 13) & (freqs <= 30), color='orange', alpha=0.10,
                                label='Beta 13-30 Hz' if i == 1 else None)
        ax.set_xlim(1, 40)
        ax.set_title(f'PSD for {ch}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def summarize_channel_differences(t1_mean, t2_mean, ch_names):
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    diffs = {}
    print('\n=== DEBUG: Channel Differences ===')
    print(f't1_mean shape: {t1_mean.shape}')
    print(f't2_mean shape: {t2_mean.shape}')
    for ch in TARGET_CHANNELS:
        idx = idx_map[ch]
        t1_signal = t1_mean[idx]
        t2_signal = t2_mean[idx]
        pointwise_diff = np.abs(t1_signal - t2_signal)
        diff = np.mean(pointwise_diff)
        diffs[ch] = diff
        print(f'\n{ch} (idx={idx}):')
        print(f'  T1 mean signal min/max: {np.min(t1_signal):.6f} / {np.max(t1_signal):.6f}')
        print(f'  T2 mean signal min/max: {np.min(t2_signal):.6f} / {np.max(t2_signal):.6f}')
        print(f'  Pointwise diff min/max: {np.min(pointwise_diff):.6f} / {np.max(pointwise_diff):.6f}')
        print(f'  Mean absolute difference: {diff:.8f}')
    sorted_channels = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    return sorted_channels


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    files, subjects = discover_edf_files(DATA_DIR)
    print(f'Found {len(files)} EDF files matching runs R04/R08/R12')
    print(f'Subject IDs: {subjects}')
    print(f'Number of subjects: {len(subjects)}')

    if not files:
        print('No matching EDF files found. Exiting.')
        sys.exit(1)

    X_all, y_all, ch_names, _ = combine_epochs(files)
    total_trials = len(y_all)
    t1_count = int(np.sum(y_all == 0))
    t2_count = int(np.sum(y_all == 1))

    print(f'\n=== Dataset Summary ===')
    print(f'Number of EDF files: {len(files)}')
    print(f'Data shape X_all: {X_all.shape} (epochs, channels, timepoints)')
    print(f'Total trials: {total_trials}')
    print(f'Total T1 trials (left imagery): {t1_count}')
    print(f'Total T2 trials (right imagery): {t2_count}')
    print(f'Number of channels: {len(ch_names)}')
    print(f'Channel names: {ch_names}')

    plot_class_balance(y_all, os.path.join(FIG_DIR, 'class_balance.png'))

    t1_mean, t2_mean = plot_mean_signals(X_all, y_all, ch_names,
                                         os.path.join(FIG_DIR, 'mean_signals_C3_Cz_C4.png'))
    plot_mean_differences(t1_mean, t2_mean, ch_names,
                          os.path.join(FIG_DIR, 'difference_signals_C3_Cz_C4.png'))

    channel_order = summarize_channel_differences(t1_mean, t2_mean, ch_names)
    print('\nChannel differences ranked (mean absolute difference):')
    for ch, score in channel_order:
        print(f'  {ch}: {score:.4f}')

    plot_psd_channels(X_all, y_all, ch_names,
                      os.path.join(FIG_DIR, 'psd_C3_Cz_C4.png'))

    print('\nFigures saved to figures/')
    print('  - class_balance.png')
    print('  - mean_signals_C3_Cz_C4.png')
    print('  - difference_signals_C3_Cz_C4.png')
    print('  - psd_C3_Cz_C4.png')
    print('\nInterpretation:')
    print('  The mean EEG plots show how T1 and T2 differ over time in C3, Cz, and C4.')
    print('  The PSD plots highlight power differences in the mu (8-12 Hz) and beta (13-30 Hz) bands.')
    print('  The strongest differences are likely in the channel with the largest')
    print('  mean absolute waveform difference, especially over motor cortex electrodes.')


if __name__ == '__main__':
    main()
