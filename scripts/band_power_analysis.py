"""
band_power_analysis.py

Compute and visualize band power (8-12 Hz mu, 13-30 Hz beta) for motor imagery.
Compare T1 (left) vs T2 (right) for channels C3, Cz, C4.

Usage:
    source venv/bin/activate
    python scripts/band_power_analysis.py
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from scipy import signal

sns.set(style='whitegrid', font_scale=1.1)

DATA_DIR = 'src/data'
FIG_DIR = 'figures'
TARGET_CHANNELS = ['C3', 'Cz', 'C4']
SFREQ = 160  # Sampling frequency


def discover_edf_files(data_dir):
    """Find all EDF files matching R04/R08/R12 pattern."""
    pattern = os.path.join(data_dir, '*.edf')
    all_paths = glob.glob(pattern)
    matches = []
    file_re = re.compile(r'.*?(S\d{3})R(04|08|12)\.edf$', re.IGNORECASE)
    for path in sorted(all_paths):
        base = os.path.basename(path)
        m = file_re.match(base)
        if m:
            matches.append(path)
    return sorted(matches)


def load_epochs(path):
    """Load EDF and extract T1/T2 epochs."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    events, event_id = mne.events_from_annotations(raw)
    if 'T1' not in event_id or 'T2' not in event_id:
        return None, None, None

    raw_filtered = raw.copy()
    raw_filtered.filter(0.5, 50.0, method='iir', verbose='ERROR')  # Wider filter to preserve freq bands

    sel_id = {'T1': event_id['T1'], 'T2': event_id['T2']}
    epochs = mne.Epochs(raw_filtered, events, event_id=sel_id, tmin=0.0, tmax=4.0,
                        baseline=None, preload=True, picks='eeg', verbose='ERROR')
    X = epochs.get_data()
    y = np.array([0 if eid == sel_id['T1'] else 1 for eid in epochs.events[:, 2]], dtype=np.int64)
    return X, y, epochs.ch_names


def combine_epochs(paths):
    """Load and concatenate epochs from multiple files."""
    X_list, y_list = [], []
    for path in paths:
        X, y, ch_names = load_epochs(path)
        if X is None:
            continue
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError('No valid EDF files found.')

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all, ch_names


def find_channel_indices(ch_names, targets):
    """Find channel indices by name prefix."""
    indices = {}
    for target in targets:
        for i, name in enumerate(ch_names):
            if name.upper().startswith(target.upper()):
                indices[target] = i
                break
        if target not in indices:
            raise ValueError(f"Channel not found: {target}")
    return indices


def compute_band_power(X, sfreq, freq_range=(8, 12)):
    """
    Compute band power using Welch's method.
    
    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_samples)
    sfreq : float
        Sampling frequency
    freq_range : tuple
        (low, high) frequency bounds
    
    Returns
    -------
    band_power : ndarray, shape (n_epochs, n_channels)
        Average power in band for each epoch and channel
    """
    n_epochs, n_channels, n_samples = X.shape
    band_power = np.zeros((n_epochs, n_channels))
    
    for ep in range(n_epochs):
        for ch in range(n_channels):
            signal_ch = X[ep, ch, :]
            freqs, psd = signal.welch(signal_ch, sfreq, nperseg=256)
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            band_power[ep, ch] = np.mean(psd[mask])
    
    return band_power


def plot_band_power_comparison(X, y, ch_names, band_name, freq_range, outpath):
    """Plot bar chart comparing T1 vs T2 band power for target channels."""
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    band_power = compute_band_power(X, SFREQ, freq_range)
    
    t1_power = band_power[y == 0]  # (n_t1, n_channels)
    t2_power = band_power[y == 1]  # (n_t2, n_channels)
    
    t1_mean = np.mean(t1_power, axis=0)
    t2_mean = np.mean(t2_power, axis=0)
    t1_std = np.std(t1_power, axis=0)
    t2_std = np.std(t2_power, axis=0)
    
    # Extract target channels
    x_pos = np.arange(len(TARGET_CHANNELS))
    width = 0.35
    
    t1_vals = [t1_mean[idx_map[ch]] for ch in TARGET_CHANNELS]
    t2_vals = [t2_mean[idx_map[ch]] for ch in TARGET_CHANNELS]
    t1_errs = [t1_std[idx_map[ch]] for ch in TARGET_CHANNELS]
    t2_errs = [t2_std[idx_map[ch]] for ch in TARGET_CHANNELS]
    
    plt.figure(figsize=(8, 5))
    plt.bar(x_pos - width/2, t1_vals, width, label='T1 (left)', color='#4c72b0', yerr=t1_errs, capsize=5)
    plt.bar(x_pos + width/2, t2_vals, width, label='T2 (right)', color='#dd8452', yerr=t2_errs, capsize=5)
    plt.xlabel('Channel')
    plt.ylabel('Power (V²/Hz)')
    plt.title(f'{band_name} band ({freq_range[0]}-{freq_range[1]} Hz) power')
    plt.xticks(x_pos, TARGET_CHANNELS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    
    return t1_vals, t2_vals, t1_errs, t2_errs, t1_mean, t2_mean, idx_map


def plot_band_power_combined(X, y, ch_names, outpath):
    """Plot mu and beta bands side-by-side for comparison."""
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    
    mu_power = compute_band_power(X, SFREQ, (8, 12))
    beta_power = compute_band_power(X, SFREQ, (13, 30))
    
    t1_mu = np.mean(mu_power[y == 0], axis=0)
    t2_mu = np.mean(mu_power[y == 1], axis=0)
    t1_beta = np.mean(beta_power[y == 0], axis=0)
    t2_beta = np.mean(beta_power[y == 1], axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_pos = np.arange(len(TARGET_CHANNELS))
    width = 0.35
    
    # Mu band
    t1_mu_vals = [t1_mu[idx_map[ch]] for ch in TARGET_CHANNELS]
    t2_mu_vals = [t2_mu[idx_map[ch]] for ch in TARGET_CHANNELS]
    axes[0].bar(x_pos - width/2, t1_mu_vals, width, label='T1 (left)', color='#4c72b0')
    axes[0].bar(x_pos + width/2, t2_mu_vals, width, label='T2 (right)', color='#dd8452')
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Power (V²/Hz)')
    axes[0].set_title('Mu band (8-12 Hz)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(TARGET_CHANNELS)
    axes[0].legend()
    
    # Beta band
    t1_beta_vals = [t1_beta[idx_map[ch]] for ch in TARGET_CHANNELS]
    t2_beta_vals = [t2_beta[idx_map[ch]] for ch in TARGET_CHANNELS]
    axes[1].bar(x_pos - width/2, t1_beta_vals, width, label='T1 (left)', color='#4c72b0')
    axes[1].bar(x_pos + width/2, t2_beta_vals, width, label='T2 (right)', color='#dd8452')
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Power (V²/Hz)')
    axes[1].set_title('Beta band (13-30 Hz)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(TARGET_CHANNELS)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_power_difference_heatmap(X, y, ch_names, outpath):
    """
    Create a heatmap of (T1 - T2) power differences across frequency bands.
    """
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    
    # Compute PSD for each epoch
    from mne.time_frequency import psd_array_welch
    
    psd_t1, freqs = psd_array_welch(X[y == 0], sfreq=SFREQ, n_per_seg=256, 
                                     n_overlap=128, verbose=False)
    psd_t2, _ = psd_array_welch(X[y == 1], sfreq=SFREQ, n_per_seg=256, 
                                 n_overlap=128, verbose=False)
    
    # Average PSDs (mean over epochs)
    psd_t1_mean = np.mean(psd_t1, axis=0)  # (n_channels, n_freqs)
    psd_t2_mean = np.mean(psd_t2, axis=0)
    
    # Extract target channels in frequency range 1-40 Hz
    freq_mask = (freqs >= 1) & (freqs <= 40)
    freqs_sub = freqs[freq_mask]
    
    ch_indices = [idx_map[ch] for ch in TARGET_CHANNELS]
    psd_diff = psd_t1_mean[ch_indices][:, freq_mask] - psd_t2_mean[ch_indices][:, freq_mask]
    
    plt.figure(figsize=(10, 4))
    im = plt.imshow(psd_diff, aspect='auto', cmap='RdBu_r', origin='lower',
                    extent=[freqs_sub[0], freqs_sub[-1], 0, len(TARGET_CHANNELS)-1])
    plt.colorbar(im, label='PSD diff (T1 - T2) [V²/Hz]')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channel')
    plt.yticks(range(len(TARGET_CHANNELS)), TARGET_CHANNELS)
    plt.title('Power difference heatmap (T1 - T2)')
    plt.axvline(8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Mu band')
    plt.axvline(12, color='green', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(13, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Beta band')
    plt.axvline(30, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    
    files = discover_edf_files(DATA_DIR)
    print(f'Found {len(files)} EDF files')
    if not files:
        print('No EDF files found. Exiting.')
        sys.exit(1)
    
    print('Loading and preprocessing EEG data...')
    X_all, y_all, ch_names = combine_epochs(files)
    print(f'Loaded {X_all.shape[0]} epochs, {X_all.shape[1]} channels')
    print(f'T1 (left): {int(np.sum(y_all == 0))} trials')
    print(f'T2 (right): {int(np.sum(y_all == 1))} trials')
    
    # Compute band powers
    print('\nComputing band powers...')
    idx_map = find_channel_indices(ch_names, TARGET_CHANNELS)
    
    # Mu band
    print('  Mu band (8-12 Hz)...')
    t1_mu, t2_mu, t1_mu_err, t2_mu_err, t1_mu_all, t2_mu_all, _ = plot_band_power_comparison(
        X_all, y_all, ch_names, 'Mu', (8, 12), os.path.join(FIG_DIR, 'band_power_mu.png')
    )
    
    # Beta band
    print('  Beta band (13-30 Hz)...')
    t1_beta, t2_beta, t1_beta_err, t2_beta_err, t1_beta_all, t2_beta_all, _ = plot_band_power_comparison(
        X_all, y_all, ch_names, 'Beta', (13, 30), os.path.join(FIG_DIR, 'band_power_beta.png')
    )
    
    # Combined comparison
    print('  Creating combined visualization...')
    plot_band_power_combined(X_all, y_all, ch_names, 
                            os.path.join(FIG_DIR, 'band_power_combined.png'))
    
    # Heatmap
    print('  Creating power difference heatmap...')
    plot_power_difference_heatmap(X_all, y_all, ch_names,
                                 os.path.join(FIG_DIR, 'power_difference_heatmap.png'))
    
    # Print detailed statistics
    print('\n' + '='*60)
    print('BAND POWER ANALYSIS SUMMARY')
    print('='*60)
    
    print('\nMU BAND (8-12 Hz):')
    print('-' * 40)
    for i, ch in enumerate(TARGET_CHANNELS):
        t1_val = t1_mu[i]
        t2_val = t2_mu[i]
        diff = ((t2_val - t1_val) / t1_val * 100) if t1_val != 0 else 0
        print(f'{ch:3s}  T1: {t1_val:.6e} ± {t1_mu_err[i]:.6e}  '
              f'T2: {t2_val:.6e} ± {t2_mu_err[i]:.6e}  '
              f'Change: {diff:+.1f}%')
    
    print('\nBETA BAND (13-30 Hz):')
    print('-' * 40)
    for i, ch in enumerate(TARGET_CHANNELS):
        t1_val = t1_beta[i]
        t2_val = t2_beta[i]
        diff = ((t2_val - t1_val) / t1_val * 100) if t1_val != 0 else 0
        print(f'{ch:3s}  T1: {t1_val:.6e} ± {t1_beta_err[i]:.6e}  '
              f'T2: {t2_val:.6e} ± {t2_beta_err[i]:.6e}  '
              f'Change: {diff:+.1f}%')
    
    print('\n' + '='*60)
    print('NEUROSCIENCE INTERPRETATION')
    print('='*60)
    print('''
Motor imagery studies typically show:
  - Mu desynchronization (8-12 Hz power DECREASE) over contralateral motor cortex
    during movement/imagery (event-related desynchronization, ERD)
  - C3 over left motor cortex: should show mu DECREASE during RIGHT hand (T2) imagery
  - C4 over right motor cortex: should show mu DECREASE during LEFT hand (T1) imagery
  - Cz (midline): shows general motor effects

If T1 (left) shows HIGHER mu power than T2 (right):
  - C3 channel: suggests stronger desynchronization for T2 (right imagery)
    ✓ This is EXPECTED (right imagery → left motor cortex desync → C3 shows it)
  - C4 channel: suggests stronger desynchronization for T2 (right imagery)
    ✓ This is EXPECTED (right imagery → right motor cortex desync → C4 shows it)

If T1 (left) shows LOWER mu power than T2 (right):
  - C3 channel: suggests stronger desynchronization for T1 (left imagery)
    ✓ This is EXPECTED (left imagery → left motor cortex desync → C3 shows it)
  - C4 channel: suggests stronger desynchronization for T1 (left imagery)
    ✓ This is EXPECTED (left imagery → right motor cortex desync → C4 shows it)

BETA BAND EFFECTS (13-30 Hz):
  - Similar lateralization as mu band, but typically weaker
  - Often more sustained during movement/imagery (event-related synchronization, ERS)
  - Can reflect motor planning and execution preparation
    ''')
    
    print('\nFigures saved to figures/:')
    print('  - band_power_mu.png')
    print('  - band_power_beta.png')
    print('  - band_power_combined.png')
    print('  - power_difference_heatmap.png')


if __name__ == '__main__':
    main()
