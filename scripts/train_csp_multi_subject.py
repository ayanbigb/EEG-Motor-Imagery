"""
Train CSP-based classifier on multiple PhysioNet motor imagery EDFs.

Loads S001R04, S001R08, S001R12, extracts T1/T2 epochs (0-4s), filters 8-30Hz,
concatenates data, fits an sklearn Pipeline: CSP -> StandardScaler -> LogisticRegression/SVM,
evaluates and saves a confusion matrix, and compares to baseline 50%.

Usage:
    source venv/bin/activate
    python scripts/train_csp_multi_subject.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mne
    from mne.decoding import CSP
except Exception:
    print("MNE or mne.decoding.CSP not available. Install mne: python -m pip install mne")
    raise

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_and_process_edf(edf_path):
    """Load EDF, filter, extract T1/T2 epochs, return X,y (epochs form).

    X: (n_epochs, n_channels, n_times)
    y: (n_epochs,) with 0=T1, 1=T2
    """
    if not os.path.exists(edf_path):
        print(f"WARNING: {edf_path} not found")
        return None, None

    print(f"Loading {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')

    events, event_id = mne.events_from_annotations(raw)
    if 'T1' not in event_id or 'T2' not in event_id:
        print(f"WARNING: T1/T2 not present in {edf_path}")
        return None, None

    # Filter in band
    raw_f = raw.copy()
    raw_f.filter(8.0, 30.0, method='iir', verbose='WARNING')

    sel = {'T1': event_id['T1'], 'T2': event_id['T2']}
    epochs = mne.Epochs(raw_f, events, event_id=sel, tmin=0.0, tmax=4.0,
                        baseline=None, preload=True, picks='eeg', verbose='WARNING')

    X = epochs.get_data()
    y = np.array([0 if eid == sel['T1'] else 1 for eid in epochs.events[:, 2]], dtype=np.int64)
    n_t1 = int(np.sum(y == 0))
    n_t2 = int(np.sum(y == 1))
    print(f"  -> {X.shape[0]} epochs ({n_t1} T1, {n_t2} T2) from {os.path.basename(edf_path)}")
    return X, y


def load_files(file_list):
    Xs, ys = [], []
    for f in file_list:
        X, y = load_and_process_edf(f)
        if X is None:
            continue
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise RuntimeError("No data loaded from files")
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return X_all, y_all


def plot_confusion(cm, outpath, acc):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['T1_left','T2_right'], yticklabels=['T1_left','T2_right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (CSP model acc={acc*100:.1f}%)')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(outpath, dpi=100)
    plt.close()


def main():
    files = [
        'src/data/S001R04.edf',
        'src/data/S001R08.edf',
        'src/data/S001R12.edf',
    ]

    print('Loading and concatenating epochs from files...')
    X_all, y_all = load_files(files)
    print(f'Combined X shape: {X_all.shape}')
    print(f'Combined y shape: {y_all.shape}')
    print(f'Total T1: {int(np.sum(y_all==0))}, Total T2: {int(np.sum(y_all==1))}')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)

    # CSP transformer (use 6 components; common choice: 4-6)
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

    # Build pipeline: CSP -> scaler -> classifier
    clf = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([
        ('csp', csp),
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])

    print('\nFitting CSP + LogisticRegression pipeline...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'CSP model Test Accuracy: {acc*100:.2f}%')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=['T1_left','T2_right']))

    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion matrix:')
    print(cm)

    outpath = 'figures/confusion_matrix_csp.png'
    plot_confusion(cm, outpath, acc)
    print(f'Saved confusion matrix to {outpath}')

    # Compare to baseline
    baseline = 0.50
    diff = (acc - baseline)
    print('\nComparison to baseline (flattened-feature SVM: 50%):')
    print(f'  CSP model accuracy: {acc*100:.2f}%')
    print(f'  Baseline accuracy  : {baseline*100:.2f}%')
    if diff > 0:
        print(f'  Improvement: {diff*100:.2f} percentage points')
    elif diff < 0:
        print(f'  Worse by: {-diff*100:.2f} percentage points')
    else:
        print('  No change')


if __name__ == '__main__':
    main()
