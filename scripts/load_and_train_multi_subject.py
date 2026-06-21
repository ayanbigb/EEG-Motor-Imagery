"""
load_and_train_multi_subject.py

Load multiple EDF files, extract and concatenate T1/T2 epochs,
then train an SVM classifier on flattened feature vectors.

Files to load:
  - src/data/S001R04.edf
  - src/data/S001R08.edf
  - src/data/S001R12.edf

Steps:
1. Load each EDF file
2. Filter 8-30 Hz
3. Extract T1 and T2 events
4. Create epochs from 0 to 4 seconds
5. Extract X and y arrays
6. Concatenate across subjects
7. Flatten epochs into feature vectors
8. Split train/test
9. Train SVM
10. Evaluate and print results

Usage:
    source venv/bin/activate
    python scripts/load_and_train_multi_subject.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mne
except Exception:
    print("Error: MNE-Python not found. Install with: python -m pip install mne")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================================
# STEP 1: Load and process a single EDF file
# ============================================================================

def load_and_process_edf(edf_path):
    """
    Load an EDF file, filter, extract T1/T2 epochs, and return X, y.
    
    Returns:
        X (n_epochs, n_channels, n_timepoints)
        y (n_epochs,) with 0=T1 (left), 1=T2 (right)
    """
    # Check if file exists
    if not os.path.exists(edf_path):
        print(f"  WARNING: File not found: {edf_path}")
        return None, None

    # Load EDF
    print(f"  Loading {edf_path}...")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
    except Exception as e:
        print(f"  ERROR loading {edf_path}: {e}")
        return None, None

    # Extract events from annotations
    try:
        events, event_id = mne.events_from_annotations(raw)
    except Exception as e:
        print(f"  ERROR extracting events: {e}")
        return None, None

    # Check that T1 and T2 are present
    if 'T1' not in event_id or 'T2' not in event_id:
        print(f"  WARNING: T1 or T2 not found in event_id")
        return None, None

    # Filter 8-30 Hz
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=8.0, h_freq=30.0, method='iir', verbose='WARNING')

    # Create epochs for T1 and T2 from 0 to 4 seconds
    sel_event_id = {'T1': event_id['T1'], 'T2': event_id['T2']}
    try:
        epochs = mne.Epochs(
            raw_filtered,
            events,
            event_id=sel_event_id,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            preload=True,
            picks='eeg',
            verbose='WARNING',
        )
    except Exception as e:
        print(f"  ERROR creating epochs: {e}")
        return None, None

    # Extract X and y
    X = epochs.get_data()
    y = np.array(
        [0 if eid == sel_event_id['T1'] else 1 for eid in epochs.events[:, 2]],
        dtype=np.int64
    )

    n_t1 = np.sum(y == 0)
    n_t2 = np.sum(y == 1)
    print(f"    -> {X.shape[0]} epochs ({n_t1} T1, {n_t2} T2), shape {X.shape}")

    return X, y


# ============================================================================
# STEP 2: Load multiple files and concatenate
# ============================================================================

def load_multi_subject(file_paths):
    """
    Load multiple EDF files and concatenate X and y arrays.
    
    Args:
        file_paths: list of EDF file paths
        
    Returns:
        X_all (n_total_epochs, n_channels, n_timepoints)
        y_all (n_total_epochs,)
    """
    X_list = []
    y_list = []

    for fpath in file_paths:
        X, y = load_and_process_edf(fpath)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)

    if not X_list:
        raise ValueError("No data loaded from any file")

    # Concatenate
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


# ============================================================================
# STEP 3: Flatten epochs into feature vectors
# ============================================================================

def flatten_epochs(X):
    """
    Flatten each epoch into a single feature vector.
    
    Input shape: (n_epochs, n_channels, n_timepoints)
    Output shape: (n_epochs, n_channels * n_timepoints)
    """
    n_epochs, n_channels, n_times = X.shape
    X_flat = X.reshape(n_epochs, n_channels * n_times)
    return X_flat


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MULTI-SUBJECT EEG LOADING AND SVM TRAINING")
    print("=" * 70)
    print()

    # Define files to load
    files_to_load = [
        "src/data/S001R04.edf",
        "src/data/S001R08.edf",
        "src/data/S001R12.edf",
    ]

    # Step 1: Load and concatenate
    print("STEP 1: Loading and processing multiple EDF files")
    print("-" * 70)
    X_all, y_all = load_multi_subject(files_to_load)
    print()

    # Step 2: Print combined dataset info
    print("STEP 2: Combined dataset information")
    print("-" * 70)
    n_total = len(y_all)
    n_t1 = np.sum(y_all == 0)
    n_t2 = np.sum(y_all == 1)
    print(f"Combined X shape: {X_all.shape}")
    print(f"Combined y shape: {y_all.shape}")
    print(f"Total T1 (left) trials : {n_t1}")
    print(f"Total T2 (right) trials: {n_t2}")
    print()

    # Step 3: Flatten epochs
    print("STEP 3: Flattening epochs into feature vectors")
    print("-" * 70)
    X_flat = flatten_epochs(X_all)
    print(f"Flattened feature shape: {X_flat.shape}")
    print(f"  ({X_flat.shape[0]} samples, {X_flat.shape[1]} features)")
    print()

    # Step 4: Train/test split
    print("STEP 4: Splitting into train and test sets")
    print("-" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set : {X_test.shape[0]} samples")
    print()

    # Step 5: Train SVM
    print("STEP 5: Training SVM classifier")
    print("-" * 70)
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    print("SVM trained (linear kernel, C=1.0)")
    print()

    # Step 6: Evaluate
    print("STEP 6: Evaluating classifier")
    print("-" * 70)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['T1_left', 'T2_right']))
    print()

    # Step 7: Confusion matrix
    print("STEP 7: Confusion Matrix")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print()

    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['T1_left', 'T2_right'],
        yticklabels=['T1_left', 'T2_right'],
        cbar_kws={'label': 'Count'},
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.1f}%)')
    os.makedirs('figures', exist_ok=True)
    outpath = 'figures/confusion_matrix_multi_subject_svm.png'
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()
    print(f"Confusion matrix saved to {outpath}")
    print()

    # Step 8: Explanation
    print("=" * 70)
    print("STEP 8: Explanation of Results")
    print("=" * 70)
    print()
    print("What the accuracy means:")
    print(f"- We trained an SVM on {X_train.shape[0]} epochs from {len(files_to_load)} subjects.")
    print(f"- The model achieved {acc*100:.2f}% accuracy on {X_test.shape[0]} test epochs.")
    print()
    print("Interpretation:")
    if acc >= 0.80:
        print(f"  EXCELLENT: {acc*100:.1f}% accuracy indicates the SVM learned strong")
        print("  discriminative patterns between left (T1) and right (T2) motor imagery.")
    elif acc >= 0.70:
        print(f"  GOOD: {acc*100:.1f}% accuracy shows reasonable discrimination between")
        print("  left and right motor imagery with room for improvement.")
    elif acc >= 0.60:
        print(f"  MODERATE: {acc*100:.1f}% accuracy suggests some separability between")
        print("  classes, but performance is limited. Better features (CSP) or more data")
        print("  could improve results.")
    else:
        print(f"  POOR: {acc*100:.1f}% accuracy indicates weak discrimination. The flattened")
        print("  raw features may not be sufficient. Consider:")
        print("    - CSP (Common Spatial Patterns) for spatial feature extraction")
        print("    - Band power features instead of raw signal")
        print("    - Feature normalization or scaling")
        print("    - More epochs or subjects")
    print()
    print("Confusion Matrix Interpretation:")
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    print(f"  True Negatives (T1 predicted T1) : {tn}")
    print(f"  False Positives (T1 predicted T2): {fp}")
    print(f"  False Negatives (T2 predicted T1): {fn}")
    print(f"  True Positives (T2 predicted T2) : {tp}")
    print()
    if fp > fn:
        print("  → The model tends to predict T2 more often (bias towards right imagery)")
    elif fn > fp:
        print("  → The model tends to predict T1 more often (bias towards left imagery)")
    else:
        print("  → The model is balanced between T1 and T2 predictions")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
