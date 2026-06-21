"""
Train and evaluate an SVM on T1 vs T2 motor imagery using simple features.

Steps:
1. Load preprocessed epochs from `data/preprocessed_data.npz` (created by preprocessing)
2. Select epochs corresponding to T1 and T2
3. Compute log-bandpower features per channel (log of mean squared amplitude)
4. Split into train/test sets
5. Train linear SVM and evaluate accuracy
6. Plot and save confusion matrix

Run:
    source venv/bin/activate
    python scripts/train_svm_T1T2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_PATH = "data/preprocessed_data.npz"
TEST_SIZE = 0.3
RANDOM_STATE = 42


def load_T1_T2(npz_path):
    # Load saved preprocessed arrays
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Preprocessed data not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_all = data["X"]  # shape (n_epochs, channels, times)
    y_all = data["y"]  # event ids
    event_id = data["event_id"].item() if isinstance(data["event_id"].item(), dict) else data["event_id"].item()

    # Find event IDs for T1 and T2
    # event_id is typically a dict like {'T0':1, 'T1':2, 'T2':3}
    id_T1 = event_id.get('T1')
    id_T2 = event_id.get('T2')
    if id_T1 is None or id_T2 is None:
        raise ValueError("T1 and/or T2 not present in event_id mapping")

    sel_mask = (y_all == id_T1) | (y_all == id_T2)
    X = X_all[sel_mask]
    y = y_all[sel_mask]
    # Map to binary labels: T1 -> 0 (left), T2 -> 1 (right)
    y_bin = np.array([0 if v == id_T1 else 1 for v in y], dtype=np.int64)
    return X, y_bin


def compute_log_bandpower(X):
    # X shape: (n_epochs, channels, times)
    # Feature: log of mean squared amplitude per channel (simple bandpower proxy)
    power = np.mean(X ** 2, axis=2)  # (n_epochs, channels)
    # numerical stability
    eps = 1e-12
    logbp = np.log(power + eps)
    return logbp


def main():
    print("Loading preprocessed data and selecting T1/T2 epochs...")
    X_epochs, y = load_T1_T2(DATA_PATH)
    print(f"Selected epochs shape: {X_epochs.shape}")

    # Compute features
    print("Computing log-bandpower features...")
    X_feat = compute_log_bandpower(X_epochs)
    print(f"Feature matrix shape: {X_feat.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train/test sizes: {X_train.shape[0]} / {X_test.shape[0]}")

    # Train a linear SVM
    print("Training linear SVM...")
    clf = SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc*100:.2f}%")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=['T1_left','T2_right']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['T1_left','T2_right'], yticklabels=['T1_left','T2_right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (acc={acc*100:.1f}%)')
    os.makedirs('figures', exist_ok=True)
    outpath = 'figures/confusion_matrix_svm.png'
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved confusion matrix to {outpath}")

    # Explain results briefly
    print('\nExplanation:')
    print('- Features: log-bandpower per channel (simple, interpretable features).')
    print('- Classifier: linear SVM trained on these features. Performance depends on SNR and number of trials.')
    print(f'- With {X_feat.shape[0]} total trials (train/test split {X_train.shape[0]}/{X_test.shape[0]}), the SVM achieved {acc*100:.2f}% test accuracy.')
    print('- Confusion matrix saved; inspect which class is confused for improvement ideas (CSP, feature selection, more trials).')


if __name__ == '__main__':
    main()
