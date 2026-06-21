"""
visualize_and_cv.py

Create publication-style figures and run 5-fold stratified cross-validation
for CSP-based and baseline flattened-feature classifiers on combined EDFs.

Outputs saved to `figures/`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mne
    from mne.decoding import CSP
except Exception:
    print("MNE not available. Install with: python -m pip install mne")
    raise

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


FILES = [
    'src/data/S001R04.edf',
    'src/data/S001R08.edf',
    'src/data/S001R12.edf',
]


def load_and_process_edf(edf_path):
    if not os.path.exists(edf_path):
        return None, None, None
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    events, event_id = mne.events_from_annotations(raw)
    if 'T1' not in event_id or 'T2' not in event_id:
        return None, None, None
    raw_f = raw.copy()
    raw_f.filter(8.0, 30.0, method='iir', verbose='ERROR')
    sel = {'T1': event_id['T1'], 'T2': event_id['T2']}
    epochs = mne.Epochs(raw_f, events, event_id=sel, tmin=0.0, tmax=4.0,
                        baseline=None, preload=True, picks='eeg', verbose='ERROR')
    X = epochs.get_data()
    y = np.array([0 if eid == sel['T1'] else 1 for eid in epochs.events[:, 2]], dtype=np.int64)
    ch_names = epochs.ch_names
    return X, y, ch_names


def load_all(files):
    Xs, ys = [], []
    ch_names = None
    for f in files:
        X, y, ch = load_and_process_edf(f)
        if X is None:
            print(f"Skipping {f}")
            continue
        Xs.append(X)
        ys.append(y)
        if ch_names is None:
            ch_names = ch
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return X_all, y_all, ch_names


def plot_class_balance(y, outpath):
    counts = np.bincount(y)
    labels = ['T1 (left)', 'T2 (right)']
    plt.figure(figsize=(4,3))
    sns.barplot(x=labels, y=counts, palette='muted')
    plt.ylabel('Number of trials')
    plt.title('Class balance')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_eeg_examples(X, ch_names, outpath, n_epochs=3, channels=[0,10,20,30]):
    # plot first n_epochs for selected channels
    times = np.linspace(0, 4.0, X.shape[2])
    plt.figure(figsize=(10, 6))
    offset = 0
    for ei in range(n_epochs):
        for ci, ch in enumerate(channels):
            data = X[ei, ch, :]
            plt.plot(times, data + offset, label=f'epoch{ei}_ch{ch_names[ch]}')
            offset += np.max(np.abs(data)) * 2.5
    plt.xlabel('Time (s)')
    plt.title('EEG examples (selected channels)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_csp_components(csp, ch_names, outpath, n_components=6):
    # csp.patterns_ shape: (n_components, n_channels)
    pats = csp.patterns_[:n_components]
    nrows = int(np.ceil(n_components / 2))
    plt.figure(figsize=(10, nrows*2.5))
    for i, pat in enumerate(pats):
        ax = plt.subplot(nrows, 2, i+1)
        ax.bar(range(len(pat)), pat)
        ax.set_xticks(range(len(pat))[::4])
        ax.set_xticklabels([ch_names[i] for i in range(0, len(pat), 4)], rotation=45)
        ax.set_title(f'CSP pattern {i+1}')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_cv_and_report(X, y, pipeline, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_acc = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_acc.append(acc)
        print(f'  Fold {fold} accuracy: {acc*100:.2f}%')
    mean = np.mean(fold_acc)
    std = np.std(fold_acc)
    return fold_acc, mean, std


def main():
    os.makedirs('figures', exist_ok=True)
    X_all, y_all, ch_names = load_all(FILES)
    print(f'Loaded combined X shape: {X_all.shape}, y shape: {y_all.shape}')

    # 1. Class balance
    plot_class_balance(y_all, 'figures/class_balance.png')

    # 2. EEG examples
    plot_eeg_examples(X_all, ch_names, 'figures/eeg_examples.png')

    # 3. CSP model (single-split and CV)
    csp = CSP(n_components=6, log=True)
    pipeline_csp = Pipeline([
        ('csp', csp),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

    # Single train/test split (stratified)
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    pipeline_csp.fit(Xtr, ytr)
    ypred = pipeline_csp.predict(Xte)
    acc_single = accuracy_score(yte, ypred)
    cm_single = confusion_matrix(yte, ypred)
    plot_confusion = lambda cm, out: (plt.figure(), sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['T1','T2'], yticklabels=['T1','T2']), plt.title('CSP single-split conf matrix'), plt.savefig(out), plt.close())
    plot_confusion(cm_single, 'figures/confusion_csp_single.png')

    print('\nCSP single-split accuracy: {:.2f}%'.format(acc_single*100))

    # 4. 5-fold stratified CV for CSP
    print('\nRunning 5-fold stratified CV for CSP pipeline...')
    fold_acc, mean_acc, std_acc = run_cv_and_report(X_all, y_all, pipeline_csp, n_splits=5)
    print(f'CV mean accuracy: {mean_acc*100:.2f}%, std: {std_acc*100:.2f}%')

    # 5. Baseline flattened SVM pipeline
    n_epochs, n_channels, n_times = X_all.shape
    X_flat = X_all.reshape(n_epochs, n_channels * n_times)
    pipeline_base = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear')),
    ])

    # single-split baseline
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_flat, y_all, test_size=0.3, random_state=42, stratify=y_all)
    pipeline_base.fit(Xb_tr, yb_tr)
    yb_pred = pipeline_base.predict(Xb_te)
    acc_base_single = accuracy_score(yb_te, yb_pred)
    cm_base_single = confusion_matrix(yb_te, yb_pred)
    plt.figure(); sns.heatmap(cm_base_single, annot=True, fmt='d', cmap='Blues', xticklabels=['T1','T2'], yticklabels=['T1','T2']); plt.title('Baseline single-split conf matrix'); plt.tight_layout(); plt.savefig('figures/confusion_baseline_single.png'); plt.close()

    # 5-fold CV baseline
    print('\nRunning 5-fold stratified CV for baseline pipeline...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_fold_acc = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X_flat, y_all), 1):
        pipeline_base.fit(X_flat[train_idx], y_all[train_idx])
        ypred = pipeline_base.predict(X_flat[test_idx])
        acc = accuracy_score(y_all[test_idx], ypred)
        base_fold_acc.append(acc)
        print(f'  Baseline fold {i} acc: {acc*100:.2f}%')
    base_mean = np.mean(base_fold_acc)
    base_std = np.std(base_fold_acc)

    # 6. Accuracy comparison plot
    labels = ['Baseline single', 'Baseline CV mean', 'CSP single', 'CSP CV mean']
    vals = [acc_base_single*100, base_mean*100, acc_single*100, mean_acc*100]
    errs = [0, base_std*100, 0, std_acc*100]
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=vals, palette='muted')
    for i, e in enumerate(errs):
        if e > 0:
            plt.errorbar(i, vals[i], yerr=e, color='k', capsize=5)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy comparison')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('figures/accuracy_comparison.png', dpi=150)
    plt.close()

    # 7. Save CSP component visualizations
    # Fit CSP on full data to visualize patterns
    csp_vis = CSP(n_components=6, log=True)
    csp_vis.fit(X_all, y_all)
    plot_csp_components(csp_vis, ch_names, 'figures/csp_components.png', n_components=6)

    # 8. Print summary
    print('\nSummary:')
    print(f'  CSP single-split acc: {acc_single*100:.2f}%')
    print(f'  CSP 5-fold CV mean:   {mean_acc*100:.2f}% ± {std_acc*100:.2f}%')
    print(f'  Baseline single-split acc: {acc_base_single*100:.2f}%')
    print(f'  Baseline 5-fold CV mean:   {base_mean*100:.2f}% ± {base_std*100:.2f}%')
    print('\nFold accuracies (CSP):', [f'{a*100:.2f}%' for a in fold_acc])
    print('Fold accuracies (Baseline):', [f'{a*100:.2f}%' for a in base_fold_acc])

    # Advice on more data
    print('\nShould you download more data?')
    print('  Yes. More subjects and runs will almost certainly help. CSP benefits from')
    print('  more trials to estimate covariance matrices robustly. Aim for more runs/subjects,')
    print('  and consider cross-subject generalization strategies (domain adaptation) if scaling.')


if __name__ == "__main__":
    main()
