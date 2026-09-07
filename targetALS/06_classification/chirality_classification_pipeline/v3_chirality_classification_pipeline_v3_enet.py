#!/usr/bin/env python3
"""
Chirality Descriptor Classification Pipeline — V3 (Elastic Net Selection)
=========================================================================
Replaces the filter-based feature selection (f-ANOVA + MI + power gates)
with Elastic Net Logistic Regression as an embedded feature selector.

Rationale:
  - Filter methods (V1/V2) converged on a single feature because each filter
    stage independently pruned candidates. Elastic Net jointly evaluates all
    features under a combined L1+L2 penalty:
      * L1 (lasso) drives sparsity → automatic feature selection
      * L2 (ridge) handles correlated features → doesn't arbitrarily drop
        one of two correlated-but-informative features
  - The alpha (regularization strength) and l1_ratio (L1 vs L2 balance) are
    tuned inside each LOOCV iteration via inner 3-fold CV, so the sparsity
    level adapts per fold rather than relying on fixed power thresholds.
  - This is a single coherent model rather than a multi-stage filter cascade,
    reducing the degrees of freedom in the pipeline design.

Pipeline structure (no leakage):
  5-fold outer CV
    → LOOCV inner on ~31 training samples
      → Per LOO iteration:
        1. StandardScale on LOO-train (30 samples)
        2. Elastic Net LR with inner 3-fold CV to select alpha + l1_ratio
        3. Extract features with |coef| > 0 (nonzero after L1)
      → Stable features: appear in ≥50% of LOO iterations
    → Train SVM/LR/RF on outer training set using stable features
    → Predict outer held-out test samples
  Performance from outer held-out predictions only.
  Conserved features for interpretation only.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
import json
from collections import Counter, defaultdict

# ─── CONFIG ─────────────────────────────────────────────────────────
OUTER_FOLDS = 5
RANDOM_STATE = 42
FEATURE_FREQ_THRESHOLD = 0.5  # feature must appear in ≥50% of LOO iterations

# Elastic Net grid: (C_inverse=alpha, l1_ratio)
# Lower C = stronger regularization = more sparsity
# l1_ratio closer to 1 = more L1 (lasso-like), closer to 0 = more L2 (ridge-like)
ENET_PARAMS = {
    'Cs': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    'l1_ratios': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    'inner_cv': 3,  # inner CV folds for tuning alpha/l1_ratio
}

# Downstream classifier hyperparams (small grid, tuned per outer fold)
DOWNSTREAM_GRIDS = {
    'SVM': [
        {'C': 0.1, 'gamma': 'scale'},
        {'C': 1.0, 'gamma': 'scale'},
        {'C': 10.0, 'gamma': 'scale'},
    ],
    'LR': [
        {'C': 0.01},
        {'C': 0.1},
        {'C': 1.0},
    ],
    'RF': [
        {'n_estimators': 100, 'max_depth': 2},
        {'n_estimators': 100, 'max_depth': 3},
        {'n_estimators': 200, 'max_depth': 3},
    ],
}

# ─── DATA LOADING ───────────────────────────────────────────────────
def load_data():
    """Load and merge descriptors with labels."""
    labels = pd.read_csv('sample_labels.csv')
    desc = pd.read_csv('chirality_interp_descriptors.csv')

    labels['code_clean'] = labels['code'].str.replace('.', '_', regex=False).str.replace(' ', '_')
    desc['code_clean'] = desc['code'].str.replace('.', '_', regex=False).str.replace(' ', '_')

    merged = desc.merge(labels[['code_clean', 'group']], on='code_clean', how='inner')

    feature_cols = [c for c in desc.columns if c not in ('code', 'code_clean')]
    X = merged[feature_cols].values.astype(np.float64)
    y = (merged['group'] == 'ALS').astype(int).values
    sample_codes = merged['code_clean'].values

    return X, y, sample_codes, feature_cols

# ─── ELASTIC NET FEATURE SELECTION ──────────────────────────────────
def elastic_net_select(X_train, y_train):
    """
    Fit Elastic Net Logistic Regression with inner CV to tune alpha + l1_ratio.
    Return indices of features with nonzero coefficients.

    Uses LogisticRegression with penalty='elasticnet' and saga solver.
    Inner CV (3-fold stratified) selects best (C, l1_ratio) by balanced accuracy.
    """
    best_score = -1
    best_C = 0.01
    best_l1 = 0.5
    best_coef = None

    inner_cv = StratifiedKFold(n_splits=ENET_PARAMS['inner_cv'], shuffle=True,
                                random_state=RANDOM_STATE)

    for C in ENET_PARAMS['Cs']:
        for l1_ratio in ENET_PARAMS['l1_ratios']:
            scores = []
            try:
                for icv_train, icv_val in inner_cv.split(X_train, y_train):
                    # Check class balance in folds
                    if len(np.unique(y_train[icv_train])) < 2:
                        continue
                    if len(np.unique(y_train[icv_val])) < 2:
                        continue

                    scaler = StandardScaler()
                    X_icv_tr = scaler.fit_transform(X_train[icv_train])
                    X_icv_val = scaler.transform(X_train[icv_val])

                    clf = LogisticRegression(
                        penalty='elasticnet',
                        C=C,
                        l1_ratio=l1_ratio,
                        solver='saga',
                        max_iter=10000,
                        random_state=RANDOM_STATE,
                        class_weight='balanced',
                    )
                    clf.fit(X_icv_tr, y_train[icv_train])
                    preds = clf.predict(X_icv_val)
                    scores.append(balanced_accuracy_score(y_train[icv_val], preds))

                if scores:
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_C = C
                        best_l1 = l1_ratio
            except Exception:
                continue

    # Refit on full LOO training set with best params
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    enet = LogisticRegression(
        penalty='elasticnet',
        C=best_C,
        l1_ratio=best_l1,
        solver='saga',
        max_iter=10000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
    )
    enet.fit(X_scaled, y_train)

    coef = enet.coef_.ravel()
    nonzero_idx = np.where(np.abs(coef) > 1e-10)[0]

    # Sort by absolute coefficient magnitude (most important first)
    if len(nonzero_idx) > 0:
        sorted_order = np.argsort(np.abs(coef[nonzero_idx]))[::-1]
        nonzero_idx = nonzero_idx[sorted_order]

    return nonzero_idx.tolist(), coef, best_C, best_l1, best_score

# ─── INNER LOOCV ────────────────────────────────────────────────────
def inner_loocv_enet(X_train, y_train, clf_name, clf_factory_grid):
    """
    LOOCV inner loop:
      1. Per iteration: Elastic Net selects features on LOO-train
      2. Downstream classifier (SVM/LR/RF) trained on selected features
      3. Predict LOO held-out sample
      4. Track feature stability across iterations

    Returns: predictions, probas, stable_features, feature_counter,
             best_downstream_params, enet_diagnostics
    """
    n_train = len(y_train)
    loo = LeaveOneOut()

    predictions = np.zeros(n_train)
    probas = np.zeros(n_train)
    all_selected_features = []
    all_n_features = []
    all_enet_params = []
    downstream_votes = Counter()

    for loo_train_idx, loo_test_idx in loo.split(X_train, y_train):
        X_lt = X_train[loo_train_idx]
        y_lt = y_train[loo_train_idx]
        X_ltest = X_train[loo_test_idx]

        # ── Step 1: Elastic Net feature selection ──
        selected, coef, best_C, best_l1, enet_score = elastic_net_select(X_lt, y_lt)

        all_enet_params.append({'C': best_C, 'l1_ratio': best_l1})
        all_n_features.append(len(selected))

        # Fallback if Elastic Net selects nothing (overly regularized)
        if len(selected) == 0:
            # Use top feature by absolute coefficient even if "zero"
            selected = [int(np.argmax(np.abs(coef)))]

        all_selected_features.append(set(selected))

        # ── Step 2: Scale selected features ──
        scaler = StandardScaler()
        X_sel_train = scaler.fit_transform(X_lt[:, selected])
        X_sel_test = scaler.transform(X_ltest[:, selected])

        # ── Step 3: Downstream classifier with quick param selection ──
        best_ds_score = -1
        best_ds_params = clf_factory_grid[0]

        for params in clf_factory_grid:
            clf = make_classifier(clf_name, params)
            clf.fit(X_sel_train, y_lt)
            train_preds = clf.predict(X_sel_train)
            score = balanced_accuracy_score(y_lt, train_preds)
            if score > best_ds_score:
                best_ds_score = score
                best_ds_params = params

        downstream_votes[str(best_ds_params)] += 1

        # ── Step 4: Final prediction ──
        clf = make_classifier(clf_name, best_ds_params)
        clf.fit(X_sel_train, y_lt)

        predictions[loo_test_idx] = clf.predict(X_sel_test)
        if hasattr(clf, 'predict_proba'):
            probas[loo_test_idx] = clf.predict_proba(X_sel_test)[:, 1]
        elif hasattr(clf, 'decision_function'):
            probas[loo_test_idx] = clf.decision_function(X_sel_test)

    # ── Stable features ──
    feature_counter = Counter()
    for feat_set in all_selected_features:
        feature_counter.update(feat_set)

    stable_features = [f for f, count in feature_counter.items()
                       if count / n_train >= FEATURE_FREQ_THRESHOLD]
    stable_features.sort(key=lambda f: feature_counter[f], reverse=True)

    # Best downstream params
    best_ds_params_overall = eval(downstream_votes.most_common(1)[0][0])

    # Diagnostics
    enet_diag = {
        'mean_n_features': np.mean(all_n_features),
        'std_n_features': np.std(all_n_features),
        'median_n_features': np.median(all_n_features),
        'min_n_features': int(np.min(all_n_features)),
        'max_n_features': int(np.max(all_n_features)),
        'common_C': Counter([p['C'] for p in all_enet_params]).most_common(3),
        'common_l1': Counter([p['l1_ratio'] for p in all_enet_params]).most_common(3),
    }

    return (predictions, probas, stable_features, feature_counter,
            best_ds_params_overall, enet_diag)

# ─── CLASSIFIER FACTORY ────────────────────────────────────────────
def make_classifier(clf_name, params):
    """Instantiate downstream classifier."""
    if clf_name == 'SVM':
        return SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **params)
    elif clf_name == 'LR':
        return LogisticRegression(solver='lbfgs', max_iter=5000,
                                   random_state=RANDOM_STATE, **params)
    elif clf_name == 'RF':
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)

# ─── OUTER MODEL TRAINING ──────────────────────────────────────────
def train_outer_model(X_train, y_train, X_test, stable_features,
                      clf_name, best_params):
    """Train on full outer training set with stable features."""
    if len(stable_features) == 0:
        # Emergency fallback: run Elastic Net on full training set
        selected, coef, _, _, _ = elastic_net_select(X_train, y_train)
        if len(selected) == 0:
            selected = [int(np.argmax(np.abs(coef)))]
        stable_features = selected

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[:, stable_features])
    X_te = scaler.transform(X_test[:, stable_features])

    # Calibrated classifier for better probability estimates
    base_clf = make_classifier(clf_name, best_params)
    try:
        cal_clf = CalibratedClassifierCV(base_clf, cv=min(3, len(y_train)),
                                          method='sigmoid')
        cal_clf.fit(X_tr, y_train)
        preds = cal_clf.predict(X_te)
        probs = cal_clf.predict_proba(X_te)[:, 1]
    except Exception:
        base_clf.fit(X_tr, y_train)
        preds = base_clf.predict(X_te)
        if hasattr(base_clf, 'predict_proba'):
            probs = base_clf.predict_proba(X_te)[:, 1]
        elif hasattr(base_clf, 'decision_function'):
            probs = base_clf.decision_function(X_te)
        else:
            probs = preds.astype(float)

    return preds, probs

# ─── FEATURE ANNOTATION ────────────────────────────────────────────
def annotate_feature(feat_name):
    """Parse feature name into biological components."""
    info = {'name': feat_name, 'chiralities': [], 'descriptor': '',
            'timepoint': '', 'is_pairwise': False}

    # Extract timepoint
    for tp in ['0h', '6h', '24h']:
        if feat_name.endswith(f'_{tp}'):
            info['timepoint'] = tp
            base = feat_name[:-len(tp) - 1]
            break
    else:
        base = feat_name

    # Pairwise vs single
    if '_vs_' in base:
        info['is_pairwise'] = True
        parts = base.split('_vs_')
        info['chiralities'].append(parts[0].replace('ch', '(').replace('_', ',') + ')')
        rest = parts[1]
        for desc_type in ['ratio', 'eemdist']:
            if f'_{desc_type}' in rest:
                info['descriptor'] = desc_type
                ch2 = rest.replace(f'_{desc_type}', '')
                info['chiralities'].append(ch2.replace('ch', '(').replace('_', ',') + ')')
                break
    else:
        for desc_type in ['gauss_max', 'gauss_auc', 'gauss_fwhm',
                          'gauss_em_center', 'gauss_ex_center',
                          'gauss_skew', 'gauss_kurt']:
            if f'_{desc_type}' in base:
                info['descriptor'] = desc_type
                ch = base.replace(f'_{desc_type}', '')
                info['chiralities'].append(ch.replace('ch', '(').replace('_', ',') + ')')
                break

    return info

# ─── MAIN PIPELINE ──────────────────────────────────────────────────
def run_pipeline():
    print("=" * 70)
    print("CHIRALITY CLASSIFICATION PIPELINE — V3 (ELASTIC NET SELECTION)")
    print("=" * 70)

    X, y, sample_codes, feature_cols = load_data()
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Balance: ALS={y.sum()}, CTR={len(y)-y.sum()}")
    print(f"\nElastic Net grid: {len(ENET_PARAMS['Cs'])} C values × "
          f"{len(ENET_PARAMS['l1_ratios'])} l1_ratios = "
          f"{len(ENET_PARAMS['Cs']) * len(ENET_PARAMS['l1_ratios'])} combos")
    print(f"Inner CV for Elastic Net tuning: {ENET_PARAMS['inner_cv']}-fold")

    # Handle non-finite values
    if not np.all(np.isfinite(X)):
        print("WARNING: Non-finite values found, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf_names = ['SVM', 'LR', 'RF']
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    all_outer_preds = {name: np.zeros(len(y)) for name in clf_names}
    all_outer_probs = {name: np.zeros(len(y)) for name in clf_names}
    fold_features = {name: [] for name in clf_names}
    fold_metrics = {name: [] for name in clf_names}
    fold_n_features = {name: [] for name in clf_names}
    ensemble_probs = np.zeros(len(y))

    print(f"\n{'='*70}")
    print(f"RUNNING {OUTER_FOLDS}-FOLD OUTER CV")
    print(f"{'='*70}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n{'─'*50}")
        print(f"Outer Fold {fold_idx+1}/{OUTER_FOLDS}")
        print(f"  Train: {len(train_idx)} (ALS={y_train.sum()}, "
              f"CTR={len(y_train)-y_train.sum()})")
        print(f"  Test:  {len(test_idx)} (ALS={y_test.sum()}, "
              f"CTR={len(y_test)-y_test.sum()})")

        fold_ensemble_probs = np.zeros(len(test_idx))

        for clf_name in clf_names:
            print(f"\n  [{clf_name}] Inner LOOCV with Elastic Net selection...")

            (loo_preds, loo_probs, stable_feats, feat_counter,
             best_ds_params, enet_diag) = inner_loocv_enet(
                X_train, y_train, clf_name, DOWNSTREAM_GRIDS[clf_name]
            )

            inner_acc = accuracy_score(y_train, loo_preds)
            try:
                inner_auc = roc_auc_score(y_train, loo_probs)
            except:
                inner_auc = 0.5

            print(f"  [{clf_name}] Inner: acc={inner_acc:.3f}, AUC={inner_auc:.3f}")
            print(f"  [{clf_name}] Elastic Net diagnostics:")
            print(f"        Features per iter: "
                  f"{enet_diag['mean_n_features']:.1f}±{enet_diag['std_n_features']:.1f} "
                  f"(range {enet_diag['min_n_features']}–{enet_diag['max_n_features']})")
            print(f"        Most common C: {enet_diag['common_C']}")
            print(f"        Most common l1_ratio: {enet_diag['common_l1']}")
            print(f"  [{clf_name}] Downstream params: {best_ds_params}")
            print(f"  [{clf_name}] Stable features ({len(stable_feats)}):")

            for feat_idx in stable_feats[:15]:
                freq = feat_counter[feat_idx] / len(y_train)
                info = annotate_feature(feature_cols[feat_idx])
                pair_tag = " [PAIR]" if info['is_pairwise'] else ""
                print(f"        {feature_cols[feat_idx]} "
                      f"(freq={freq:.0%}, {info['descriptor']}, "
                      f"ch={info['chiralities']}, tp={info['timepoint']}{pair_tag})")

            if len(stable_feats) > 15:
                print(f"        ... and {len(stable_feats)-15} more")

            # Outer fold prediction
            preds, probs = train_outer_model(
                X_train, y_train, X_test, stable_feats,
                clf_name, best_ds_params
            )

            all_outer_preds[clf_name][test_idx] = preds
            all_outer_probs[clf_name][test_idx] = probs
            fold_features[clf_name].append(set(stable_feats))
            fold_n_features[clf_name].append(len(stable_feats))

            fold_ensemble_probs += probs / len(clf_names)

            fold_acc = accuracy_score(y_test, preds)
            try:
                fold_auc = roc_auc_score(y_test, probs)
            except:
                fold_auc = 0.5
            fold_metrics[clf_name].append({'acc': fold_acc, 'auc': fold_auc})
            print(f"  [{clf_name}] Outer test: acc={fold_acc:.3f}, AUC={fold_auc:.3f}")

        ensemble_probs[test_idx] = fold_ensemble_probs

    # ─── AGGREGATE RESULTS ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (outer fold held-out predictions)")
    print(f"{'='*70}")

    summary_rows = []

    for clf_name in clf_names:
        preds = all_outer_preds[clf_name]
        probs = all_outer_probs[clf_name]

        acc = accuracy_score(y, preds)
        bal_acc = balanced_accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probs)
        except:
            auc = 0.5
        f1 = f1_score(y, preds)
        mcc = matthews_corrcoef(y, preds)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_accs = [m['acc'] for m in fold_metrics[clf_name]]
        fold_aucs = [m['auc'] for m in fold_metrics[clf_name]]

        print(f"\n  {clf_name}:")
        print(f"    Accuracy:          {acc:.3f}")
        print(f"    Balanced Accuracy: {bal_acc:.3f}")
        print(f"    AUC:               {auc:.3f}")
        print(f"    F1:                {f1:.3f}")
        print(f"    MCC:               {mcc:.3f}")
        print(f"    Sensitivity:       {sens:.3f}")
        print(f"    Specificity:       {spec:.3f}")
        print(f"    Confusion Matrix:  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"    Per-fold ACC: {[f'{a:.3f}' for a in fold_accs]}")
        print(f"    Per-fold AUC: {[f'{a:.3f}' for a in fold_aucs]}")
        print(f"    ACC mean±std: {np.mean(fold_accs):.3f}±{np.std(fold_accs):.3f}")
        print(f"    AUC mean±std: {np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}")
        print(f"    Avg stable features/fold: {np.mean(fold_n_features[clf_name]):.1f}")

        summary_rows.append({
            'Classifier': clf_name,
            'Accuracy': round(acc, 3),
            'Balanced_Accuracy': round(bal_acc, 3),
            'AUC': round(auc, 3),
            'F1': round(f1, 3),
            'MCC': round(mcc, 3),
            'Sensitivity': round(sens, 3),
            'Specificity': round(spec, 3),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Fold_ACC_mean': round(np.mean(fold_accs), 3),
            'Fold_ACC_std': round(np.std(fold_accs), 3),
            'Fold_AUC_mean': round(np.mean(fold_aucs), 3),
            'Fold_AUC_std': round(np.std(fold_aucs), 3),
            'Avg_stable_features': round(np.mean(fold_n_features[clf_name]), 1),
        })

    # ── Ensemble ──
    ens_preds = (ensemble_probs >= 0.5).astype(int)
    ens_acc = accuracy_score(y, ens_preds)
    ens_bal_acc = balanced_accuracy_score(y, ens_preds)
    try:
        ens_auc = roc_auc_score(y, ensemble_probs)
    except:
        ens_auc = 0.5
    ens_f1 = f1_score(y, ens_preds)
    ens_mcc = matthews_corrcoef(y, ens_preds)
    tn, fp, fn, tp = confusion_matrix(y, ens_preds).ravel()

    print(f"\n  ENSEMBLE (soft vote SVM+LR+RF):")
    print(f"    Accuracy:          {ens_acc:.3f}")
    print(f"    Balanced Accuracy: {ens_bal_acc:.3f}")
    print(f"    AUC:               {ens_auc:.3f}")
    print(f"    F1:                {ens_f1:.3f}")
    print(f"    MCC:               {ens_mcc:.3f}")
    print(f"    Confusion Matrix:  TP={tp} FP={fp} FN={fn} TN={tn}")

    summary_rows.append({
        'Classifier': 'Ensemble',
        'Accuracy': round(ens_acc, 3),
        'Balanced_Accuracy': round(ens_bal_acc, 3),
        'AUC': round(ens_auc, 3),
        'F1': round(ens_f1, 3),
        'MCC': round(ens_mcc, 3),
    })

    # ─── CONSERVED FEATURES ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONSERVED FEATURES ACROSS FOLDS (interpretation only)")
    print(f"{'='*70}")

    conserved_info = {}

    for clf_name in clf_names:
        if not fold_features[clf_name]:
            continue

        feat_fold_count = Counter()
        for feat_set in fold_features[clf_name]:
            feat_fold_count.update(feat_set)

        conserved = set.intersection(*fold_features[clf_name]) \
            if all(fold_features[clf_name]) else set()
        robust_feats = {f: c for f, c in feat_fold_count.items() if c >= 3}

        print(f"\n  {clf_name}:")
        print(f"    All {OUTER_FOLDS} folds ({len(conserved)}):")
        for idx in sorted(conserved):
            info = annotate_feature(feature_cols[idx])
            print(f"      ✓ {feature_cols[idx]}  "
                  f"[{info['descriptor']}] ch={info['chiralities']} {info['timepoint']}")

        print(f"    ≥3/{OUTER_FOLDS} folds ({len(robust_feats)}):")
        for idx in sorted(robust_feats, key=lambda x: robust_feats[x], reverse=True):
            info = annotate_feature(feature_cols[idx])
            print(f"      • {feature_cols[idx]}  "
                  f"({robust_feats[idx]}/{OUTER_FOLDS}) "
                  f"[{info['descriptor']}] ch={info['chiralities']} {info['timepoint']}")

        conserved_info[clf_name] = {
            'all_folds': [{'index': int(i), 'name': feature_cols[i],
                          **annotate_feature(feature_cols[i])}
                         for i in sorted(conserved)],
            'robust_3plus': [{'index': int(i), 'name': feature_cols[i],
                             'n_folds': int(robust_feats[i]),
                             **annotate_feature(feature_cols[i])}
                            for i in sorted(robust_feats,
                                          key=lambda x: robust_feats[x], reverse=True)]
        }

    # ─── CROSS-CLASSIFIER CONVERGENCE ───────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-CLASSIFIER FEATURE CONVERGENCE")
    print(f"{'='*70}")

    all_robust = defaultdict(set)
    for clf_name in clf_names:
        feat_fold_count = Counter()
        for feat_set in fold_features[clf_name]:
            feat_fold_count.update(feat_set)
        for f, c in feat_fold_count.items():
            if c >= 3:
                all_robust[f].add(clf_name)

    multi_clf = {f: clfs for f, clfs in all_robust.items() if len(clfs) >= 2}
    if multi_clf:
        print("\n  Features robust in ≥2 classifiers:")
        for idx in sorted(multi_clf, key=lambda x: len(multi_clf[x]), reverse=True):
            info = annotate_feature(feature_cols[idx])
            print(f"    {feature_cols[idx]}  → {multi_clf[idx]}  "
                  f"[{info['descriptor']}] ch={info['chiralities']} {info['timepoint']}")
    else:
        print("\n  No features robust across multiple classifiers.")

    # ─── SAVE ───────────────────────────────────────────────────────
    pd.DataFrame(summary_rows).to_csv('v3_enet_classification_summary.csv',
                                       index=False)

    pred_rows = []
    for clf_name in clf_names:
        for i in range(len(y)):
            pred_rows.append({
                'sample': sample_codes[i],
                'true_label': 'ALS' if y[i] == 1 else 'CTR',
                'classifier': clf_name,
                'predicted': 'ALS' if all_outer_preds[clf_name][i] == 1 else 'CTR',
                'probability': round(float(all_outer_probs[clf_name][i]), 4),
                'correct': int(all_outer_preds[clf_name][i] == y[i]),
            })
    for i in range(len(y)):
        pred_rows.append({
            'sample': sample_codes[i],
            'true_label': 'ALS' if y[i] == 1 else 'CTR',
            'classifier': 'Ensemble',
            'predicted': 'ALS' if ens_preds[i] == 1 else 'CTR',
            'probability': round(float(ensemble_probs[i]), 4),
            'correct': int(ens_preds[i] == y[i]),
        })
    pd.DataFrame(pred_rows).to_csv('v3_enet_per_sample_predictions.csv',
                                    index=False)

    with open('v3_enet_conserved_features.json', 'w') as f:
        json.dump(conserved_info, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("FILES SAVED")
    print(f"{'='*70}")
    print("  v3_enet_classification_summary.csv")
    print("  v3_enet_per_sample_predictions.csv")
    print("  v3_enet_conserved_features.json")
    print("\nDone!")

if __name__ == '__main__':
    run_pipeline()
