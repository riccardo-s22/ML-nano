#!/usr/bin/env python3
"""
Chirality Descriptor Classification Pipeline
=============================================
5-fold outer CV  →  LOOCV inner  →  f-ANOVA + correlation filter + power accumulation
No data leakage: all feature selection inside training folds only.
Performance reported from outer fold held-out predictions (Option A).
Conserved features reported for interpretation only.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.feature_selection import f_classif
from scipy import stats
import json
from collections import Counter

# ─── CONFIG ─────────────────────────────────────────────────────────
OUTER_FOLDS = 5
TARGET_POWER = 0.90
MAX_FEATURES = 5          # hard cap given n~39
CORR_THRESHOLD = 0.80     # remove features with |r| > this
RANDOM_STATE = 42
FEATURE_FREQ_THRESHOLD = 0.5  # feature must appear in >50% of LOOCV iterations

# ─── DATA LOADING ───────────────────────────────────────────────────
print("=" * 70)
print("CHIRALITY DESCRIPTOR CLASSIFICATION PIPELINE")
print("=" * 70)

labels = pd.read_csv('sample_labels.csv')
desc = pd.read_csv('chirality_interp_descriptors_rem.csv')

# Harmonise code columns for merging
labels['code_clean'] = labels['code'].str.replace('.', '_', regex=False).str.replace(' ', '_')
desc['code_clean'] = desc['code'].str.replace('.', '_', regex=False).str.replace(' ', '_')

merged = desc.merge(labels[['code_clean', 'group']], on='code_clean', how='inner')
print(f"Merged samples: {len(merged)}  (ALS={sum(merged['group']=='ALS')}, CTR={sum(merged['group']=='CTRL')})")

feature_cols = [c for c in desc.columns if c not in ('code', 'code_clean')]
X = merged[feature_cols].values.astype(np.float64)
y = (merged['group'] == 'ALS').astype(int).values  # 1=ALS, 0=CTR
sample_codes = merged['code_clean'].values

print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"Label balance: ALS={y.sum()}, CTR={len(y)-y.sum()}")

# ─── HELPER: Cohen's d ─────────────────────────────────────────────
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std

# ─── HELPER: Approximate power (two-sample t-test) ─────────────────
def approx_power(d, n1, n2, alpha=0.05):
    """Approximate power for two-sample t-test given Cohen's d and group sizes."""
    from scipy.stats import norm
    if abs(d) < 1e-10:
        return alpha  # no effect
    se = np.sqrt(1/n1 + 1/n2)
    ncp = abs(d) / se  # non-centrality parameter
    z_alpha = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power

# ─── HELPER: Correlation filter ────────────────────────────────────
def correlation_filter(X_sel, selected_indices, threshold=CORR_THRESHOLD):
    """Remove features correlated above threshold, keeping higher-ranked ones."""
    if len(selected_indices) <= 1:
        return selected_indices
    
    corr_mat = np.abs(np.corrcoef(X_sel.T))
    keep = [selected_indices[0]]
    
    for i in range(1, len(selected_indices)):
        # Check correlation with all already-kept features
        idx_in_sel = i
        correlated = False
        for kept_pos, kept_idx in enumerate(keep):
            kept_sel_pos = selected_indices.index(kept_idx) if kept_idx in selected_indices else 0
            # Find positions within X_sel
            pos_i = i
            pos_k = list(selected_indices).index(kept_idx)
            if corr_mat[pos_i, pos_k] > threshold:
                correlated = True
                break
        if not correlated:
            keep.append(selected_indices[i])
    
    return keep

# ─── HELPER: Inner LOOCV feature selection + classification ────────
def inner_loocv(X_train, y_train, clf_name):
    """
    Run LOOCV on training set with per-iteration feature selection
    and 3-fold CV hyperparameter tuning (no hyperparameter trap).
    Returns: predictions, selected features per iteration, fold-level stable features, best_params.
    """
    n_train = len(y_train)
    loo = LeaveOneOut()
    
    predictions = np.zeros(n_train)
    probas = np.zeros(n_train)
    all_selected_features = []
    hyperparam_votes = Counter()
    
    for loo_train_idx, loo_test_idx in loo.split(X_train, y_train):
        X_loo_train = X_train[loo_train_idx]
        y_loo_train = y_train[loo_train_idx]
        X_loo_test = X_train[loo_test_idx]
        
        # Step 1: f-ANOVA ranking
        f_scores, p_values = f_classif(X_loo_train, y_loo_train)
        
        # Handle NaN f-scores
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        ranked_indices = np.argsort(f_scores)[::-1]  # descending
        
        # Step 2: Power-driven feature accumulation with correlation filter
        n1 = np.sum(y_loo_train == 1)
        n0 = np.sum(y_loo_train == 0)
        
        selected = []
        for idx in ranked_indices:
            if len(selected) >= MAX_FEATURES:
                break
            
            # Compute Cohen's d for this feature
            x_als = X_loo_train[y_loo_train == 1, idx]
            x_ctr = X_loo_train[y_loo_train == 0, idx]
            d = cohens_d(x_als, x_ctr)
            pwr = approx_power(d, n1, n0)
            
            if pwr < 0.3:  # skip very weak features
                continue
            
            # Check correlation with already selected
            if len(selected) > 0:
                corrs = [abs(np.corrcoef(X_loo_train[:, idx], 
                                          X_loo_train[:, s])[0, 1]) 
                         for s in selected]
                if max(corrs) > CORR_THRESHOLD:
                    continue
            
            selected.append(idx)
            
            # Check if cumulative power meets target
            powers = []
            for s in selected:
                x_a = X_loo_train[y_loo_train == 1, s]
                x_c = X_loo_train[y_loo_train == 0, s]
                powers.append(approx_power(cohens_d(x_a, x_c), n1, n0))
            
            # Combined power: P(detect at least one) = 1 - prod(1-p_i)
            combined_power = 1 - np.prod([1 - p for p in powers])
            if combined_power >= TARGET_POWER:
                break
        
        # Fallback: at least 1 feature
        if len(selected) == 0:
            selected = [ranked_indices[0]]
        
        all_selected_features.append(set(selected))
        
        # Step 3: Scale
        scaler = StandardScaler()
        X_sel_train = scaler.fit_transform(X_loo_train[:, selected])
        X_sel_test = scaler.transform(X_loo_test[:, selected])
        
        # Step 4: Hyperparameter selection via inner 3-fold CV
        best_score = -1
        best_params = HYPERPARAM_GRIDS[clf_name][0]
        
        inner_hp_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                       random_state=RANDOM_STATE)
        for params in HYPERPARAM_GRIDS[clf_name]:
            hp_fold_scores = []
            for hp_train_idx, hp_val_idx in inner_hp_cv.split(X_sel_train, y_loo_train):
                clf = make_classifier(clf_name, params)
                clf.fit(X_sel_train[hp_train_idx], y_loo_train[hp_train_idx])
                val_preds = clf.predict(X_sel_train[hp_val_idx])
                hp_fold_scores.append(balanced_accuracy_score(
                    y_loo_train[hp_val_idx], val_preds))
            score = np.mean(hp_fold_scores)
            if score > best_score:
                best_score = score
                best_params = params
        
        hyperparam_votes[str(best_params)] += 1
        
        # Step 5: Classify with best params
        clf = make_classifier(clf_name, best_params)
        clf.fit(X_sel_train, y_loo_train)
        
        predictions[loo_test_idx] = clf.predict(X_sel_test)
        if hasattr(clf, 'predict_proba'):
            probas[loo_test_idx] = clf.predict_proba(X_sel_test)[:, 1]
        elif hasattr(clf, 'decision_function'):
            probas[loo_test_idx] = clf.decision_function(X_sel_test)
    
    # Identify stable features: appear in > FREQ_THRESHOLD of LOO iterations
    feature_counter = Counter()
    for feat_set in all_selected_features:
        feature_counter.update(feat_set)
    
    stable_features = [f for f, count in feature_counter.items()
                       if count / n_train >= FEATURE_FREQ_THRESHOLD]
    
    # Sort stable features by frequency
    stable_features.sort(key=lambda f: feature_counter[f], reverse=True)
    
    # Most-voted hyperparameters
    best_params_overall = eval(hyperparam_votes.most_common(1)[0][0])
    
    return predictions, probas, stable_features, feature_counter, best_params_overall

# ─── HELPER: Train final model on outer fold with stable features ──
def train_outer_model(X_train, y_train, X_test, stable_features, clf_name, best_params):
    """Train on full outer training set using stable features, predict test."""
    if len(stable_features) == 0:
        # Fallback: use top feature by f-ANOVA on full training set
        f_scores, _ = f_classif(X_train, y_train)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        stable_features = [np.argmax(f_scores)]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[:, stable_features])
    X_te = scaler.transform(X_test[:, stable_features])
    
    clf = make_classifier(clf_name, best_params)
    clf.fit(X_tr, y_train)
    
    preds = clf.predict(X_te)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(X_te)[:, 1]
    elif hasattr(clf, 'decision_function'):
        probs = clf.decision_function(X_te)
    else:
        probs = preds.astype(float)
    
    return preds, probs

# ─── CLASSIFIERS ────────────────────────────────────────────────────
classifiers = {
    'SVM': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                       random_state=RANDOM_STATE),
    'LR':  lambda: LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000,
                                       random_state=RANDOM_STATE),
    'RF':  lambda: RandomForestClassifier(n_estimators=100, max_depth=3,
                                           random_state=RANDOM_STATE),
}

# ─── HYPERPARAMETER GRIDS + FACTORY ────────────────────────────────
HYPERPARAM_GRIDS = {
    'SVM': [
        {'C': 0.1, 'gamma': 'scale'},
        {'C': 1.0, 'gamma': 'scale'},
        {'C': 10.0, 'gamma': 'scale'},
        {'C': 1.0, 'gamma': 'auto'},
    ],
    'LR': [
        {'C': 0.01},
        {'C': 0.1},
        {'C': 1.0},
        {'C': 10.0},
    ],
    'RF': [
        {'n_estimators': 100, 'max_depth': 2},
        {'n_estimators': 100, 'max_depth': 3},
        {'n_estimators': 200, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
    ],
}

def make_classifier(clf_name, params):
    """Instantiate classifier with given hyperparameters."""
    if clf_name == 'SVM':
        return SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **params)
    elif clf_name == 'LR':
        return LogisticRegression(solver='lbfgs', max_iter=5000,
                                   random_state=RANDOM_STATE, **params)
    elif clf_name == 'RF':
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)

# ─── MAIN PIPELINE ──────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results = {}
all_outer_preds = {name: np.zeros(len(y)) for name in classifiers}
all_outer_probs = {name: np.zeros(len(y)) for name in classifiers}
fold_features = {name: [] for name in classifiers}
fold_metrics = {name: [] for name in classifiers}

print(f"\n{'='*70}")
print(f"RUNNING {OUTER_FOLDS}-FOLD OUTER CV WITH LOOCV INNER FEATURE SELECTION")
print(f"{'='*70}")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\n--- Outer Fold {fold_idx+1}/{OUTER_FOLDS} ---")
    print(f"    Train: {len(train_idx)} (ALS={y_train.sum()}, CTR={len(y_train)-y_train.sum()})")
    print(f"    Test:  {len(test_idx)} (ALS={y_test.sum()}, CTR={len(y_test)-y_test.sum()})")
    
    for clf_name, clf_factory in classifiers.items():
        print(f"\n    [{clf_name}] Running LOOCV inner loop...")
        
        # Inner LOOCV: feature selection + get stable features + best params
        loo_preds, loo_probs, stable_feats, feat_counter, best_params = inner_loocv(
            X_train, y_train, clf_name
        )
        
        # Inner LOOCV accuracy (diagnostic only)
        inner_acc = accuracy_score(y_train, loo_preds)
        inner_auc = roc_auc_score(y_train, loo_probs) if len(np.unique(loo_probs)) > 1 else 0.5
        
        print(f"    [{clf_name}] Inner LOOCV: acc={inner_acc:.3f}, AUC={inner_auc:.3f}")
        print(f"    [{clf_name}] Best params: {best_params}")
        print(f"    [{clf_name}] Stable features (>{FEATURE_FREQ_THRESHOLD*100:.0f}% freq): {len(stable_feats)}")
        print(f"    [{clf_name}] Feature indices: {stable_feats}")
        print(f"    [{clf_name}] Feature names: {[feature_cols[i] for i in stable_feats[:10]]}")
        
        # Train on full outer training set with stable features, predict test
        preds, probs = train_outer_model(X_train, y_train, X_test, stable_feats, clf_name, best_params)
        
        all_outer_preds[clf_name][test_idx] = preds
        all_outer_probs[clf_name][test_idx] = probs
        fold_features[clf_name].append(set(stable_feats))
        
        # Per-fold test metrics
        fold_acc = accuracy_score(y_test, preds)
        fold_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 and len(np.unique(probs)) > 1 else 0.5
        fold_metrics[clf_name].append({'acc': fold_acc, 'auc': fold_auc})
        
        print(f"    [{clf_name}] Outer fold test: acc={fold_acc:.3f}, AUC={fold_auc:.3f}")

# ─── AGGREGATE RESULTS ──────────────────────────────────────────────
print(f"\n{'='*70}")
print("AGGREGATE RESULTS (from outer fold held-out predictions)")
print(f"{'='*70}")

summary_rows = []

for clf_name in classifiers:
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
    
    print(f"\n  {clf_name}:")
    print(f"    Accuracy:          {acc:.3f}")
    print(f"    Balanced Accuracy: {bal_acc:.3f}")
    print(f"    AUC:               {auc:.3f}")
    print(f"    F1:                {f1:.3f}")
    print(f"    MCC:               {mcc:.3f}")
    print(f"    Sensitivity:       {sens:.3f}")
    print(f"    Specificity:       {spec:.3f}")
    print(f"    Confusion Matrix:  TP={tp} FP={fp} FN={fn} TN={tn}")
    
    # Per-fold metrics
    fold_accs = [m['acc'] for m in fold_metrics[clf_name]]
    fold_aucs = [m['auc'] for m in fold_metrics[clf_name]]
    print(f"    Per-fold ACC: {[f'{a:.3f}' for a in fold_accs]}")
    print(f"    Per-fold AUC: {[f'{a:.3f}' for a in fold_aucs]}")
    print(f"    ACC mean±std: {np.mean(fold_accs):.3f}±{np.std(fold_accs):.3f}")
    print(f"    AUC mean±std: {np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}")
    
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
    })

# ─── CONSERVED FEATURES (interpretation only) ───────────────────────
print(f"\n{'='*70}")
print("CONSERVED FEATURES ACROSS FOLDS (for interpretation only)")
print(f"{'='*70}")

conserved_info = {}

for clf_name in classifiers:
    # Features that appear in ALL folds
    if fold_features[clf_name]:
        all_feats_union = set().union(*fold_features[clf_name])
        conserved = set.intersection(*fold_features[clf_name]) if fold_features[clf_name] else set()
        
        # Also track frequency across folds
        feat_fold_count = Counter()
        for feat_set in fold_features[clf_name]:
            feat_fold_count.update(feat_set)
        
        # Features in at least 3/5 folds
        robust_feats = {f: c for f, c in feat_fold_count.items() if c >= 3}
        
        print(f"\n  {clf_name}:")
        print(f"    Features in ALL {OUTER_FOLDS} folds: {len(conserved)}")
        if conserved:
            for idx in sorted(conserved):
                print(f"      - {feature_cols[idx]} (idx={idx})")
        
        print(f"    Features in ≥3/{OUTER_FOLDS} folds: {len(robust_feats)}")
        for idx in sorted(robust_feats, key=lambda x: robust_feats[x], reverse=True):
            print(f"      - {feature_cols[idx]} (idx={idx}, folds={robust_feats[idx]}/{OUTER_FOLDS})")
        
        conserved_info[clf_name] = {
            'all_folds': [{'index': int(i), 'name': feature_cols[i]} for i in sorted(conserved)],
            'robust_3plus': [{'index': int(i), 'name': feature_cols[i], 'n_folds': int(robust_feats[i])} 
                           for i in sorted(robust_feats, key=lambda x: robust_feats[x], reverse=True)]
        }

# ─── SAVE RESULTS ───────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('classification_summary.csv', index=False)

# Detailed per-sample predictions
pred_rows = []
for clf_name in classifiers:
    for i in range(len(y)):
        pred_rows.append({
            'sample': sample_codes[i],
            'true_label': 'ALS' if y[i] == 1 else 'CTR',
            'classifier': clf_name,
            'predicted': 'ALS' if all_outer_preds[clf_name][i] == 1 else 'CTR',
            'probability': round(all_outer_probs[clf_name][i], 4),
            'correct': int(all_outer_preds[clf_name][i] == y[i]),
        })
pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv('per_sample_predictions.csv', index=False)

# Save conserved features
with open('conserved_features.json', 'w') as f:
    json.dump(conserved_info, f, indent=2)

print(f"\n{'='*70}")
print("FILES SAVED")
print(f"{'='*70}")
print("  classification_summary.csv")
print("  per_sample_predictions.csv")
print("  conserved_features.json")
print("\nDone!")
