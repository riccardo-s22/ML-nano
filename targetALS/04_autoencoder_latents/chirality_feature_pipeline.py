#!/usr/bin/env python3
"""
Chirality Descriptor Feature Extraction Pipeline
=================================================
Design (Option A — no leakage):
  - 5-fold stratified outer CV (≈31 train / ≈8 test per fold)
  - Inside each outer fold: LOOCV on training samples
  - Inside each LOOCV iteration (on ~30 samples):
      1. f-ANOVA ranking (sklearn SelectKBest)
      2. Correlation filter: greedily remove features |r|>0.8
         with already-selected ones
      3. Power-driven accumulation: add features (in f-ANOVA rank
         order) until projected power ≥ 90% or MAX_FEATURES hit
      4. Train SVM / LR / RF on selected features, predict held-out
  - Per outer fold: consensus features = selected in ≥50% of LOOCV iters
  - Retrain on all fold-training samples with consensus features → predict test
  - Aggregate outer-fold predictions for unbiased performance
  - Cross-fold conserved features = intersection across 5 folds (interpretability only)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from scipy.stats import pearsonr, t as t_dist, nct

warnings.filterwarnings('ignore')

# ─────────────────────────── CONFIG ───────────────────────────
N_OUTER_FOLDS = 5
MAX_FEATURES = 5          # hard cap given n≈39
POWER_TARGET = 0.90
POWER_ALPHA = 0.05
CORR_THRESHOLD = 0.80     # remove features correlated above this
CONSENSUS_FRAC = 0.50     # feature must appear in ≥50% of LOOCV iters
RANDOM_STATE = 42
N_PERMUTATIONS = 500     # for permutation testing

# ─────────────────────────── DATA LOADING ───────────────────────────
def load_data(desc_path, labels_path):
    """Load and merge descriptors with labels."""
    desc = pd.read_csv(desc_path)
    labels = pd.read_csv(labels_path)
    # Harmonize codes (dots vs underscores)
    labels['code_clean'] = labels['code'].str.replace('.', '_', regex=False)
    merged = desc.merge(labels[['code_clean', 'group']],
                        left_on='code', right_on='code_clean')
    feature_cols = [c for c in desc.columns if c != 'code']
    X = merged[feature_cols].values.astype(np.float64)
    y = LabelEncoder().fit_transform(merged['group'].values)  # ALS=0, CTRL=1
    sample_codes = merged['code'].values
    
    print(f"Loaded {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Class distribution: ALS={np.sum(y==0)}, CTRL={np.sum(y==1)}")
    return X, y, feature_cols, sample_codes


# ─────────────────────────── FEATURE SELECTION (INNER) ───────────────────────────
def compute_cohens_d(x, y_binary, feature_idx):
    """Compute Cohen's d for a single feature between two groups."""
    g0 = x[y_binary == 0, feature_idx]
    g1 = x[y_binary == 1, feature_idx]
    n0, n1 = len(g0), len(g1)
    if n0 < 2 or n1 < 2:
        return 0.0
    pooled_std = np.sqrt(((n0-1)*np.var(g0, ddof=1) + (n1-1)*np.var(g1, ddof=1))
                         / (n0+n1-2))
    if pooled_std < 1e-12:
        return 0.0
    return abs(np.mean(g0) - np.mean(g1)) / pooled_std


def compute_power(effect_size, n0, n1, alpha=0.05):
    """Compute statistical power for a two-sample t-test using non-central t."""
    if effect_size < 1e-6:
        return 0.0
    try:
        df = n0 + n1 - 2
        # Non-centrality parameter
        nc = effect_size * np.sqrt(n0 * n1 / (n0 + n1))
        # Critical value from central t
        t_crit = t_dist.ppf(1 - alpha / 2, df)
        # Power = P(|T| > t_crit) under non-central t
        pw = 1 - nct.cdf(t_crit, df, nc) + nct.cdf(-t_crit, df, nc)
        return min(max(pw, 0.0), 1.0)
    except Exception:
        return 0.0


def compute_multivariate_power(X_sel, y, alpha=0.05):
    """
    Estimate multivariate discriminative power.
    Projects selected features onto the LDA direction, computes Cohen's d
    on the projected scores, then computes power.
    """
    g0 = X_sel[y == 0]
    g1 = X_sel[y == 1]
    n0, n1 = len(g0), len(g1)
    
    if X_sel.shape[1] == 1:
        d = compute_cohens_d(X_sel, y, 0)
    else:
        # Fisher LDA projection
        mu0, mu1 = g0.mean(axis=0), g1.mean(axis=0)
        Sw = np.cov(g0.T, ddof=1) * (n0-1) + np.cov(g1.T, ddof=1) * (n1-1)
        Sw /= (n0 + n1 - 2)
        # Regularize
        Sw += np.eye(Sw.shape[0]) * 1e-6
        try:
            w = np.linalg.solve(Sw, mu1 - mu0)
        except np.linalg.LinAlgError:
            w = mu1 - mu0
        proj0 = g0 @ w
        proj1 = g1 @ w
        pooled = np.sqrt(((n0-1)*np.var(proj0, ddof=1) + (n1-1)*np.var(proj1, ddof=1))
                         / (n0+n1-2))
        d = abs(np.mean(proj0) - np.mean(proj1)) / max(pooled, 1e-12)
    
    return compute_power(d, n0, n1, alpha)


def select_features_inner(X_train, y_train, feature_names,
                          max_features=MAX_FEATURES,
                          power_target=POWER_TARGET,
                          corr_threshold=CORR_THRESHOLD):
    """
    Feature selection for one LOOCV training iteration.
    Returns list of selected feature indices and names.
    """
    n_samples, n_features = X_train.shape
    
    # Step 1: f-ANOVA ranking
    F_scores, p_values = f_classif(X_train, y_train)
    # Replace NaN with worst
    F_scores = np.nan_to_num(F_scores, nan=0.0)
    ranked_indices = np.argsort(-F_scores)  # descending
    
    # Step 2+3: Greedy forward selection with correlation filter + power gate
    selected_idx = []
    
    for candidate in ranked_indices:
        if len(selected_idx) >= max_features:
            break
        
        # Correlation filter: skip if |r| > threshold with any selected
        skip = False
        for sel in selected_idx:
            r, _ = pearsonr(X_train[:, candidate], X_train[:, sel])
            if abs(r) > corr_threshold:
                skip = True
                break
        if skip:
            continue
        
        # Add candidate
        selected_idx.append(candidate)
        
        # Check multivariate power
        X_sel = X_train[:, selected_idx]
        scaler = StandardScaler()
        X_sel_scaled = scaler.fit_transform(X_sel)
        pw = compute_multivariate_power(X_sel_scaled, y_train)
        
        if pw >= power_target:
            break
    
    # Ensure at least 1 feature
    if len(selected_idx) == 0:
        selected_idx = [ranked_indices[0]]
    
    selected_names = [feature_names[i] for i in selected_idx]
    return selected_idx, selected_names


# ─────────────────────────── MODEL DEFINITIONS ───────────────────────────
def get_models():
    return {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                    random_state=RANDOM_STATE),
        'LR': LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                  random_state=RANDOM_STATE),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=3,
                                      random_state=RANDOM_STATE)
    }


# ─────────────────────────── INNER LOOCV ───────────────────────────
def run_inner_loocv(X_fold_train, y_fold_train, feature_names, model_name):
    """
    Run LOOCV within one outer fold for one model type.
    Returns: consensus features, LOOCV accuracy, per-iter feature selections
    """
    loo = LeaveOneOut()
    n = len(y_fold_train)
    
    predictions = np.zeros(n, dtype=int)
    probas = np.zeros(n)
    feature_counter = Counter()
    per_iter_features = []
    
    for train_idx, val_idx in loo.split(X_fold_train):
        X_tr, X_val = X_fold_train[train_idx], X_fold_train[val_idx]
        y_tr, y_val = y_fold_train[train_idx], y_fold_train[val_idx]
        
        # Feature selection on training split only
        sel_idx, sel_names = select_features_inner(X_tr, y_tr, feature_names)
        per_iter_features.append(sel_names)
        
        for name in sel_names:
            feature_counter[name] += 1
        
        # Scale and train
        scaler = StandardScaler()
        X_tr_sel = scaler.fit_transform(X_tr[:, sel_idx])
        X_val_sel = scaler.transform(X_val[:, sel_idx])
        
        model = get_models()[model_name]
        model.fit(X_tr_sel, y_tr)
        
        predictions[val_idx[0]] = model.predict(X_val_sel)[0]
        probas[val_idx[0]] = model.predict_proba(X_val_sel)[0, 1]
    
    # Consensus: features selected in >= CONSENSUS_FRAC of iterations
    min_count = int(np.ceil(CONSENSUS_FRAC * n))
    consensus_features = [feat for feat, count in feature_counter.items()
                          if count >= min_count]
    
    # If no feature meets consensus, take the most frequent one
    if len(consensus_features) == 0:
        consensus_features = [feature_counter.most_common(1)[0][0]]
    
    loocv_acc = accuracy_score(y_fold_train, predictions)
    try:
        loocv_auc = roc_auc_score(y_fold_train, probas)
    except ValueError:
        loocv_auc = np.nan
    
    return consensus_features, loocv_acc, loocv_auc, feature_counter, per_iter_features


# ─────────────────────────── OUTER 5-FOLD CV ───────────────────────────
def run_outer_cv(X, y, feature_names, sample_codes):
    """
    Main pipeline: 5-fold outer CV with LOOCV inner.
    """
    skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    
    results = {}
    all_fold_features = {m: [] for m in ['SVM', 'LR', 'RF']}
    outer_predictions = {m: {'y_true': [], 'y_pred': [], 'y_prob': [],
                             'sample_codes': []}
                         for m in ['SVM', 'LR', 'RF']}
    fold_details = []
    
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"OUTER FOLD {fold_i+1}/{N_OUTER_FOLDS}")
        print(f"  Train: {len(train_idx)} samples | Test: {len(test_idx)} samples")
        print(f"  Train class dist: ALS={np.sum(y[train_idx]==0)}, CTRL={np.sum(y[train_idx]==1)}")
        print(f"{'='*60}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_info = {'fold': fold_i+1,
                     'n_train': len(train_idx),
                     'n_test': len(test_idx),
                     'models': {}}
        
        for model_name in ['SVM', 'LR', 'RF']:
            print(f"\n  --- {model_name} ---")
            
            # Inner LOOCV: feature selection + validation
            consensus_feats, loocv_acc, loocv_auc, feat_counter, _ = \
                run_inner_loocv(X_train, y_train, feature_names, model_name)
            
            print(f"  LOOCV Acc: {loocv_acc:.3f} | AUC: {loocv_auc:.3f}")
            print(f"  Consensus features ({len(consensus_feats)}): {consensus_feats}")
            
            all_fold_features[model_name].append(set(consensus_feats))
            
            # Retrain on full fold-training data with consensus features
            feat_idx = [feature_names.index(f) for f in consensus_feats]
            scaler = StandardScaler()
            X_train_sel = scaler.fit_transform(X_train[:, feat_idx])
            X_test_sel = scaler.transform(X_test[:, feat_idx])
            
            model = get_models()[model_name]
            model.fit(X_train_sel, y_train)
            
            y_pred = model.predict(X_test_sel)
            y_prob = model.predict_proba(X_test_sel)[:, 1]
            
            outer_predictions[model_name]['y_true'].extend(y_test.tolist())
            outer_predictions[model_name]['y_pred'].extend(y_pred.tolist())
            outer_predictions[model_name]['y_prob'].extend(y_prob.tolist())
            outer_predictions[model_name]['sample_codes'].extend(
                sample_codes[test_idx].tolist())
            
            fold_info['models'][model_name] = {
                'consensus_features': consensus_feats,
                'n_consensus': len(consensus_feats),
                'loocv_acc': loocv_acc,
                'loocv_auc': loocv_auc,
                'test_acc': accuracy_score(y_test, y_pred),
                'top_features_by_count': feat_counter.most_common(10)
            }
        
        fold_details.append(fold_info)
    
    # ─── Aggregate outer CV performance ───
    print(f"\n{'='*60}")
    print("AGGREGATE OUTER CV PERFORMANCE")
    print(f"{'='*60}")
    
    for model_name in ['SVM', 'LR', 'RF']:
        yt = np.array(outer_predictions[model_name]['y_true'])
        yp = np.array(outer_predictions[model_name]['y_pred'])
        yprob = np.array(outer_predictions[model_name]['y_prob'])
        
        acc = accuracy_score(yt, yp)
        bal_acc = balanced_accuracy_score(yt, yp)
        try:
            auc = roc_auc_score(yt, yprob)
        except ValueError:
            auc = np.nan
        
        print(f"\n  {model_name}:")
        print(f"    Accuracy:          {acc:.3f}")
        print(f"    Balanced Accuracy: {bal_acc:.3f}")
        print(f"    AUC:               {auc:.3f}")
        print(f"    Confusion Matrix:\n{confusion_matrix(yt, yp)}")
        
        results[model_name] = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'auc': auc,
            'confusion_matrix': confusion_matrix(yt, yp).tolist()
        }
    
    # ─── Cross-fold conserved features (interpretability only) ───
    print(f"\n{'='*60}")
    print("CROSS-FOLD CONSERVED FEATURES (Interpretability)")
    print(f"{'='*60}")
    
    for model_name in ['SVM', 'LR', 'RF']:
        fold_sets = all_fold_features[model_name]
        # Strict intersection
        conserved = set.intersection(*fold_sets) if fold_sets else set()
        # Also compute frequency across folds
        all_feats = [f for s in fold_sets for f in s]
        freq = Counter(all_feats)
        
        print(f"\n  {model_name}:")
        print(f"    Strict intersection (all 5 folds): {conserved if conserved else 'None'}")
        print(f"    Features in ≥4 folds: "
              f"{[f for f,c in freq.items() if c >= 4]}")
        print(f"    Features in ≥3 folds: "
              f"{[f for f,c in freq.items() if c >= 3]}")
        
        results[model_name]['conserved_strict'] = list(conserved)
        results[model_name]['conserved_4of5'] = [f for f,c in freq.items() if c >= 4]
        results[model_name]['conserved_3of5'] = [f for f,c in freq.items() if c >= 3]
        results[model_name]['fold_feature_freq'] = dict(freq)
    
    return results, outer_predictions, fold_details


# ─────────────────────────── PERMUTATION TEST ───────────────────────────
def run_permutation_test(X, y, feature_names, results, n_perms=N_PERMUTATIONS):
    """
    Permutation test: shuffle labels, re-run outer CV, compare AUC.
    Uses a lightweight version (only consensus features from real run).
    """
    print(f"\n{'='*60}")
    print(f"PERMUTATION TEST ({n_perms} permutations)")
    print(f"{'='*60}")
    
    rng = np.random.RandomState(RANDOM_STATE)
    perm_aucs = {m: [] for m in ['SVM', 'LR', 'RF']}
    
    for perm_i in range(n_perms):
        y_perm = rng.permutation(y)
        
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        
        for model_name in ['SVM', 'LR', 'RF']:
            y_true_all, y_prob_all = [], []
            
            for train_idx, test_idx in skf.split(X, y_perm):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_perm[train_idx], y_perm[test_idx]
                
                # Quick feature selection (f-ANOVA top-k, no LOOCV)
                k = min(MAX_FEATURES, 3)
                F_scores, _ = f_classif(X_train, y_train)
                F_scores = np.nan_to_num(F_scores, nan=0.0)
                top_idx = np.argsort(-F_scores)[:k]
                
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[:, top_idx])
                X_te = scaler.transform(X_test[:, top_idx])
                
                model = get_models()[model_name]
                model.fit(X_tr, y_train)
                y_prob_all.extend(model.predict_proba(X_te)[:, 1].tolist())
                y_true_all.extend(y_test.tolist())
            
            try:
                perm_aucs[model_name].append(
                    roc_auc_score(y_true_all, y_prob_all))
            except ValueError:
                pass
        
        if (perm_i + 1) % 100 == 0:
            print(f"  Completed {perm_i+1}/{n_perms} permutations")
    
    # Compute p-values
    perm_results = {}
    for model_name in ['SVM', 'LR', 'RF']:
        real_auc = results[model_name]['auc']
        null_dist = np.array(perm_aucs[model_name])
        p_value = (np.sum(null_dist >= real_auc) + 1) / (len(null_dist) + 1)
        
        print(f"\n  {model_name}:")
        print(f"    Real AUC: {real_auc:.3f}")
        print(f"    Null mean: {null_dist.mean():.3f} ± {null_dist.std():.3f}")
        print(f"    p-value: {p_value:.4f}")
        
        perm_results[model_name] = {
            'real_auc': real_auc,
            'null_mean': float(null_dist.mean()),
            'null_std': float(null_dist.std()),
            'p_value': float(p_value)
        }
    
    return perm_results


# ─────────────────────────── MAIN ───────────────────────────
def main():
    import time
    t0 = time.time()
    
    # Paths
    desc_path = '/mnt/user-data/uploads/chirality_interp_descriptors.csv'
    labels_path = '/mnt/user-data/uploads/sample_labels.csv'
    output_dir = '/mnt/user-data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load
    X, y, feature_names, sample_codes = load_data(desc_path, labels_path)
    sys.stdout.flush()
    
    # Run pipeline
    results, outer_predictions, fold_details = \
        run_outer_cv(X, y, feature_names, sample_codes)
    sys.stdout.flush()
    print(f"\nOuter CV completed in {time.time()-t0:.1f}s")
    
    # Permutation test
    t1 = time.time()
    perm_results = run_permutation_test(X, y, feature_names, results)
    print(f"Permutation test completed in {time.time()-t1:.1f}s")
    sys.stdout.flush()
    
    # ─── Save results ───
    # Summary JSON
    output = {
        'config': {
            'n_outer_folds': N_OUTER_FOLDS,
            'max_features': MAX_FEATURES,
            'power_target': POWER_TARGET,
            'corr_threshold': CORR_THRESHOLD,
            'consensus_fraction': CONSENSUS_FRAC,
            'n_permutations': N_PERMUTATIONS,
        },
        'performance': results,
        'permutation_tests': perm_results,
    }
    
    json_path = os.path.join(output_dir, 'chirality_feature_pipeline_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Per-sample predictions CSV
    rows = []
    for model_name in ['SVM', 'LR', 'RF']:
        preds = outer_predictions[model_name]
        for i in range(len(preds['y_true'])):
            rows.append({
                'model': model_name,
                'sample_code': preds['sample_codes'][i],
                'y_true': preds['y_true'][i],
                'y_pred': preds['y_pred'][i],
                'y_prob': preds['y_prob'][i],
            })
    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(output_dir, 'chirality_pipeline_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    
    # Fold details CSV
    fold_rows = []
    for fd in fold_details:
        for model_name, minfo in fd['models'].items():
            fold_rows.append({
                'fold': fd['fold'],
                'model': model_name,
                'n_train': fd['n_train'],
                'n_test': fd['n_test'],
                'n_consensus_features': minfo['n_consensus'],
                'consensus_features': '; '.join(minfo['consensus_features']),
                'loocv_acc': minfo['loocv_acc'],
                'loocv_auc': minfo['loocv_auc'],
                'test_acc': minfo['test_acc'],
            })
    fold_df = pd.DataFrame(fold_rows)
    fold_path = os.path.join(output_dir, 'chirality_pipeline_fold_details.csv')
    fold_df.to_csv(fold_path, index=False)
    
    print(f"\n{'='*60}")
    print("OUTPUTS SAVED:")
    print(f"  {json_path}")
    print(f"  {pred_path}")
    print(f"  {fold_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
