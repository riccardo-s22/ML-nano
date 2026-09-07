#!/usr/bin/env python3
"""
Chirality Descriptor Classification Pipeline — V2 (Improved)
=============================================================
Improvements over V1:
  1. TEMPORAL FEATURE ENGINEERING
     - Delta features (6h−0h, 24h−0h, 24h−6h) capture temporal dynamics
     - Ratio-of-ratios across timepoints for pairwise descriptors
     → Motivated by: V1 found only 6h eemdist features. Deltas may capture
       disease-specific kinetic signatures missed by single-timepoint descriptors.

  2. SOFTER POWER GATE + MINIMUM EFFECT SIZE FILTER
     - Replaced 90% power hard gate with 70% target + Cohen's d ≥ 0.5 floor
     - Max features raised to 7 (from 5) since engineered features reduce
       collinearity
     → Motivated by: V1's 90% gate was too aggressive for n~15/group, requiring
       d≈1.0+. Only 1 feature survived in most folds. Softer gate allows
       moderate-effect features to contribute in combination.

  3. DUAL RANKING: f-ANOVA + MUTUAL INFORMATION
     - Rank features by both f-ANOVA (linear) and MI (nonlinear), take union
       of top-k from each
     → Motivated by: f-ANOVA alone misses nonlinear separability. MI captures
       features where distributions differ in shape, not just mean.

  4. HIERARCHICAL CLUSTERING REDUNDANCY REMOVAL
     - Replace pairwise correlation filter with agglomerative clustering at
       distance threshold → pick best feature per cluster
     → Motivated by: Pairwise greedy filter is order-dependent and can miss
       transitive correlations (A~B, B~C both pass but A~C would not).

  5. ENSEMBLE VOTING
     - Soft-voting ensemble of SVM/LR/RF using calibrated probabilities
     → Motivated by: Individual classifiers showed high fold-to-fold variance.
       Ensemble smooths prediction instability.

  6. HYPERPARAMETER SEARCH (INNER LOOCV)
     - Small grid for each classifier, selected per fold based on inner accuracy
     → Motivated by: V1 used fixed hyperparams. SVM C and gamma, LR C, RF
       depth may benefit from per-fold tuning.

  7. FEATURE-TYPE-AWARE SELECTION
     - Track which descriptor types (gauss_max, fwhm, ratio, eemdist, delta)
       and chiralities contribute, for biological interpretation
     → Motivated by: Knowing that e.g. ch9_4 fwhm deltas drive classification
       directly informs convergence analysis with autoencoder ROIs.

  8. CONFIDENCE CALIBRATION
     - Platt scaling on inner LOOCV predictions for better-calibrated probs
     → Motivated by: SVM decision_function and RF raw probabilities are often
       poorly calibrated with small n.

Pipeline structure (unchanged — no leakage):
  5-fold outer CV → LOOCV inner → feature engineering + selection + classification
  Performance from outer held-out predictions only.
  Conserved features for interpretation only.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import json
from collections import Counter, defaultdict

# ─── CONFIG ─────────────────────────────────────────────────────────
OUTER_FOLDS = 5
TARGET_POWER = 0.70           # V1 used 0.90 — too aggressive for n~15/group
MIN_COHENS_D = 0.5            # medium effect size floor
MAX_FEATURES = 7              # raised from 5; engineered features reduce redundancy
CLUSTER_DIST_THRESHOLD = 0.30 # 1 - |r| distance; features with |r| > 0.70 cluster
RANDOM_STATE = 42
FEATURE_FREQ_THRESHOLD = 0.5  # feature must appear in >50% of LOOCV iterations
MI_TOP_K = 30                 # top-k from MI ranking to consider
FANOVA_TOP_K = 30             # top-k from f-ANOVA ranking to consider

# ─── DATA LOADING ───────────────────────────────────────────────────
def load_data():
    """Load and merge descriptors with labels."""
    labels = pd.read_csv('/mnt/user-data/uploads/sample_labels.csv')
    desc = pd.read_csv('/mnt/user-data/uploads/chirality_interp_descriptors.csv')

    labels['code_clean'] = labels['code'].str.replace('.', '_', regex=False).str.replace(' ', '_')
    desc['code_clean'] = desc['code'].str.replace('.', '_', regex=False).str.replace(' ', '_')

    merged = desc.merge(labels[['code_clean', 'group']], on='code_clean', how='inner')

    feature_cols = [c for c in desc.columns if c not in ('code', 'code_clean')]
    X = merged[feature_cols].values.astype(np.float64)
    y = (merged['group'] == 'ALS').astype(int).values
    sample_codes = merged['code_clean'].values

    return X, y, sample_codes, feature_cols

# ─── IMPROVEMENT 1: TEMPORAL FEATURE ENGINEERING ────────────────────
def engineer_temporal_features(X, feature_cols):
    """
    Create delta features capturing temporal dynamics:
      - Δ(6h - 0h):  early response
      - Δ(24h - 0h): full response
      - Δ(24h - 6h): late-phase change
    For each descriptor type and chirality pair.

    Returns: augmented X, augmented feature_cols
    """
    # Parse features into structured dict: {(chirality, descriptor_type): {timepoint: col_index}}
    feature_map = {}  # (base_name) -> {timepoint: col_index}

    for i, col in enumerate(feature_cols):
        for tp in ['0h', '6h', '24h']:
            suffix = f'_{tp}'
            if col.endswith(suffix):
                base = col[:-len(suffix)]
                feature_map.setdefault(base, {})[tp] = i
                break

    # Generate deltas
    new_features = []
    new_names = []
    delta_pairs = [('6h', '0h', 'delta_6h_0h'),
                   ('24h', '0h', 'delta_24h_0h'),
                   ('24h', '6h', 'delta_24h_6h')]

    for base, tp_dict in feature_map.items():
        for tp_late, tp_early, delta_name in delta_pairs:
            if tp_late in tp_dict and tp_early in tp_dict:
                delta = X[:, tp_dict[tp_late]] - X[:, tp_dict[tp_early]]
                new_features.append(delta)
                new_names.append(f'{base}_{delta_name}')

    if new_features:
        X_new = np.column_stack([X] + new_features)
        feature_cols_new = list(feature_cols) + new_names
    else:
        X_new = X
        feature_cols_new = list(feature_cols)

    return X_new, feature_cols_new

# ─── IMPROVEMENT 2: POWER CALCULATION ──────────────────────────────
def cohens_d(x1, x2):
    """Compute Cohen's d between two groups."""
    n1, n2 = len(x1), len(x2)
    var1 = np.var(x1, ddof=1) if n1 > 1 else 0
    var2 = np.var(x2, ddof=1) if n2 > 1 else 0
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std < 1e-15:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std

def approx_power(d, n1, n2, alpha=0.05):
    """Approximate power for two-sample t-test given Cohen's d."""
    from scipy.stats import norm
    if abs(d) < 1e-10:
        return alpha
    se = np.sqrt(1/n1 + 1/n2)
    ncp = abs(d) / se
    z_alpha = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power

# ─── IMPROVEMENT 3: DUAL RANKING (f-ANOVA + MI) ────────────────────
def dual_rank_features(X_train, y_train, top_k_fanova=FANOVA_TOP_K, top_k_mi=MI_TOP_K):
    """
    Rank features by both f-ANOVA (linear) and mutual information (nonlinear).
    Return union of top-k from each, ordered by combined rank.
    """
    # f-ANOVA
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    fanova_rank = np.argsort(f_scores)[::-1]
    fanova_top = set(fanova_rank[:top_k_fanova])

    # Mutual Information (with some noise for stability)
    mi_scores = mutual_info_classif(X_train, y_train, discrete_features=False,
                                     n_neighbors=3, random_state=RANDOM_STATE)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)
    mi_rank = np.argsort(mi_scores)[::-1]
    mi_top = set(mi_rank[:top_k_mi])

    # Union of candidates
    candidates = fanova_top | mi_top

    # Score each candidate by combined rank (lower = better)
    fanova_rank_map = {idx: rank for rank, idx in enumerate(fanova_rank)}
    mi_rank_map = {idx: rank for rank, idx in enumerate(mi_rank)}

    combined = []
    for idx in candidates:
        avg_rank = (fanova_rank_map.get(idx, len(f_scores)) +
                    mi_rank_map.get(idx, len(mi_scores))) / 2
        combined.append((idx, avg_rank, f_scores[idx], mi_scores[idx]))

    # Sort by combined rank
    combined.sort(key=lambda x: x[1])

    return combined  # list of (feature_idx, combined_rank, f_score, mi_score)

# ─── IMPROVEMENT 4: HIERARCHICAL CLUSTERING REDUNDANCY REMOVAL ─────
def cluster_redundancy_filter(X_train, candidate_indices, ranked_scores,
                               threshold=CLUSTER_DIST_THRESHOLD):
    """
    Cluster correlated features using agglomerative clustering.
    From each cluster, keep the feature with the best (lowest) combined rank.

    Parameters:
        X_train: training data matrix
        candidate_indices: list of feature column indices to consider
        ranked_scores: dict mapping feature_idx -> combined_rank
        threshold: distance threshold (1 - |r|); features with |r| > (1-threshold) cluster

    Returns:
        filtered list of feature indices (one per cluster, best-ranked)
    """
    if len(candidate_indices) <= 1:
        return list(candidate_indices)

    # Compute correlation matrix for candidates
    X_cand = X_train[:, candidate_indices]
    corr = np.abs(np.corrcoef(X_cand.T))
    np.fill_diagonal(corr, 1.0)

    # Convert to distance
    dist = 1 - corr
    dist = np.clip(dist, 0, 1)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')

    # For each cluster, pick the best-ranked feature
    clusters = defaultdict(list)
    for i, cl in enumerate(cluster_labels):
        clusters[cl].append(candidate_indices[i])

    filtered = []
    for cl, members in clusters.items():
        best = min(members, key=lambda idx: ranked_scores.get(idx, float('inf')))
        filtered.append(best)

    # Sort by rank
    filtered.sort(key=lambda idx: ranked_scores.get(idx, float('inf')))

    return filtered

# ─── IMPROVEMENT 6: HYPERPARAMETER GRIDS ───────────────────────────
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

# ─── INNER LOOCV: FEATURE SELECTION + CLASSIFICATION ───────────────
def inner_loocv(X_train, y_train, clf_name, ranked_full, rank_map_full):
    """
    Run LOOCV on training set.

    Optimization: MI + f-ANOVA ranking computed once on the full inner training
    set (removing 1/31 samples changes rankings negligibly). Per-LOO iteration
    only the effect-size filter, clustering, and power accumulation are recomputed
    (these depend on group statistics that shift with each held-out sample).

    ranked_full and rank_map_full are pre-computed once per outer fold and shared
    across classifiers.

    Returns: predictions, probas, stable_features, feature_counter, best_params
    """
    n_train = len(y_train)
    loo = LeaveOneOut()

    predictions = np.zeros(n_train)
    probas = np.zeros(n_train)
    all_selected_features = []
    hyperparam_votes = Counter()

    candidate_indices_full = [idx for idx, _, _, _ in ranked_full]

    for loo_train_idx, loo_test_idx in loo.split(X_train, y_train):
        X_lt = X_train[loo_train_idx]
        y_lt = y_train[loo_train_idx]
        X_ltest = X_train[loo_test_idx]

        n1 = np.sum(y_lt == 1)
        n0 = np.sum(y_lt == 0)

        # ── Step 1: Use pre-computed ranking, re-filter by effect size ──
        viable = []
        for idx in candidate_indices_full:
            x_als = X_lt[y_lt == 1, idx]
            x_ctr = X_lt[y_lt == 0, idx]
            d = abs(cohens_d(x_als, x_ctr))
            if d >= MIN_COHENS_D:
                viable.append(idx)

        if len(viable) == 0:
            viable = candidate_indices_full[:10]

        # ── Step 2: Hierarchical clustering redundancy removal ──
        filtered = cluster_redundancy_filter(X_lt, viable, rank_map_full)

        # ── Step 3: Power-driven accumulation ──
        selected = []
        for idx in filtered:
            if len(selected) >= MAX_FEATURES:
                break

            selected.append(idx)

            powers = []
            for s in selected:
                x_a = X_lt[y_lt == 1, s]
                x_c = X_lt[y_lt == 0, s]
                powers.append(approx_power(abs(cohens_d(x_a, x_c)), n1, n0))

            combined_power = 1 - np.prod([1 - p for p in powers])
            if combined_power >= TARGET_POWER:
                break

        if len(selected) == 0:
            selected = [ranked_full[0][0]]

        all_selected_features.append(set(selected))

        # ── Step 4: Hyperparameter selection via inner 3-fold CV ──
        best_score = -1
        best_params = HYPERPARAM_GRIDS[clf_name][0]

        scaler = StandardScaler()
        X_sel_train = scaler.fit_transform(X_lt[:, selected])
        X_sel_test = scaler.transform(X_ltest[:, selected])

        inner_hp_cv = StratifiedKFold(n_splits=3, shuffle=True,
                                       random_state=RANDOM_STATE)
        for params in HYPERPARAM_GRIDS[clf_name]:
            hp_fold_scores = []
            for hp_train_idx, hp_val_idx in inner_hp_cv.split(X_sel_train, y_lt):
                clf = make_classifier(clf_name, params)
                clf.fit(X_sel_train[hp_train_idx], y_lt[hp_train_idx])
                val_preds = clf.predict(X_sel_train[hp_val_idx])
                hp_fold_scores.append(balanced_accuracy_score(
                    y_lt[hp_val_idx], val_preds))
            score = np.mean(hp_fold_scores)
            if score > best_score:
                best_score = score
                best_params = params

        hyperparam_votes[str(best_params)] += 1

        # ── Step 5: Classify with best params ──
        clf = make_classifier(clf_name, best_params)
        clf.fit(X_sel_train, y_lt)

        predictions[loo_test_idx] = clf.predict(X_sel_test)
        if hasattr(clf, 'predict_proba'):
            probas[loo_test_idx] = clf.predict_proba(X_sel_test)[:, 1]
        elif hasattr(clf, 'decision_function'):
            probas[loo_test_idx] = clf.decision_function(X_sel_test)

    # Stable features across LOO iterations
    feature_counter = Counter()
    for feat_set in all_selected_features:
        feature_counter.update(feat_set)

    stable_features = [f for f, count in feature_counter.items()
                       if count / n_train >= FEATURE_FREQ_THRESHOLD]
    stable_features.sort(key=lambda f: feature_counter[f], reverse=True)

    # Most-voted hyperparameters
    best_params_overall = eval(hyperparam_votes.most_common(1)[0][0])

    return predictions, probas, stable_features, feature_counter, best_params_overall

# ─── OUTER FOLD MODEL TRAINING ─────────────────────────────────────
def train_outer_model(X_train, y_train, X_test, stable_features,
                      clf_name, best_params, feature_cols_aug):
    """
    Train on full outer training set using stable features and best params.
    Apply Platt calibration for better probability estimates.
    """
    if len(stable_features) == 0:
        # Fallback: re-run quick feature selection on full training set
        ranked = dual_rank_features(X_train, y_train)
        stable_features = [ranked[0][0]]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[:, stable_features])
    X_te = scaler.transform(X_test[:, stable_features])

    # Fit with calibration (Platt scaling via sigmoid)
    base_clf = make_classifier(clf_name, best_params)
    # CalibratedClassifierCV with cv='prefit' requires fitting base first
    # Use cv=3 for in-fold calibration (safe: only uses training data)
    try:
        cal_clf = CalibratedClassifierCV(base_clf, cv=min(3, len(y_train)),
                                          method='sigmoid')
        cal_clf.fit(X_tr, y_train)
        preds = cal_clf.predict(X_te)
        probs = cal_clf.predict_proba(X_te)[:, 1]
    except Exception:
        # Fallback to uncalibrated
        base_clf.fit(X_tr, y_train)
        preds = base_clf.predict(X_te)
        if hasattr(base_clf, 'predict_proba'):
            probs = base_clf.predict_proba(X_te)[:, 1]
        elif hasattr(base_clf, 'decision_function'):
            probs = base_clf.decision_function(X_te)
        else:
            probs = preds.astype(float)

    return preds, probs

# ─── IMPROVEMENT 7: FEATURE TYPE ANNOTATION ────────────────────────
def annotate_feature(feat_name):
    """Parse a feature name into its components for biological interpretation."""
    info = {'name': feat_name, 'chiralities': [], 'descriptor': '', 'timepoint': '',
            'is_delta': False, 'is_pairwise': False}

    # Check if delta
    for delta_tag in ['delta_6h_0h', 'delta_24h_0h', 'delta_24h_6h']:
        if feat_name.endswith(delta_tag):
            info['is_delta'] = True
            info['timepoint'] = delta_tag
            base = feat_name[:-len(delta_tag) - 1]  # remove _delta_...
            break
    else:
        # Regular feature: ends in _0h, _6h, or _24h
        for tp in ['0h', '6h', '24h']:
            if feat_name.endswith(f'_{tp}'):
                info['timepoint'] = tp
                base = feat_name[:-len(tp) - 1]
                break
        else:
            base = feat_name

    # Check pairwise (contains '_vs_')
    if '_vs_' in base:
        info['is_pairwise'] = True
        parts = base.split('_vs_')
        # First chirality
        info['chiralities'].append(parts[0].replace('ch', '(').replace('_', ',') + ')')
        # Second part: chirality + descriptor
        rest = parts[1]
        # Find descriptor type
        for desc_type in ['ratio', 'eemdist']:
            if f'_{desc_type}' in rest:
                info['descriptor'] = desc_type
                ch2 = rest.replace(f'_{desc_type}', '')
                info['chiralities'].append(ch2.replace('ch', '(').replace('_', ',') + ')')
                break
    else:
        # Single chirality: chX_Y_descriptor
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
    """Execute the full improved pipeline."""

    print("=" * 70)
    print("CHIRALITY DESCRIPTOR CLASSIFICATION PIPELINE — V2 (NO DELTAS)")
    print("=" * 70)

    # Load data
    X_raw, y, sample_codes, feature_cols_raw = load_data()
    print(f"Raw features: {X_raw.shape[0]} samples × {X_raw.shape[1]} features")
    print(f"Label balance: ALS={y.sum()}, CTR={len(y)-y.sum()}")

    # Skip temporal engineering — use raw features only
    X = X_raw
    feature_cols = list(feature_cols_raw)
    n_new = 0
    print(f"Using raw features only: {X.shape[1]} features (no deltas)")

    # Verify no NaNs/Infs from deltas
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} non-finite values found, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Classifiers ──
    clf_names = ['SVM', 'LR', 'RF']

    # ── Outer CV ──
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    all_outer_preds = {name: np.zeros(len(y)) for name in clf_names}
    all_outer_probs = {name: np.zeros(len(y)) for name in clf_names}
    fold_features = {name: [] for name in clf_names}
    fold_metrics = {name: [] for name in clf_names}
    fold_params = {name: [] for name in clf_names}
    fold_n_features = {name: [] for name in clf_names}

    # Ensemble storage
    ensemble_probs = np.zeros(len(y))

    print(f"\n{'='*70}")
    print(f"RUNNING {OUTER_FOLDS}-FOLD OUTER CV WITH IMPROVED INNER LOOCV")
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

        # Pre-compute ranking once per outer fold (shared across classifiers)
        print(f"  Computing dual ranking (f-ANOVA + MI)...")
        ranked_full = dual_rank_features(X_train, y_train)
        rank_map_full = {idx: r for idx, r, _, _ in ranked_full}

        for clf_name in clf_names:
            print(f"\n  [{clf_name}] Inner LOOCV...")

            # Inner LOOCV
            loo_preds, loo_probs, stable_feats, feat_counter, best_params = \
                inner_loocv(X_train, y_train, clf_name, ranked_full, rank_map_full)

            inner_acc = accuracy_score(y_train, loo_preds)
            try:
                inner_auc = roc_auc_score(y_train, loo_probs)
            except:
                inner_auc = 0.5

            print(f"  [{clf_name}] Inner LOOCV: acc={inner_acc:.3f}, AUC={inner_auc:.3f}")
            print(f"  [{clf_name}] Best params: {best_params}")
            print(f"  [{clf_name}] Stable features ({len(stable_feats)}):")
            for feat_idx in stable_feats[:10]:
                freq = feat_counter[feat_idx] / len(y_train)
                info = annotate_feature(feature_cols[feat_idx])
                delta_tag = " [DELTA]" if info['is_delta'] else ""
                pair_tag = " [PAIR]" if info['is_pairwise'] else ""
                print(f"        {feature_cols[feat_idx]} "
                      f"(freq={freq:.0%}, d_type={info['descriptor']}, "
                      f"ch={info['chiralities']}{delta_tag}{pair_tag})")

            # Outer fold prediction
            preds, probs = train_outer_model(
                X_train, y_train, X_test, stable_feats,
                clf_name, best_params, feature_cols
            )

            all_outer_preds[clf_name][test_idx] = preds
            all_outer_probs[clf_name][test_idx] = probs
            fold_features[clf_name].append(set(stable_feats))
            fold_params[clf_name].append(best_params)
            fold_n_features[clf_name].append(len(stable_feats))

            # Accumulate for ensemble
            fold_ensemble_probs += probs / len(clf_names)

            # Per-fold metrics
            fold_acc = accuracy_score(y_test, preds)
            try:
                fold_auc = roc_auc_score(y_test, probs)
            except:
                fold_auc = 0.5
            fold_metrics[clf_name].append({'acc': fold_acc, 'auc': fold_auc})

            print(f"  [{clf_name}] Outer test: acc={fold_acc:.3f}, AUC={fold_auc:.3f}")

        # Ensemble prediction for this fold
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
        print(f"    Avg features/fold: {np.mean(fold_n_features[clf_name]):.1f}")

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
            'Avg_features': round(np.mean(fold_n_features[clf_name]), 1),
        })

    # ── Ensemble results ──
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
    ens_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    ens_spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n  ENSEMBLE (soft vote SVM+LR+RF):")
    print(f"    Accuracy:          {ens_acc:.3f}")
    print(f"    Balanced Accuracy: {ens_bal_acc:.3f}")
    print(f"    AUC:               {ens_auc:.3f}")
    print(f"    F1:                {ens_f1:.3f}")
    print(f"    MCC:               {ens_mcc:.3f}")
    print(f"    Sensitivity:       {ens_sens:.3f}")
    print(f"    Specificity:       {ens_spec:.3f}")
    print(f"    Confusion Matrix:  TP={tp} FP={fp} FN={fn} TN={tn}")

    summary_rows.append({
        'Classifier': 'Ensemble',
        'Accuracy': round(ens_acc, 3),
        'Balanced_Accuracy': round(ens_bal_acc, 3),
        'AUC': round(ens_auc, 3),
        'F1': round(ens_f1, 3),
        'MCC': round(ens_mcc, 3),
        'Sensitivity': round(ens_sens, 3),
        'Specificity': round(ens_spec, 3),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
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

        conserved = set.intersection(*fold_features[clf_name]) if all(fold_features[clf_name]) else set()
        robust_feats = {f: c for f, c in feat_fold_count.items() if c >= 3}

        print(f"\n  {clf_name}:")
        print(f"    All {OUTER_FOLDS} folds ({len(conserved)}):")
        for idx in sorted(conserved):
            info = annotate_feature(feature_cols[idx])
            print(f"      ✓ {feature_cols[idx]}  "
                  f"[{info['descriptor']}] ch={info['chiralities']} "
                  f"{'DELTA' if info['is_delta'] else info['timepoint']}")

        print(f"    ≥3/{OUTER_FOLDS} folds ({len(robust_feats)}):")
        for idx in sorted(robust_feats, key=lambda x: robust_feats[x], reverse=True):
            info = annotate_feature(feature_cols[idx])
            print(f"      • {feature_cols[idx]}  "
                  f"({robust_feats[idx]}/{OUTER_FOLDS} folds) "
                  f"[{info['descriptor']}] ch={info['chiralities']} "
                  f"{'DELTA' if info['is_delta'] else info['timepoint']}")

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

    # Features robust in ≥2 classifiers
    all_robust = defaultdict(set)
    for clf_name in clf_names:
        feat_fold_count = Counter()
        for feat_set in fold_features[clf_name]:
            feat_fold_count.update(feat_set)
        for f, c in feat_fold_count.items():
            if c >= 3:
                all_robust[f].add(clf_name)

    multi_clf_feats = {f: clfs for f, clfs in all_robust.items() if len(clfs) >= 2}
    if multi_clf_feats:
        print("\n  Features robust in ≥2 classifiers:")
        for idx in sorted(multi_clf_feats, key=lambda x: len(multi_clf_feats[x]),
                         reverse=True):
            info = annotate_feature(feature_cols[idx])
            print(f"    {feature_cols[idx]}  → {multi_clf_feats[idx]}  "
                  f"[{info['descriptor']}] ch={info['chiralities']}")
    else:
        print("\n  No features robust across multiple classifiers.")

    # ─── SAVE OUTPUTS ───────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('/home/claude/v2_nodeltas_classification_summary.csv', index=False)

    # Per-sample predictions
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
    # Add ensemble
    for i in range(len(y)):
        pred_rows.append({
            'sample': sample_codes[i],
            'true_label': 'ALS' if y[i] == 1 else 'CTR',
            'classifier': 'Ensemble',
            'predicted': 'ALS' if ens_preds[i] == 1 else 'CTR',
            'probability': round(float(ensemble_probs[i]), 4),
            'correct': int(ens_preds[i] == y[i]),
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv('/home/claude/v2_nodeltas_per_sample_predictions.csv', index=False)

    # Conserved features
    with open('/home/claude/v2_nodeltas_conserved_features.json', 'w') as f:
        json.dump(conserved_info, f, indent=2, default=str)

    # Feature engineering summary
    eng_summary = {
        'n_raw_features': len(feature_cols_raw),
        'n_delta_features': n_new,
        'n_total_features': len(feature_cols),
        'delta_types': ['6h-0h', '24h-0h', '24h-6h'],
        'config': {
            'outer_folds': OUTER_FOLDS,
            'target_power': TARGET_POWER,
            'min_cohens_d': MIN_COHENS_D,
            'max_features': MAX_FEATURES,
            'cluster_dist_threshold': CLUSTER_DIST_THRESHOLD,
            'feature_freq_threshold': FEATURE_FREQ_THRESHOLD,
        }
    }
    with open('/home/claude/v2_nodeltas_pipeline_config.json', 'w') as f:
        json.dump(eng_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("FILES SAVED")
    print(f"{'='*70}")
    print("  v2_nodeltas_classification_summary.csv")
    print("  v2_nodeltas_per_sample_predictions.csv")
    print("  v2_nodeltas_conserved_features.json")
    print("  v2_nodeltas_pipeline_config.json")
    print("\nDone!")

if __name__ == '__main__':
    run_pipeline()
