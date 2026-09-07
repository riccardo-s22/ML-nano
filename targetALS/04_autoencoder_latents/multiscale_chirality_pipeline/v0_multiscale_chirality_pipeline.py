#!/usr/bin/env python3
"""
MULTI-SCALE CHIRALITY-ANCHORED FEATURE EXTRACTION PIPELINE
============================================================

Extracts physics-informed features from EEM spectroscopy data using
the 12 known carbon nanotube chirality positions as anchors.

Three scales of information:
1. LOCAL:    Spectral behavior at/near each chirality position
2. PAIRWISE: Interactions between chirality pairs (ratios, corridors)
3. TEMPORAL: Dynamic changes across 0h -> 6h -> 24h

Classification uses properly nested LOOCV (no data leakage).
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)


# =============================================================================
# CHIRALITY DEFINITIONS
# =============================================================================

CHIRALITIES = {
    '(8,3)':  {'emission': 973.98,  'excitation': 673.94, 'height': 0.480172},
    '(6,5)':  {'emission': 987.82,  'excitation': 577.12, 'height': 0.762043},
    '(7,5)':  {'emission': 1047.81, 'excitation': 653.32, 'height': 0.648413},
    '(10,2)': {'emission': 1080.60, 'excitation': 745.92, 'height': 0.388838},
    '(9,4)':  {'emission': 1131.96, 'excitation': 731.39, 'height': 0.553377},
    '(8,4)':  {'emission': 1130.34, 'excitation': 599.78, 'height': 0.590734},
    '(7,6)':  {'emission': 1138.19, 'excitation': 659.79, 'height': 0.724726},
    '(8,6)':  {'emission': 1200.03, 'excitation': 727.40, 'height': 0.318030},
    '(8,7)':  {'emission': 1288.27, 'excitation': 740.87, 'height': 0.133987},
    '(9,5)':  {'emission': 1262.98, 'excitation': 685.15, 'height': 0.189759},
    '(10,3)': {'emission': 1267.70, 'excitation': 648.97, 'height': 0.189759},
    '(10,5)': {'emission': 1282.97, 'excitation': 801.23, 'height': 0.100000},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_spectral_axes(filepath):
    """Extract emission and excitation wavelength axes from an Excel file."""
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    ex_re = re.compile(r'Excitation_\s*0*([0-9]+(?:\.[0-9]+)?)')
    header = rows[0]
    excitation = np.array([float(ex_re.search(str(h)).group(1))
                           for h in header[1:] if ex_re.search(str(h))])
    emission = np.array([float(r[0]) for r in rows[1:]
                         if isinstance(r[0], (int, float))])
    return emission, excitation


def load_eem_matrix(filepath):
    """Load EEM data matrix from Excel (raw values, no normalization)."""
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    data = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0:
            continue
        row_data = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue
            if isinstance(cell, (int, float)) and cell is not None:
                row_data.append(float(cell))
            else:
                row_data.append(0.0)
        if row_data:
            data.append(row_data)
    return np.array(data, dtype=np.float64)


def match_file_to_code(directory, code):
    """Find the Excel file matching a sample code in a directory."""
    # The code looks like "1.3d__570.opj_with_emission" and files like
    # "1_3d__570_opj_with_emission_0h.xlsx"
    # Extract the numeric prefix
    code_prefix = code.split('.')[0]  # e.g., "1" from "1.3d__570..."

    for f in sorted(os.listdir(directory)):
        if not f.endswith('.xlsx') or f.startswith('~$'):
            continue
        # Extract numeric prefix from filename
        f_prefix = f.split('_')[0].split('.')[0]
        # Handle special codes like P1, P2, F1, etc.
        if code_prefix == f_prefix:
            return os.path.join(directory, f)
        # Also try matching the full code pattern
        code_normalized = code.replace('.', '_')
        f_base = f.replace('.xlsx', '')
        if f_base.startswith(code_normalized):
            return os.path.join(directory, f)

    raise FileNotFoundError(f"No file for code '{code}' (prefix '{code_prefix}') in {directory}")


# =============================================================================
# CHIRALITY INDEX MAPPING
# =============================================================================

def map_chirality_to_indices(emission_axis, excitation_axis):
    """Map each chirality to its (row, col) index in the EEM matrix."""
    chir_indices = {}
    for name, coords in CHIRALITIES.items():
        em_idx = int(np.argmin(np.abs(emission_axis - coords['emission'])))
        exc_idx = int(np.argmin(np.abs(excitation_axis - coords['excitation'])))
        chir_indices[name] = {
            'em_idx': em_idx,
            'exc_idx': exc_idx,
            'em_nm': float(emission_axis[em_idx]),
            'exc_nm': float(excitation_axis[exc_idx]),
        }
    return chir_indices


# =============================================================================
# FEATURE EXTRACTION: LOCAL SCALE
# =============================================================================

def extract_local_features(matrix, chir_indices, emission_axis, excitation_axis,
                           em_radius_nm=15, exc_radius_nm=10):
    """
    Extract local spectral features around each chirality position.

    For each chirality:
    - peak_value: intensity at the exact chirality position
    - integrated: sum of intensities in the neighborhood patch
    - patch_mean: mean intensity in the neighborhood
    - patch_std: standard deviation (spectral shape proxy)
    - em_centroid_shift: shift of intensity centroid from expected emission
    - exc_centroid_shift: shift of intensity centroid from expected excitation
    """
    H, W = matrix.shape
    em_step = float(np.median(np.diff(emission_axis)))
    exc_step = float(np.median(np.diff(excitation_axis)))

    em_radius_px = max(1, int(round(em_radius_nm / em_step)))
    exc_radius_px = max(1, int(round(exc_radius_nm / exc_step)))

    features = {}
    for name, idx in chir_indices.items():
        em_i = idx['em_idx']
        exc_i = idx['exc_idx']

        em_lo = max(0, em_i - em_radius_px)
        em_hi = min(H, em_i + em_radius_px + 1)
        exc_lo = max(0, exc_i - exc_radius_px)
        exc_hi = min(W, exc_i + exc_radius_px + 1)

        patch = matrix[em_lo:em_hi, exc_lo:exc_hi]
        peak_val = matrix[em_i, exc_i]

        integrated = float(np.sum(patch))
        patch_mean = float(np.mean(patch))
        patch_std = float(np.std(patch))

        # Emission centroid shift
        em_weights = np.sum(np.abs(patch), axis=1)
        em_range = emission_axis[em_lo:em_hi]
        if em_weights.sum() > 0:
            em_centroid = float(np.average(em_range, weights=np.abs(em_weights)))
            em_shift = em_centroid - CHIRALITIES[name]['emission']
        else:
            em_shift = 0.0

        # Excitation centroid shift
        exc_weights = np.sum(np.abs(patch), axis=0)
        exc_range = excitation_axis[exc_lo:exc_hi]
        if exc_weights.sum() > 0:
            exc_centroid = float(np.average(exc_range, weights=np.abs(exc_weights)))
            exc_shift = exc_centroid - CHIRALITIES[name]['excitation']
        else:
            exc_shift = 0.0

        features[f'{name}_peak'] = peak_val
        features[f'{name}_integrated'] = integrated
        features[f'{name}_mean'] = patch_mean
        features[f'{name}_std'] = patch_std
        features[f'{name}_em_shift'] = em_shift
        features[f'{name}_exc_shift'] = exc_shift

    return features


# =============================================================================
# FEATURE EXTRACTION: PAIRWISE SCALE
# =============================================================================

def extract_pairwise_features(matrix, chir_indices, n_corridor_samples=20):
    """
    Extract pairwise interaction features between chirality pairs.

    For each pair:
    - intensity_ratio: ratio of peak values
    - corridor_mean: mean intensity along path connecting the two positions
    - corridor_std: variability along the corridor
    """
    features = {}
    chir_names = sorted(chir_indices.keys())

    for name_a, name_b in combinations(chir_names, 2):
        idx_a = chir_indices[name_a]
        idx_b = chir_indices[name_b]

        val_a = matrix[idx_a['em_idx'], idx_a['exc_idx']]
        val_b = matrix[idx_b['em_idx'], idx_b['exc_idx']]

        max_abs = max(abs(val_a), abs(val_b))
        min_abs = min(abs(val_a), abs(val_b))
        ratio = min_abs / max_abs if max_abs > 0 else 1.0

        # Corridor sampling
        em_points = np.linspace(idx_a['em_idx'], idx_b['em_idx'], n_corridor_samples)
        exc_points = np.linspace(idx_a['exc_idx'], idx_b['exc_idx'], n_corridor_samples)

        H, W = matrix.shape
        corridor_vals = []
        for em_pt, exc_pt in zip(em_points, exc_points):
            ei = int(round(np.clip(em_pt, 0, H - 1)))
            xi = int(round(np.clip(exc_pt, 0, W - 1)))
            corridor_vals.append(matrix[ei, xi])
        corridor_vals = np.array(corridor_vals)

        pair_key = f'{name_a}_{name_b}'
        features[f'pair_{pair_key}_ratio'] = ratio
        features[f'pair_{pair_key}_corr_mean'] = float(np.mean(corridor_vals))
        features[f'pair_{pair_key}_corr_std'] = float(np.std(corridor_vals))

    return features


# =============================================================================
# FEATURE EXTRACTION: TEMPORAL SCALE
# =============================================================================

def extract_temporal_features(local_0h, local_6h, local_24h):
    """
    Extract temporal dynamics from local chirality features.

    - delta_6h, delta_24h: absolute change
    - slope_24h: rate of change
    - ratio_6h, ratio_24h: fold change
    """
    temporal = {}
    for key in local_0h:
        v0 = local_0h[key]
        v6 = local_6h.get(key, 0.0)
        v24 = local_24h.get(key, 0.0)

        temporal[f'{key}_delta6'] = v6 - v0
        temporal[f'{key}_delta24'] = v24 - v0
        temporal[f'{key}_slope24'] = (v24 - v0) / 24.0

        if abs(v0) > 1e-30:
            temporal[f'{key}_ratio6'] = v6 / v0
            temporal[f'{key}_ratio24'] = v24 / v0
        else:
            temporal[f'{key}_ratio6'] = 1.0
            temporal[f'{key}_ratio24'] = 1.0

    return temporal


# =============================================================================
# FULL FEATURE EXTRACTION
# =============================================================================

def extract_all_features(data_dict, chir_indices, emission_axis, excitation_axis):
    """Extract multi-scale features for a single sample."""
    all_features = {}

    # LOCAL features per timepoint
    local_per_tp = {}
    for tp in ['0h', '6h', '24h']:
        local = extract_local_features(
            data_dict[tp], chir_indices, emission_axis, excitation_axis)
        local_per_tp[tp] = local
        for k, v in local.items():
            all_features[f'{k}_{tp}'] = v

    # PAIRWISE features per timepoint
    for tp in ['0h', '6h', '24h']:
        pairwise = extract_pairwise_features(data_dict[tp], chir_indices)
        for k, v in pairwise.items():
            all_features[f'{k}_{tp}'] = v

    # TEMPORAL features (local chirality features only)
    temporal = extract_temporal_features(
        local_per_tp['0h'], local_per_tp['6h'], local_per_tp['24h'])
    all_features.update(temporal)

    return all_features


# =============================================================================
# CLASSIFICATION WITH NESTED LOOCV
# =============================================================================

def classify_loocv(X, y, codes, groups, method='RF', n_features_select=None):
    """LOOCV classification with feature selection INSIDE the loop."""
    loo = LeaveOneOut()
    y_pred, y_score = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if n_features_select and n_features_select < X_train_s.shape[1]:
            selector = SelectKBest(f_classif, k=n_features_select)
            X_train_s = selector.fit_transform(X_train_s, y_train)
            X_test_s = selector.transform(X_test_s)

        if method == 'RF':
            clf = RandomForestClassifier(
                n_estimators=500, max_depth=3, min_samples_leaf=3,
                class_weight='balanced', random_state=SEED)
        elif method == 'SVM':
            clf = SVC(kernel='linear', C=1.0, class_weight='balanced',
                      probability=True, random_state=SEED)
        elif method == 'LR':
            clf = LogisticRegression(
                penalty='l2', C=0.1, max_iter=5000,
                class_weight='balanced', solver='liblinear', random_state=SEED)
        elif method == 'GBM':
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                min_samples_leaf=3, random_state=SEED)
        else:
            raise ValueError(f"Unknown method: {method}")

        clf.fit(X_train_s, y_train)
        y_pred.append(int(clf.predict(X_test_s)[0]))

        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X_test_s)[0]
            classes = list(clf.classes_)
            y_score.append(float(proba[classes.index(1)]) if 1 in classes else 0.5)
        elif hasattr(clf, 'decision_function'):
            y_score.append(float(clf.decision_function(X_test_s)[0]))
        else:
            y_score.append(float(y_pred[-1]))

    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = 0.5

    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        sens = spec = 0.0

    return {
        'method': method,
        'n_features_used': n_features_select or X.shape[1],
        'accuracy': acc, 'balanced_accuracy': bal_acc, 'auc': auc,
        'sensitivity': sens, 'specificity': spec,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'y_pred': y_pred, 'y_score': y_score,
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance(X, y, feature_names, top_n=30):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=3, min_samples_leaf=3,
        class_weight='balanced', random_state=SEED)
    rf.fit(X_s, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], importances[i]) for i in sorted_idx]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance(importances, output_path, top_n=25):
    names = [x[0] for x in importances[:top_n]]
    vals = [x[1] for x in importances[:top_n]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), vals, color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Feature Importance (RF)')
    ax.set_title('Top Features for ALS/CTRL Classification')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_feature_distributions(feature_df, importances, y, output_path, top_n=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (name, imp) in enumerate(importances[:top_n]):
        ax = axes[i // 3, i % 3]
        if name in feature_df.columns:
            vals_als = feature_df.loc[y == 1, name].values
            vals_ctrl = feature_df.loc[y == 0, name].values
            ax.hist(vals_als, alpha=0.6, label='ALS', color='red', bins=10)
            ax.hist(vals_ctrl, alpha=0.6, label='CTRL', color='blue', bins=10)
        ax.set_title(f'{name}\n(imp={imp:.4f})', fontsize=9)
        ax.legend(fontsize=8)
    plt.suptitle('Top Discriminative Features: ALS vs CTRL', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Multi-scale chirality-anchored feature extraction for ALS classification')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing timepoint xlsx files OR comma-sep 3 dirs')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='CSV with columns: code, group')
    parser.add_argument('--output_dir', type=str, default='./multiscale_results')
    parser.add_argument('--n_features_select', type=int, default=None,
                        help='Number of features to select in CV (None = try multiple)')
    parser.add_argument('--em_radius', type=float, default=15.0,
                        help='Emission radius in nm for local patches')
    parser.add_argument('--exc_radius', type=float, default=10.0,
                        help='Excitation radius in nm for local patches')
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-SCALE CHIRALITY-ANCHORED FEATURE PIPELINE")
    print("=" * 70)

    # --- Load labels ---
    labels_df = pd.read_csv(args.labels_csv)
    codes = labels_df['code'].astype(str).tolist()
    groups = labels_df['group'].tolist()
    y = np.array([1 if g == 'ALS' else 0 for g in groups], dtype=int)
    print(f"\nSamples: {len(codes)} (ALS: {y.sum()}, CTRL: {len(y) - y.sum()})")

    # --- Resolve timepoint directories ---
    data_parts = args.data_dir.split(',')
    if len(data_parts) == 3:
        tp_dirs = data_parts
    else:
        data_dir = Path(args.data_dir)
        all_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        tp_dirs = []
        for suffix in ['0h', '6h', '24h']:
            matches = [str(d) for d in all_dirs if suffix in d.name]
            if matches:
                tp_dirs.append(matches[0])
        if len(tp_dirs) != 3:
            # Maybe all files are in one directory with timepoint suffixes
            tp_dirs = [args.data_dir] * 3

    print(f"Timepoint directories:")
    for i, d in enumerate(tp_dirs):
        print(f"  {'0h 6h 24h'.split()[i]}: {d}")

    # --- Parse spectral axes ---
    first_file = None
    for f in sorted(os.listdir(tp_dirs[0])):
        if f.endswith('.xlsx') and not f.startswith('~$'):
            first_file = os.path.join(tp_dirs[0], f)
            break
    emission_axis, excitation_axis = parse_spectral_axes(first_file)
    print(f"\nSpectral grid: {len(emission_axis)} emission x {len(excitation_axis)} excitation")
    print(f"Emission: {emission_axis[0]:.1f} - {emission_axis[-1]:.1f} nm")
    print(f"Excitation: {excitation_axis[0]:.1f} - {excitation_axis[-1]:.1f} nm")

    # --- Map chiralities ---
    chir_indices = map_chirality_to_indices(emission_axis, excitation_axis)
    print(f"\n12 Chirality positions mapped:")
    for name, idx in sorted(chir_indices.items()):
        print(f"  {name:>7s}: Em={idx['em_nm']:.1f}nm (row {idx['em_idx']}), "
              f"Exc={idx['exc_nm']:.1f}nm (col {idx['exc_idx']})")

    # --- Extract features ---
    print(f"\nExtracting multi-scale features...")
    all_features_list = []
    feature_names = None

    for i, code in enumerate(codes):
        sample_data = {}
        for tp_label, tp_dir in zip(['0h', '6h', '24h'], tp_dirs):
            fp = match_file_to_code(tp_dir, code)
            sample_data[tp_label] = load_eem_matrix(fp)

        feats = extract_all_features(sample_data, chir_indices,
                                     emission_axis, excitation_axis)
        all_features_list.append(feats)
        if feature_names is None:
            feature_names = sorted(feats.keys())

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{len(codes)} samples processed...")

    print(f"  Done! {len(feature_names)} raw features per sample")

    # --- Build feature matrix ---
    X = np.array([[f[name] for name in feature_names] for f in all_features_list])

    # Remove zero-variance
    stds = X.std(axis=0)
    nonzero_mask = stds > 1e-30
    X_clean = X[:, nonzero_mask]
    clean_names = [n for n, m in zip(feature_names, nonzero_mask) if m]
    print(f"After removing zero-variance: {X_clean.shape[1]} features")

    # Remove highly correlated (r > 0.95)
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X_clean)
    corr = np.corrcoef(X_scaled.T)
    high_corr = set()
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[1]):
            if abs(corr[i, j]) > 0.95:
                high_corr.add(j)
    keep_mask = [i not in high_corr for i in range(len(clean_names))]
    X_final = X_clean[:, keep_mask]
    final_names = [n for n, k in zip(clean_names, keep_mask) if k]
    print(f"After removing correlated (r>0.95): {X_final.shape[1]} features")

    # --- Save feature matrix ---
    feature_df = pd.DataFrame(X_final, columns=final_names)
    feature_df.insert(0, 'code', codes)
    feature_df.insert(1, 'group', groups)
    feature_df.insert(2, 'label', y)
    feature_df.to_csv(outdir / 'feature_matrix.csv', index=False)

    # --- Feature importance ---
    print(f"\n{'=' * 70}")
    print("FEATURE IMPORTANCE")
    print(f"{'=' * 70}")
    importances = compute_feature_importance(X_final, y, final_names, top_n=30)
    print("\nTop 20 features:")
    for i, (name, imp) in enumerate(importances[:20]):
        print(f"  {i+1:2d}. {name:50s} {imp:.4f}")

    plot_feature_importance(importances, outdir / 'feature_importance.png')
    plot_top_feature_distributions(feature_df, importances, y, outdir / 'top_feature_distributions.png')

    # --- Classification ---
    print(f"\n{'=' * 70}")
    print("CLASSIFICATION (LOOCV - No Data Leakage)")
    print(f"{'=' * 70}")

    feature_counts = [10, 20, 30, 50, None]
    all_results = []

    for n_feat in feature_counts:
        n_label = n_feat if n_feat else X_final.shape[1]
        print(f"\n--- Top {n_label} features ---")
        for method in ['RF', 'SVM', 'LR', 'GBM']:
            result = classify_loocv(X_final, y, codes, groups,
                                    method=method, n_features_select=n_feat)
            result['n_features_total'] = X_final.shape[1]
            all_results.append(result)
            print(f"  {method:4s}: Acc={result['accuracy']:.3f}, "
                  f"BalAcc={result['balanced_accuracy']:.3f}, "
                  f"AUC={result['auc']:.3f}, "
                  f"Sens={result['sensitivity']:.3f}, "
                  f"Spec={result['specificity']:.3f}")

            pred_df = pd.DataFrame({
                'code': codes, 'group': groups, 'y_true': y,
                'y_pred': result['y_pred'], 'y_score': result['y_score'],
            })
            pred_df.to_csv(outdir / f'predictions_{method}_top{n_label}.csv', index=False)

    # --- Summary ---
    summary_df = pd.DataFrame([{
        'method': r['method'], 'n_features': r['n_features_used'],
        'accuracy': r['accuracy'], 'balanced_accuracy': r['balanced_accuracy'],
        'auc': r['auc'], 'sensitivity': r['sensitivity'], 'specificity': r['specificity'],
    } for r in all_results])
    summary_df.to_csv(outdir / 'classification_results.csv', index=False)

    best = max(all_results, key=lambda x: x['auc'])
    print(f"\n{'=' * 70}")
    print(f"BEST: {best['method']} with {best['n_features_used']} features")
    print(f"  Acc={best['accuracy']:.3f}, BalAcc={best['balanced_accuracy']:.3f}, "
          f"AUC={best['auc']:.3f}")
    print(f"  Sens={best['sensitivity']:.3f}, Spec={best['specificity']:.3f}")
    print(f"{'=' * 70}")

    config = {
        'n_samples': len(codes), 'n_chiralities': 12,
        'n_features_raw': len(feature_names),
        'n_features_final': X_final.shape[1],
        'best_method': best['method'], 'best_auc': best['auc'],
    }
    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {outdir}")


if __name__ == '__main__':
    main()
