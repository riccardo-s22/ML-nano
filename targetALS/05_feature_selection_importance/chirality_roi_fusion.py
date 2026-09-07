#!/usr/bin/env python3
"""
chirality_roi_fusion.py
========================
Fuses autoencoder-derived consensus ROI masks with chirality physical
descriptors to create a biologically-grounded, data-driven feature set.

Logic:
  1. Load ROI masks (binary, per-timepoint) from the autoencoder pipeline.
  2. Map the 12 known DNA chirality positions (nm) → pixel coordinates.
  3. For each timepoint, determine which chirality peaks fall INSIDE
     the ROI (with a configurable tolerance radius).
  4. Keep only the Gaussian-fit descriptors for those "in-ROI" chiralities.
  5. Optionally add temporal deltas (6h−0h, 24h−0h) and pairwise ratios
     for the selected chiralities.
  6. Run strict leak-free nested LOOCV classification.
  7. Validate with bootstrap CI, permutation test, repeated k-fold.

This bridges the two worlds:
  - The autoencoder learns WHERE in the EEM matrix signal lives (ROIs).
  - The chirality descriptors tell us WHAT physical quantity changes there.

USAGE:
  python chirality_roi_fusion.py ^
    --roi_dir proxy_results_v14v3/consensus_union ^
    --descriptors_csv chirality_interp_descriptors.csv ^
    --labels_csv sample_labels.csv ^
    --eem_reference out_0h/1_3d__570_opj_with_emission.xlsx ^
    --output_dir chirality_roi_fusion_results

  # Or scan all 4 ROI strategies at once:
  python chirality_roi_fusion.py ^
    --roi_parent_dir proxy_results_v14v3 ^
    --descriptors_csv chirality_interp_descriptors.csv ^
    --labels_csv sample_labels.csv ^
    --eem_reference out_0h/1_3d__570_opj_with_emission.xlsx ^
    --output_dir chirality_roi_fusion_results

  # Or use continuous importance maps + percentile threshold:
  python chirality_roi_fusion.py ^
    --importance_maps importance_dir/final_consensus_0h.npy,importance_dir/final_consensus_6h.npy,importance_dir/final_consensus_24h.npy ^
    --importance_percentile 90 ^
    --descriptors_csv chirality_interp_descriptors.csv ^
    --labels_csv sample_labels.csv ^
    --eem_reference out_0h/1_3d__570_opj_with_emission.xlsx ^
    --output_dir chirality_roi_fusion_results
"""

import os, sys, argparse, warnings, json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, nct as nct_dist, t as t_dist
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

TP_NAMES = ["0h", "6h", "24h"]

CHIRALITY_POSITIONS = {
    "ch8_3":  {"emission_nm": 973.98,  "excitation_nm": 673.94},
    "ch6_5":  {"emission_nm": 987.82,  "excitation_nm": 577.12},
    "ch7_5":  {"emission_nm": 1047.81, "excitation_nm": 653.32},
    "ch10_2": {"emission_nm": 1080.60, "excitation_nm": 745.92},
    "ch9_4":  {"emission_nm": 1131.96, "excitation_nm": 731.39},
    "ch8_4":  {"emission_nm": 1130.34, "excitation_nm": 599.78},
    "ch7_6":  {"emission_nm": 1138.19, "excitation_nm": 659.79},
    "ch8_6":  {"emission_nm": 1200.03, "excitation_nm": 727.40},
    "ch8_7":  {"emission_nm": 1288.27, "excitation_nm": 740.87},
    "ch9_5":  {"emission_nm": 1262.98, "excitation_nm": 685.15},
    "ch10_3": {"emission_nm": 1267.70, "excitation_nm": 648.97},
    "ch10_5": {"emission_nm": 1282.97, "excitation_nm": 801.23},
}

# Descriptor names — auto-detected from the CSV at runtime.
# New Gaussian-fit naming (chirality_interpolated_extraction.py)
GAUSS_DESCRIPTORS = [
    "gauss_max", "gauss_em_center", "gauss_ex_center",
    "gauss_fwhm", "gauss_auc", "gauss_sigma_left", "gauss_sigma_right",
    "gauss_skew", "gauss_kurt",
    "gauss_r2", "gauss_rmse", "gauss_npts",
]

# Old patch-based naming (chirality_descriptor_extraction.py / chirality_full_descriptors.csv)
PATCH_DESCRIPTORS = [
    "mean", "max", "center", "median", "integral",
    "std", "range", "cv", "iqr",
    "prominence", "sharpness", "snr", "peak2ring", "contrast",
    "kurtosis", "skewness",
    "grad_center", "grad_mean",
    "fwhm_frac", "bg_integral",
]

# GoF descriptors to exclude from classification (diagnostics only)
GOF_DESCRIPTORS = {"gauss_r2", "gauss_rmse", "gauss_npts"}

CLF_COLORS = {"SVM": "#e74c3c", "LR": "#2980b9", "RF": "#27ae60"}


# ============================================================================
# Wavelength axis extraction
# ============================================================================

def load_eem_axes(filepath):
    """Extract emission and excitation wavelength axes from a sample EEM file."""
    from openpyxl import load_workbook
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    header = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    exc_ax = []
    for h in header[1:]:
        if h is None:
            exc_ax.append(np.nan)
        else:
            s = str(h)
            if "Excitation_" in s:
                s = s.split("Excitation_")[-1]
            try:
                exc_ax.append(float(s))
            except ValueError:
                exc_ax.append(np.nan)
    exc_ax = np.array(exc_ax, dtype=np.float64)

    em_ax = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        try:
            em_ax.append(float(row[0]) if row[0] is not None else np.nan)
        except (TypeError, ValueError):
            em_ax.append(np.nan)
    em_ax = np.array(em_ax, dtype=np.float64)
    return em_ax, exc_ax


# ============================================================================
# ROI loading
# ============================================================================

def load_roi_masks_from_dir(roi_dir):
    """Load binary ROI masks: roi_0h.npy, roi_6h.npy, roi_24h.npy."""
    rd = Path(roi_dir)
    masks = {}
    for tp in TP_NAMES:
        p = rd / f"roi_{tp}.npy"
        if p.exists():
            masks[tp] = np.load(p).astype(bool)
        else:
            raise FileNotFoundError(f"Missing {p}")
    return masks


def load_roi_masks_from_importance(map_paths, percentile=90):
    """Load continuous importance maps and threshold at given percentile."""
    masks = {}
    for tp, mp in zip(TP_NAMES, map_paths):
        imp = np.load(mp)
        thr = np.percentile(imp[imp > 0], percentile) if (imp > 0).any() else 0
        masks[tp] = imp >= thr
    return masks


# ============================================================================
# Chirality-in-ROI mapping
# ============================================================================

def nm_to_pixel(em_nm, ex_nm, em_ax, exc_ax):
    """Convert (emission_nm, excitation_nm) → (row_idx, col_idx)."""
    row = int(np.argmin(np.abs(em_ax - em_nm)))
    col = int(np.argmin(np.abs(exc_ax - ex_nm)))
    return row, col


def check_chirality_in_roi(masks, em_ax, exc_ax, radius_nm=10.0):
    """
    For each timepoint, check which chiralities fall inside the ROI mask
    (within a tolerance radius in nm).

    Returns:
      in_roi: dict[tp] → list of chirality names inside the ROI
      convergence: list of dicts with detailed overlap info
    """
    # Compute pixel spacing
    em_step = np.median(np.diff(em_ax))
    exc_step = np.median(np.diff(exc_ax))

    # Radius in pixels (asymmetric: em vs exc spacing differ)
    r_em_px = max(1, int(np.ceil(radius_nm / em_step)))
    r_exc_px = max(1, int(np.ceil(radius_nm / exc_step)))

    H = len(em_ax)
    W = len(exc_ax)

    in_roi = {tp: [] for tp in TP_NAMES}
    convergence = []

    for chir_name, pos in CHIRALITY_POSITIONS.items():
        em_nm = pos["emission_nm"]
        ex_nm = pos["excitation_nm"]
        row, col = nm_to_pixel(em_nm, ex_nm, em_ax, exc_ax)

        for tp in TP_NAMES:
            mask = masks[tp]
            # Check neighborhood: is any pixel within radius_nm in the ROI?
            r_lo = max(0, row - r_em_px)
            r_hi = min(H, row + r_em_px + 1)
            c_lo = max(0, col - r_exc_px)
            c_hi = min(W, col + r_exc_px + 1)

            patch = mask[r_lo:r_hi, c_lo:c_hi]
            n_pixels_in_roi = int(patch.sum())
            total_patch_pixels = patch.size
            overlap_frac = n_pixels_in_roi / total_patch_pixels if total_patch_pixels > 0 else 0.0
            is_inside = n_pixels_in_roi > 0

            if is_inside:
                in_roi[tp].append(chir_name)

            convergence.append({
                "chirality": chir_name,
                "timepoint": tp,
                "emission_nm": em_nm,
                "excitation_nm": ex_nm,
                "pixel_row": row,
                "pixel_col": col,
                "roi_pixels_in_patch": n_pixels_in_roi,
                "patch_total_pixels": total_patch_pixels,
                "overlap_fraction": overlap_frac,
                "in_roi": is_inside,
            })

    return in_roi, convergence


# ============================================================================
# Feature extraction from chirality descriptors CSV
# ============================================================================

def detect_descriptor_convention(desc_df):
    """Auto-detect which naming convention the descriptor CSV uses."""
    cols = set(desc_df.columns)
    chir_names = list(CHIRALITY_POSITIONS.keys())
    
    # Check for gauss_* style (new Gaussian-fit extraction)
    gauss_hits = sum(1 for c in cols 
                     if any(c.startswith(f"{ch}_gauss_") for ch in chir_names))
    # Check for patch style (old 20-descriptor extraction)
    patch_hits = sum(1 for c in cols 
                     if any(c.startswith(f"{ch}_mean_") for ch in chir_names))
    
    if gauss_hits > patch_hits:
        return "gauss", GAUSS_DESCRIPTORS
    else:
        return "patch", PATCH_DESCRIPTORS


def filter_descriptors(desc_df, in_roi, include_pairwise=True,
                       include_deltas=True, include_gof=False):
    """
    From the full chirality descriptor CSV, select only features
    belonging to chiralities inside the ROI at each timepoint.

    Args:
        desc_df: DataFrame with columns like ch8_3_gauss_max_0h or ch8_3_mean_0h
        in_roi: dict[tp] → list of chirality names
        include_pairwise: include ratio/eemdist for in-ROI pairs
        include_deltas: add temporal deltas (6h−0h, 24h−0h)
        include_gof: include goodness-of-fit cols (r2, rmse, npts)

    Returns:
        filtered DataFrame with code + selected features + optional deltas
    """
    # Auto-detect naming convention
    convention, all_descs = detect_descriptor_convention(desc_df)
    
    # Filter out GoF descriptors unless requested
    keep_descs = [d for d in all_descs
                  if include_gof or d not in GOF_DESCRIPTORS]

    selected_cols = ["code"]
    col_reasons = {}

    # Per-chirality features at their in-ROI timepoints
    for tp in TP_NAMES:
        for chir in in_roi[tp]:
            for desc in keep_descs:
                col = f"{chir}_{desc}_{tp}"
                if col in desc_df.columns:
                    selected_cols.append(col)
                    col_reasons[col] = f"{chir} in ROI at {tp}"

    # Pairwise features: only between chiralities that are BOTH in ROI at the same tp
    if include_pairwise:
        all_chir = list(CHIRALITY_POSITIONS.keys())
        all_pairs = list(combinations(all_chir, 2))
        for tp in TP_NAMES:
            chirs_tp = set(in_roi[tp])
            for ca, cb in all_pairs:
                if ca in chirs_tp and cb in chirs_tp:
                    for suffix in ["ratio", "eemdist"]:
                        col = f"{ca}_vs_{cb}_{suffix}_{tp}"
                        if col in desc_df.columns:
                            selected_cols.append(col)
                            col_reasons[col] = f"{ca}+{cb} both in ROI at {tp}"

    # Deduplicate (a chirality could be in ROI at multiple timepoints)
    seen = set()
    unique_cols = []
    for c in selected_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    selected_cols = unique_cols

    filtered = desc_df[selected_cols].copy()

    # Temporal deltas for chiralities in ROI
    if include_deltas:
        # Union of chiralities across all timepoints
        all_in_roi = set()
        for tp in TP_NAMES:
            all_in_roi.update(in_roi[tp])

        # Positional descriptors: skip for deltas (they don't change meaningfully)
        skip_delta = {"gauss_em_center", "gauss_ex_center", "center"}

        delta_cols = {}
        for chir in sorted(all_in_roi):
            for desc in keep_descs:
                if desc in skip_delta:
                    continue

                col_0h = f"{chir}_{desc}_0h"
                col_6h = f"{chir}_{desc}_6h"
                col_24h = f"{chir}_{desc}_24h"

                if col_6h in desc_df.columns and col_0h in desc_df.columns:
                    dcol = f"{chir}_{desc}_delta_6h_0h"
                    delta_cols[dcol] = desc_df[col_6h].values - desc_df[col_0h].values

                if col_24h in desc_df.columns and col_0h in desc_df.columns:
                    dcol = f"{chir}_{desc}_delta_24h_0h"
                    delta_cols[dcol] = desc_df[col_24h].values - desc_df[col_0h].values

        for dcol, vals in delta_cols.items():
            filtered[dcol] = vals

    return filtered, col_reasons


# ============================================================================
# Power analysis
# ============================================================================

def cohens_d(g0, g1):
    n0, n1 = len(g0), len(g1)
    if n0 < 2 or n1 < 2:
        return 0.0
    v0 = np.var(g0, ddof=1)
    v1 = np.var(g1, ddof=1)
    sp = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1))
    return float(abs(g0.mean() - g1.mean()) / sp) if sp > 1e-15 else 0.0


def power_from_d(d, n0, n1, alpha=0.05):
    if d < 1e-10:
        return alpha
    df = n0 + n1 - 2
    ncp = d * np.sqrt(n0 * n1 / (n0 + n1))
    tc = t_dist.ppf(1 - alpha / 2, df)
    return float(np.clip(1 - nct_dist.cdf(tc, df, ncp) + nct_dist.cdf(-tc, df, ncp), 0, 1))


# ============================================================================
# Leak-free nested LOOCV
# ============================================================================

def get_clf(name, svm_C=1.0, lr_C=1.0):
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, C=svm_C, gamma="scale",
                    class_weight="balanced", random_state=SEED)
    elif name == "LR":
        return LogisticRegression(max_iter=5000, C=lr_C, class_weight="balanced",
                                  solver="lbfgs", random_state=SEED)
    elif name == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=3,
                                      class_weight="balanced_subsample",
                                      random_state=SEED)
    raise ValueError(name)


def select_features_for_fold(X_train, y_train, max_k, spearman_thresh):
    """f-ANOVA ranking + Spearman collinearity filter, strictly on training data."""
    n_feat = X_train.shape[1]
    var = np.var(X_train, axis=0)
    F = np.zeros(n_feat)
    ok = var > 1e-15
    if ok.any():
        Fo, _ = f_classif(X_train[:, ok], y_train)
        F[ok] = np.nan_to_num(Fo, nan=0.0)

    rank = np.argsort(-F)
    selected = []

    for feat_idx in rank:
        feat_idx = int(feat_idx)
        if len(selected) >= max_k:
            break
        xc = X_train[:, feat_idx]
        skip = False
        for si in selected:
            rho, _ = spearmanr(xc, X_train[:, si])
            if not np.isnan(rho) and abs(rho) > spearman_thresh:
                skip = True
                break
        if not skip:
            selected.append(feat_idx)

    return selected


def run_strict_loocv(X, y, feat_names, max_features=15, spearman_thresh=0.70,
                     clf_names=None, svm_C=1.0, lr_C=1.0):
    """
    Strict nested LOOCV: feature selection + scaling + training all inside each fold.
    Evaluates multiple K values to find optimal number of features.
    """
    if clf_names is None:
        clf_names = ["SVM", "LR", "RF"]

    N = len(y)
    max_k = min(max_features, X.shape[1])

    # Track predictions for every K
    predictions = {cn: {k: np.zeros(N) for k in range(1, max_k + 1)} for cn in clf_names}
    preds_class = {cn: {k: np.zeros(N, dtype=int) for k in range(1, max_k + 1)} for cn in clf_names}

    # Track which features are selected in each fold (at max_k)
    fold_features = []

    for i in range(N):
        tr = np.ones(N, dtype=bool)
        tr[i] = False
        X_train, y_train = X[tr], y[tr]
        X_test = X[i:i+1]

        # Feature selection on training data only
        selected = select_features_for_fold(X_train, y_train, max_k, spearman_thresh)
        fold_features.append([feat_names[j] for j in selected])

        for k in range(1, min(max_k + 1, len(selected) + 1)):
            idx_k = selected[:k]
            Xtr_k = X_train[:, idx_k]
            Xte_k = X_test[:, idx_k]

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr_k)
            Xte_s = sc.transform(Xte_k)

            for cn in clf_names:
                clf = get_clf(cn, svm_C=svm_C, lr_C=lr_C)
                clf.fit(Xtr_s, y_train)
                predictions[cn][k][i] = clf.predict_proba(Xte_s)[0, 1]
                preds_class[cn][k][i] = clf.predict(Xte_s)[0]

    # Aggregate results per K
    steps = []
    for k in range(1, max_k + 1):
        step = {"K": k}
        for cn in clf_names:
            scores = predictions[cn][k]
            preds = preds_class[cn][k]
            try:
                auc = roc_auc_score(y, scores)
            except ValueError:
                auc = np.nan
            ba = balanced_accuracy_score(y, preds)
            tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
            step[f"{cn}_auc"] = float(auc)
            step[f"{cn}_balacc"] = float(ba)
            step[f"{cn}_sens"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
            step[f"{cn}_spec"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
            step[f"{cn}_scores"] = scores.copy()
        steps.append(step)

    return steps, fold_features


def pick_optimal_k(steps, clf_name="LR", method="max_auc"):
    """Pick K that maximizes AUC. Simple argmax (no power gate to avoid leakage)."""
    auc_key = f"{clf_name}_auc"
    best = max(steps, key=lambda s: s[auc_key])
    return best["K"]


# ============================================================================
# Validation
# ============================================================================

def bootstrap_auc_ci(y_true, y_scores, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    N = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    aucs = np.array(aucs)
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def permutation_test_strict(X, y, max_k, spearman_thresh, clf_name,
                            n_perms=200, observed_auc=None,
                            svm_C=1.0, lr_C=1.0):
    """Full permutation test: re-do feature selection in every fold of every permutation."""
    rng = np.random.RandomState(SEED)
    nulls = []
    N = len(y)

    for p in range(n_perms):
        if (p + 1) % 50 == 0:
            print(f"      Perm {p+1}/{n_perms}")
        yp = rng.permutation(y)
        scores = np.zeros(N)

        for i in range(N):
            tr = np.ones(N, dtype=bool)
            tr[i] = False
            X_train, y_train_p = X[tr], yp[tr]
            X_test = X[i:i+1]

            sel = select_features_for_fold(X_train, y_train_p, max_k, spearman_thresh)
            if not sel:
                scores[i] = 0.5
                continue

            Xtr_k = X_train[:, sel]
            Xte_k = X_test[:, sel]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr_k)
            Xte_s = sc.transform(Xte_k)

            clf = get_clf(clf_name, svm_C=svm_C, lr_C=lr_C)
            clf.fit(Xtr_s, y_train_p)
            scores[i] = clf.predict_proba(Xte_s)[0, 1]

        try:
            nulls.append(roc_auc_score(yp, scores))
        except ValueError:
            pass

    nulls = np.array(nulls)
    pval = float(np.mean(nulls >= (observed_auc or 0.5)))
    return {"p_value": pval, "null_aucs": nulls,
            "null_mean": float(nulls.mean()), "null_std": float(nulls.std())}


def repeated_kfold_strict(X, y, target_k, spearman_thresh, clf_names,
                          n_splits=5, n_repeats=10, svm_C=1.0, lr_C=1.0):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=SEED)
    results = {cn: {"aucs": [], "bas": []} for cn in clf_names}

    for train_idx, test_idx in rskf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        sel = select_features_for_fold(X_train, y_train, target_k, spearman_thresh)
        if not sel:
            continue

        Xtr_k = X_train[:, sel]
        Xte_k = X_test[:, sel]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_k)
        Xte_s = sc.transform(Xte_k)

        for cn in clf_names:
            clf = get_clf(cn, svm_C=svm_C, lr_C=lr_C)
            clf.fit(Xtr_s, y_train)
            probs = clf.predict_proba(Xte_s)[:, 1]
            preds = clf.predict(Xte_s)

            if len(np.unique(y_test)) == 2:
                results[cn]["aucs"].append(roc_auc_score(y_test, probs))
            results[cn]["bas"].append(balanced_accuracy_score(y_test, preds))

    summary = {}
    for cn in clf_names:
        aucs = np.array(results[cn]["aucs"])
        bas = np.array(results[cn]["bas"])
        summary[cn] = {
            "auc_mean": float(aucs.mean()) if len(aucs) else np.nan,
            "auc_std": float(aucs.std()) if len(aucs) else np.nan,
            "ba_mean": float(bas.mean()) if len(bas) else np.nan,
            "ba_std": float(bas.std()) if len(bas) else np.nan,
        }
    return summary


# ============================================================================
# Figures
# ============================================================================

def fig_convergence_heatmap(convergence_df, outpath):
    """Heatmap: chirality × timepoint, colored by overlap fraction."""
    chirs = list(CHIRALITY_POSITIONS.keys())
    pivot = convergence_df.pivot(index="chirality", columns="timepoint",
                                 values="overlap_fraction")
    pivot = pivot.reindex(index=chirs, columns=TP_NAMES)

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(TP_NAMES)))
    ax.set_xticklabels(TP_NAMES, fontsize=11)
    ax.set_yticks(range(len(chirs)))
    ax.set_yticklabels(chirs, fontsize=10)

    # Annotate
    in_roi_df = convergence_df.set_index(["chirality", "timepoint"])
    for i, chir in enumerate(chirs):
        for j, tp in enumerate(TP_NAMES):
            row = in_roi_df.loc[(chir, tp)]
            val = row["overlap_fraction"]
            inside = "✓" if row["in_roi"] else "✗"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{inside}\n{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="ROI overlap fraction", shrink=0.8)
    ax.set_title("Chirality ↔ ROI Convergence", fontsize=13, pad=10)
    ax.set_xlabel("Timepoint", fontsize=11)
    ax.set_ylabel("Chirality position", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_auc_vs_k(steps, clf_names, optimal_k, outpath):
    """AUC vs number of features."""
    ks = [s["K"] for s in steps]
    fig, ax = plt.subplots(figsize=(10, 5))
    for cn in clf_names:
        aucs = [s[f"{cn}_auc"] for s in steps]
        ax.plot(ks, aucs, "-o", color=CLF_COLORS.get(cn, "gray"),
                markersize=4, linewidth=1.5, label=f"{cn}")
    ax.axvline(optimal_k, color="gold", linewidth=2.5, alpha=0.7,
               label=f"Optimal K={optimal_k}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Number of features (K)", fontsize=12)
    ax.set_ylabel("AUC (strict nested LOOCV)", fontsize=12)
    ax.set_title("Chirality-in-ROI: LOOCV AUC vs Feature Count", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_roc(y, scores_dict, boot_cis, outpath):
    """ROC curves for all classifiers."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for cn, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        lo, hi = boot_cis[cn]
        ax.plot(fpr, tpr, color=CLF_COLORS.get(cn, "gray"), linewidth=2,
                label=f"{cn}  AUC={auc:.3f}  [{lo:.3f}–{hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Chirality-in-ROI: LOOCV ROC (95% Bootstrap CI)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_permutation(perm, obs_auc, clf_name, outpath):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(perm["null_aucs"], bins=40, alpha=0.7, color="#bdc3c7",
            edgecolor="black", linewidth=0.5)
    ax.axvline(obs_auc, color="red", linewidth=2.5,
               label=f"Observed AUC = {obs_auc:.3f}")
    ax.set_xlabel("AUC (permuted labels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Permutation Test ({clf_name}): p = {perm['p_value']:.4f}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Single-strategy runner
# ============================================================================

def run_one_strategy(strategy_name, masks, desc_df, labels_df, em_ax, exc_ax,
                     args, outdir):
    """Full pipeline for one set of ROI masks."""
    sd = outdir / strategy_name
    sd.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*70}")

    # --- Merge labels ---
    labels_df_c = labels_df.copy()
    labels_df_c["code"] = labels_df_c["code"].str.replace(".", "_", regex=False)
    desc_df_c = desc_df.copy()
    desc_df_c["code"] = desc_df_c["code"].str.replace(".", "_", regex=False)
    # Drop any existing group column in descriptors to avoid _x/_y conflict
    if "group" in desc_df_c.columns:
        desc_df_c = desc_df_c.drop(columns=["group"])
    merged = desc_df_c.merge(labels_df_c[["code", "group"]], on="code", how="inner")
    if "group" not in merged.columns or len(merged) == 0:
        raise RuntimeError("Merge failed — 0 matching codes between descriptors and labels. "
                           f"Desc codes: {desc_df_c['code'].head(3).tolist()}, "
                           f"Label codes: {labels_df_c['code'].head(3).tolist()}")
    y = (merged["group"].str.upper() == "ALS").astype(int).values
    codes = merged["code"].values
    n1, n0 = int(y.sum()), int((y == 0).sum())
    print(f"  Samples: {len(y)} ({n1} ALS, {n0} CTRL)")

    # --- ROI size ---
    for tp in TP_NAMES:
        m = masks[tp]
        pct = m.sum() / m.size * 100
        print(f"  ROI {tp}: {m.sum()} px ({pct:.1f}%)")

    # --- Chirality-in-ROI ---
    print(f"\n  Checking chirality positions against ROI (radius={args.roi_radius_nm}nm)...")
    in_roi, convergence = check_chirality_in_roi(
        masks, em_ax, exc_ax, radius_nm=args.roi_radius_nm)

    conv_df = pd.DataFrame(convergence)
    conv_df.to_csv(sd / "convergence.csv", index=False)
    fig_convergence_heatmap(conv_df, sd / "fig_convergence.png")

    total_in = sum(len(v) for v in in_roi.values())
    total_possible = len(CHIRALITY_POSITIONS) * len(TP_NAMES)
    print(f"\n  Convergence: {total_in}/{total_possible} chirality×timepoint slots in ROI")
    for tp in TP_NAMES:
        names = in_roi[tp]
        print(f"    {tp}: {len(names)}/12 → {', '.join(names) if names else '(none)'}")

    if total_in == 0:
        print("  ⚠ No chiralities inside ROI — skipping classification.")
        return None

    # --- Filter descriptors ---
    filtered, col_reasons = filter_descriptors(
        merged, in_roi,
        include_pairwise=args.include_pairwise,
        include_deltas=args.include_deltas,
        include_gof=False)

    feat_cols = [c for c in filtered.columns if c != "code"]
    X = filtered[feat_cols].values.astype(np.float64)
    X[np.isnan(X) | np.isinf(X)] = 0.0

    # Drop zero-variance
    var = np.var(X, axis=0)
    keep = var > 1e-15
    if (~keep).sum():
        print(f"  Dropped {(~keep).sum()} zero-var features")
    X = X[:, keep]
    feat_cols = [feat_cols[i] for i in range(len(feat_cols)) if keep[i]]

    n_base = sum(1 for c in feat_cols if "delta" not in c)
    n_delta = sum(1 for c in feat_cols if "delta" in c)
    print(f"  Features: {len(feat_cols)} total ({n_base} base + {n_delta} delta)")

    # --- Univariate analysis ---
    univar = []
    for j, col in enumerate(feat_cols):
        vals = X[:, j]
        d_val = cohens_d(vals[y == 1], vals[y == 0])
        try:
            from scipy.stats import ttest_ind
            _, pv = ttest_ind(vals[y == 1], vals[y == 0], equal_var=False)
        except Exception:
            pv = 1.0
        try:
            auc_u = roc_auc_score(y, vals)
            auc_u = max(auc_u, 1 - auc_u)
        except Exception:
            auc_u = 0.5
        univar.append({"feature": col, "cohens_d": d_val, "p_val": pv, "auc": auc_u})
    univar_df = pd.DataFrame(univar).sort_values("auc", ascending=False)
    univar_df.to_csv(sd / "univariate_analysis.csv", index=False)

    print(f"\n  Top 5 features by univariate AUC:")
    for _, r in univar_df.head(5).iterrows():
        print(f"    {r['feature']}: AUC={r['auc']:.3f}, d={r['cohens_d']:.3f}, p={r['p_val']:.4f}")

    # --- Strict nested LOOCV ---
    clf_names = ["SVM", "LR", "RF"]
    max_k = min(args.max_features, len(feat_cols))
    print(f"\n  Strict nested LOOCV (max K={max_k})...")

    steps, fold_feats = run_strict_loocv(
        X, y, feat_cols, max_features=max_k,
        spearman_thresh=args.spearman_thresh, clf_names=clf_names,
        svm_C=args.svm_C, lr_C=args.lr_C)

    # Pick optimal K (argmax AUC, no power gate to avoid leakage)
    optimal_k = pick_optimal_k(steps, clf_name="LR")
    opt_step = next(s for s in steps if s["K"] == optimal_k)

    pd.DataFrame([{k: v for k, v in s.items() if not k.endswith("_scores")}
                   for s in steps]).to_csv(sd / "loocv_k_sweep.csv", index=False)
    fig_auc_vs_k(steps, clf_names, optimal_k, sd / "fig_auc_vs_k.png")

    print(f"\n  Optimal K={optimal_k}")
    for cn in clf_names:
        print(f"    {cn}: AUC={opt_step[f'{cn}_auc']:.3f}  "
              f"BA={opt_step[f'{cn}_balacc']:.3f}  "
              f"Sens={opt_step[f'{cn}_sens']:.3f}  "
              f"Spec={opt_step[f'{cn}_spec']:.3f}")

    # --- Feature stability ---
    feat_counts = defaultdict(int)
    for ff in fold_feats:
        for f in ff[:optimal_k]:
            feat_counts[f] += 1
    stability_df = pd.DataFrame([
        {"feature": f, "count": c, "frac": c / len(y)}
        for f, c in sorted(feat_counts.items(), key=lambda x: -x[1])
    ])
    stability_df.to_csv(sd / "feature_stability.csv", index=False)
    print(f"\n  Most stable features (selected in >50% of folds):")
    for _, r in stability_df[stability_df["frac"] > 0.5].iterrows():
        print(f"    {r['feature']}: {r['frac']:.0%}")

    # --- Bootstrap CI ---
    print(f"\n  Bootstrap 95% CI...")
    boot_cis = {}
    scores_dict = {}
    for cn in clf_names:
        scores = opt_step[f"{cn}_scores"]
        lo, hi = bootstrap_auc_ci(y, scores)
        boot_cis[cn] = (lo, hi)
        scores_dict[cn] = scores
        print(f"    {cn}: [{lo:.3f}–{hi:.3f}]")

    fig_roc(y, scores_dict, boot_cis, sd / "fig_roc.png")

    # --- Repeated k-fold ---
    print(f"\n  Repeated 5-fold CV (10 repeats)...")
    kfold_res = repeated_kfold_strict(
        X, y, optimal_k, args.spearman_thresh, clf_names,
        svm_C=args.svm_C, lr_C=args.lr_C)
    for cn in clf_names:
        r = kfold_res[cn]
        print(f"    {cn}: AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}")

    # --- Permutation test ---
    best_cn = max(clf_names, key=lambda c: opt_step[f"{c}_auc"])
    obs_auc = opt_step[f"{best_cn}_auc"]
    print(f"\n  Permutation test ({args.n_perms} shuffles, {best_cn})...")
    perm = permutation_test_strict(
        X, y, optimal_k, args.spearman_thresh, best_cn,
        n_perms=args.n_perms, observed_auc=obs_auc,
        svm_C=args.svm_C, lr_C=args.lr_C)
    print(f"    p={perm['p_value']:.4f} (null={perm['null_mean']:.3f}±{perm['null_std']:.3f})")
    fig_permutation(perm, obs_auc, best_cn, sd / "fig_permutation.png")

    # --- Summary ---
    summary = {
        "strategy": strategy_name,
        "n_chiralities_in_roi": total_in,
        "n_features_total": len(feat_cols),
        "n_features_base": n_base,
        "n_features_delta": n_delta,
        "optimal_k": optimal_k,
        "roi_radius_nm": args.roi_radius_nm,
    }
    for cn in clf_names:
        summary[f"{cn}_loocv_auc"] = opt_step[f"{cn}_auc"]
        summary[f"{cn}_loocv_balacc"] = opt_step[f"{cn}_balacc"]
        summary[f"{cn}_kfold_auc"] = kfold_res[cn]["auc_mean"]
    summary[f"{best_cn}_perm_p"] = perm["p_value"]

    with open(sd / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fuse autoencoder ROIs with chirality physical descriptors")

    # ROI source (one of these required)
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument("--roi_dir",
        help="Directory with roi_0h.npy, roi_6h.npy, roi_24h.npy")
    roi_group.add_argument("--roi_parent_dir",
        help="Parent dir with strategy subdirs (consensus_union, etc.)")
    roi_group.add_argument("--importance_maps",
        help="Comma-separated paths to continuous importance maps (0h,6h,24h)")

    parser.add_argument("--importance_percentile", type=float, default=90,
        help="Percentile threshold for importance maps (default: 90)")

    # Data inputs
    parser.add_argument("--descriptors_csv", required=True,
        help="chirality_interp_descriptors.csv")
    parser.add_argument("--labels_csv", required=True,
        help="sample_labels.csv")
    parser.add_argument("--eem_reference", required=True,
        help="Any sample EEM xlsx (to get wavelength axes)")

    # ROI matching
    parser.add_argument("--roi_radius_nm", type=float, default=10.0,
        help="Tolerance radius (nm) for chirality-in-ROI check (default: 10)")

    # Feature options
    parser.add_argument("--include_pairwise", action="store_true", default=True,
        help="Include pairwise ratio/eemdist features")
    parser.add_argument("--no_pairwise", dest="include_pairwise", action="store_false")
    parser.add_argument("--include_deltas", action="store_true", default=True,
        help="Include temporal delta features")
    parser.add_argument("--no_deltas", dest="include_deltas", action="store_false")

    # Classification
    parser.add_argument("--max_features", type=int, default=15)
    parser.add_argument("--spearman_thresh", type=float, default=0.70)
    parser.add_argument("--svm_C", type=float, default=1.0)
    parser.add_argument("--lr_C", type=float, default=1.0)
    parser.add_argument("--n_perms", type=int, default=200)

    parser.add_argument("--output_dir", default="./chirality_roi_fusion_results")

    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CHIRALITY × ROI FUSION PIPELINE")
    print("=" * 70)

    # --- Load wavelength axes ---
    print(f"\nLoading EEM axes from: {args.eem_reference}")
    em_ax, exc_ax = load_eem_axes(args.eem_reference)
    print(f"  Emission: {em_ax[0]:.1f}–{em_ax[-1]:.1f} nm ({len(em_ax)} pts)")
    print(f"  Excitation: {exc_ax[0]:.1f}–{exc_ax[-1]:.1f} nm ({len(exc_ax)} pts)")

    # --- Load descriptors ---
    print(f"\nLoading chirality descriptors: {args.descriptors_csv}")
    desc_df = pd.read_csv(args.descriptors_csv)
    print(f"  Shape: {desc_df.shape}")

    # --- Load labels ---
    labels_df = pd.read_csv(args.labels_csv)
    print(f"  Labels: {len(labels_df)} samples")

    # --- Determine ROI source(s) ---
    strategies_to_run = {}

    if args.roi_dir:
        name = Path(args.roi_dir).name
        strategies_to_run[name] = load_roi_masks_from_dir(args.roi_dir)

    elif args.roi_parent_dir:
        pdir = Path(args.roi_parent_dir)
        # Look for strategy subdirectories with roi_*.npy
        for subdir in sorted(pdir.iterdir()):
            if subdir.is_dir():
                roi_files = list(subdir.glob("roi_*.npy"))
                if len(roi_files) >= 3:
                    try:
                        masks = load_roi_masks_from_dir(subdir)
                        strategies_to_run[subdir.name] = masks
                    except FileNotFoundError:
                        pass
        if not strategies_to_run:
            print(f"  No strategy dirs with roi_*.npy found in {pdir}")
            sys.exit(1)

    elif args.importance_maps:
        paths = [p.strip() for p in args.importance_maps.split(",")]
        if len(paths) != 3:
            print("ERROR: --importance_maps needs exactly 3 comma-separated paths")
            sys.exit(1)
        name = f"importance_p{int(args.importance_percentile)}"
        strategies_to_run[name] = load_roi_masks_from_importance(
            paths, percentile=args.importance_percentile)

    else:
        print("ERROR: Provide one of --roi_dir, --roi_parent_dir, or --importance_maps")
        sys.exit(1)

    print(f"\n  Strategies to evaluate: {list(strategies_to_run.keys())}")

    # --- Run each strategy ---
    all_summaries = []
    for sname, masks in strategies_to_run.items():
        result = run_one_strategy(
            sname, masks, desc_df, labels_df, em_ax, exc_ax, args, outdir)
        if result is not None:
            all_summaries.append(result)

    # --- Cross-strategy comparison ---
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("CROSS-STRATEGY COMPARISON")
        print(f"{'='*70}")
        comp_df = pd.DataFrame(all_summaries)
        comp_df.to_csv(outdir / "strategy_comparison.csv", index=False)

        for cn in ["SVM", "LR", "RF"]:
            col = f"{cn}_loocv_auc"
            if col in comp_df.columns:
                sub = comp_df[["strategy", col, f"{cn}_kfold_auc"]].sort_values(
                    col, ascending=False)
                print(f"\n  {cn}:")
                for _, r in sub.iterrows():
                    print(f"    {r['strategy']:<25s} LOOCV={r[col]:.3f}  "
                          f"5F-CV={r[f'{cn}_kfold_auc']:.3f}")

        best = comp_df.loc[comp_df["LR_loocv_auc"].idxmax()]
        print(f"\n  BEST: {best['strategy']} — LR LOOCV AUC={best['LR_loocv_auc']:.3f}")

    elif len(all_summaries) == 1:
        comp_df = pd.DataFrame(all_summaries)
        comp_df.to_csv(outdir / "strategy_comparison.csv", index=False)

    print(f"\nDone! Results in: {outdir}")


if __name__ == "__main__":
    main()
