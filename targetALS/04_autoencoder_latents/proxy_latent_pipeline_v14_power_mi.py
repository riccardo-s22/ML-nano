#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
proxy_latent_pipeline_v14_power_mi.py
======================================

V14 changes vs V13:
  - Adds POST-HOC POWER ANALYSIS to justify the number of latent features
    selected (Cohen's d effect size -> required N -> achieved power at actual N).
  - Adds MUTUAL INFORMATION (MI) based feature selection as an alternative /
    supplement to fANOVA + collinearity pruning.  MI captures non-linear
    redundancy between features that Pearson correlation misses.
  - New --feature_selection_mode flag:
      * "fanova"  (default, same as V13): fANOVA top-k + greedy Pearson dedup.
      * "mi"      (NEW): MI-based non-redundant feature selection (mRMR-style).
      * "fanova_mi" (NEW): fANOVA pre-filter -> MI-based redundancy pruning.
  - Power analysis outputs saved to power_analysis.csv and printed to console.
  - MI diagnostics saved to mi_redundancy_matrix.csv.
  - All V13 logic (leak-free ROIs, nested Ridge, classifiers) unchanged.

Dependencies (new):
  - scipy.stats (already present)
  - sklearn.feature_selection.mutual_info_classif (already available)

Run example:
  python proxy_latent_pipeline_v14_power_mi.py ^
    --arch_file "...\\conv_autoencoder_detailed.py" ^
    --model_dir "...\\5_fold_models_original" ^
    --tp_dirs  "...\\out_0h,...\\out_6h,...\\out_24h" ^
    --labels_csv "...\\sample_labels.csv" ^
    --output_dir "...\\proxy_latent_results_v14" ^
    --roi_mode loocv ^
    --dim_selection_mode reference ^
    --top_k_per_fold 10 ^
    --feature_selection_mode fanova_mi ^
    --mi_n_neighbors 3 ^
    --mi_redundancy_thresh 0.5
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy import stats

import torch
import torch.nn as nn

from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
)

import warnings
warnings.filterwarnings("ignore")

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Utilities (unchanged from V13)
# =============================================================================

def import_arch_module(arch_file: str):
    spec = importlib.util.spec_from_file_location("cnt_arch_module", arch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import architecture file: {arch_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_excel_as_array(filepath: str) -> np.ndarray:
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    data = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0:
            continue
        row_vals = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue
            if isinstance(cell, (int, float)) and cell is not None:
                row_vals.append(float(cell))
            else:
                row_vals.append(np.nan)
        data.append(row_vals)
    arr = np.array(data, dtype=np.float32)
    if np.isnan(arr).any():
        col_med = np.nanmedian(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_med, inds[1])
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def list_sample_codes(tp_dir: str) -> List[str]:
    codes = []
    for fn in os.listdir(tp_dir):
        if fn.lower().endswith(".xlsx") and not fn.startswith("~$"):
            codes.append(os.path.splitext(fn)[0])
    codes.sort()
    return codes


def load_all_samples(codes: List[str], tp_dirs: List[str]) -> np.ndarray:
    X_list = []
    for code in codes:
        tp_mats = []
        for tp in tp_dirs:
            fp = os.path.join(tp, f"{code}.xlsx")
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing file for code={code} at {fp}")
            tp_mats.append(load_excel_as_array(fp))
        X_list.append(np.stack(tp_mats, axis=0))
    return np.stack(X_list, axis=0)


def load_labels(labels_csv: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(labels_csv)
    if "code" not in df.columns:
        raise ValueError("labels_csv must contain a 'code' column.")
    if "group" not in df.columns:
        raise ValueError("labels_csv must contain a 'group' column (ALS/CTRL or 1/0).")
    codes = df["code"].astype(str).tolist()
    grp = df["group"].astype(str).values
    y = np.zeros(len(codes), dtype=int)
    for i, g in enumerate(grp):
        gg = g.strip().lower()
        if gg in ["als", "1", "case", "patient", "true", "yes"]:
            y[i] = 1
        elif gg in ["ctrl", "control", "0", "false", "no"]:
            y[i] = 0
        else:
            try:
                y[i] = int(float(g))
            except Exception:
                raise ValueError(f"Unrecognized group label '{g}' for code={codes[i]}")
    return codes, y, grp


def _load_state_dict_flexible(model: nn.Module, model_path: str):
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {type(state)}")
    try:
        model.load_state_dict(state, strict=True)
        return
    except Exception:
        pass
    cleaned = {}
    for k, v in state.items():
        kk = k
        for pref in ["module.", "model."]:
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) > 0:
        sys.stderr.write(
            f"[WARN] Missing keys when loading {model_path}: {missing[:10]}{'...' if len(missing)>10 else ''}\n"
        )


# =============================================================================
# Model utilities (unchanged from V13)
# =============================================================================

@torch.no_grad()
def compute_sqerr_per_sample_tp(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    X_5d = X.unsqueeze(2)
    recon_5d, _, _ = model(X_5d)
    sq = (recon_5d[:, :, 0] - X) ** 2
    return sq.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_z_agg(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    N, T, H, W = X.shape
    X_5d = X.unsqueeze(2)
    z_list = []
    for t in range(T):
        z_t = model.encoder(X_5d[:, t])
        z_list.append(z_t)
    z_stack = torch.stack(z_list, dim=1)
    z_agg, _ = model.attention(z_stack)
    return z_agg.detach().cpu().numpy()


def dim_importance_from_classifier_head(model: nn.Module, latent_dim: int) -> np.ndarray:
    lin = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.in_features == latent_dim:
            lin = m
            break
    if lin is None:
        raise RuntimeError("Could not find Linear(in_features=latent_dim) in classifier head.")
    w = lin.weight.detach().cpu().numpy()
    imp = np.sqrt(np.sum(w ** 2, axis=0))
    return imp


def get_top_dims_from_model(model: nn.Module, latent_dim: int, top_k: int) -> List[int]:
    imp = dim_importance_from_classifier_head(model, latent_dim)
    idx = np.argsort(-imp)[:top_k]
    return [int(i) for i in idx]


def get_top_dims_from_models(models: List[nn.Module], latent_dim: int, top_k_per_fold: int) -> List[int]:
    dims = set()
    for m in models:
        for d in get_top_dims_from_model(m, latent_dim, top_k_per_fold):
            dims.add(d)
    return sorted(list(dims))


# =============================================================================
# ROI computation (unchanged from V13)
# =============================================================================

def _valid_mask_from_mse(mse_map: np.ndarray, mse_max: float, mse_percentile_fallback: float) -> np.ndarray:
    valid = (mse_map < float(mse_max))
    if np.any(valid):
        return valid.astype(bool)
    thr = np.percentile(mse_map.reshape(-1), float(mse_percentile_fallback))
    valid = (mse_map <= thr)
    return valid.astype(bool)


def roi_from_linear_probe(
    X_tp_flat: np.ndarray,
    y_lat: np.ndarray,
    valid_mask_flat: np.ndarray,
    roi_percentile: float,
    alpha_probe: float = 1.0,
) -> np.ndarray:
    idx = np.flatnonzero(valid_mask_flat)
    if idx.size == 0:
        return np.zeros_like(valid_mask_flat, dtype=bool)

    Xv = X_tp_flat[:, idx].astype(np.float64)
    y = y_lat.astype(np.float64)

    mu = Xv.mean(axis=0, keepdims=True)
    sd = Xv.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs = (Xv - mu) / sd

    reg = Ridge(alpha=float(alpha_probe), fit_intercept=True, random_state=SEED)
    reg.fit(Xs, y)
    coef = np.abs(reg.coef_)

    if not np.isfinite(coef).all():
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

    thr = np.percentile(coef, float(roi_percentile))
    keep = coef >= thr

    roi = np.zeros_like(valid_mask_flat, dtype=bool)
    roi[idx[keep]] = True
    return roi


def compute_rois_from_subset(
    sqerr_by_model: np.ndarray,
    X_np: np.ndarray,
    Z_target: np.ndarray,
    dims: List[int],
    sample_idx: np.ndarray,
    roi_percentile: float,
    mse_max: float,
    mse_percentile: float,
    alpha_probe: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    tp_labels = ["0h", "6h", "24h"]
    _, _, _, H, W = sqerr_by_model.shape

    sqerr_sub = sqerr_by_model[:, sample_idx, :, :, :]
    X_sub = X_np[sample_idx]
    Z_sub = Z_target[sample_idx]
    N_sub = len(sample_idx)

    mse_maps = {}
    for t, tp in enumerate(tp_labels):
        mse_maps[tp] = sqerr_sub[:, :, t].mean(axis=(0, 1)).astype(np.float32)

    X_flat = X_sub.reshape(N_sub, 3, -1)
    latent_rois: Dict[int, Dict[str, np.ndarray]] = {d: {} for d in dims}

    for d in dims:
        y_lat = Z_sub[:, d].astype(np.float64)
        for t, tp in enumerate(tp_labels):
            valid = _valid_mask_from_mse(
                mse_maps[tp], mse_max=mse_max, mse_percentile_fallback=mse_percentile,
            )
            roi_flat = roi_from_linear_probe(
                X_tp_flat=X_flat[:, t, :],
                y_lat=y_lat,
                valid_mask_flat=valid.reshape(-1),
                roi_percentile=roi_percentile,
                alpha_probe=alpha_probe,
            )
            latent_rois[d][tp] = roi_flat.reshape(H, W)
    return latent_rois


def compute_global_rois(
    sqerr_by_model, X_np, Z_target, dims, roi_percentile, mse_max, mse_percentile, alpha_probe,
):
    all_idx = np.arange(X_np.shape[0])
    return compute_rois_from_subset(
        sqerr_by_model, X_np, Z_target, dims,
        sample_idx=all_idx,
        roi_percentile=roi_percentile,
        mse_max=mse_max,
        mse_percentile=mse_percentile,
        alpha_probe=alpha_probe,
    )


# =============================================================================
# Design matrix builders (unchanged from V13)
# =============================================================================

def ridge_design_matrix_for_dim_union(
    X_np: np.ndarray, masks_by_tp: Dict[str, np.ndarray], mode: str
) -> Optional[np.ndarray]:
    N, T, H, W = X_np.shape
    union = masks_by_tp["0h"] | masks_by_tp["6h"] | masks_by_tp["24h"]
    idx = np.flatnonzero(union.reshape(-1))
    if idx.size == 0:
        return None
    X0 = X_np[:, 0].reshape(N, -1)[:, idx]
    X6 = X_np[:, 1].reshape(N, -1)[:, idx]
    X24 = X_np[:, 2].reshape(N, -1)[:, idx]
    if mode == "delta":
        return np.concatenate([X6 - X0, X24 - X0], axis=1)
    return np.concatenate([X0, X6, X24], axis=1)


def ridge_design_matrix_for_dim_timepoint(
    X_np: np.ndarray, mask_tp: np.ndarray, tp_idx: int
) -> Optional[np.ndarray]:
    N, T, H, W = X_np.shape
    idx = np.flatnonzero(mask_tp.reshape(-1))
    if idx.size == 0:
        return None
    return X_np[:, tp_idx].reshape(N, -1)[:, idx]


def build_design_matrices(
    X_np, dims, latent_rois, proxy_feature_mode, ridge_input_mode,
):
    TP_NAMES = ["0h", "6h", "24h"]
    design, meta, feat_names = [], [], []

    if proxy_feature_mode == "per_dim":
        for d in dims:
            Xd = ridge_design_matrix_for_dim_union(X_np, latent_rois[d], mode=ridge_input_mode)
            design.append(Xd)
            meta.append({"dim": d, "tp": "all", "tp_idx": -1})
            feat_names.append(f"dim{d}")
    else:
        for d in dims:
            for tp_idx, tp_name in enumerate(TP_NAMES):
                Xd = ridge_design_matrix_for_dim_timepoint(X_np, latent_rois[d][tp_name], tp_idx=tp_idx)
                design.append(Xd)
                meta.append({"dim": d, "tp": tp_name, "tp_idx": tp_idx})
                feat_names.append(f"dim{d}_{tp_name}")

    return design, meta, feat_names


# =============================================================================
# Ridge proxy helpers (unchanged from V13)
# =============================================================================

def fit_predict_ridge(X_tr, y_tr, X_te, alpha):
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    Xs_tr = (X_tr - mu) / sd
    Xs_te = (X_te - mu) / sd
    reg = Ridge(alpha=float(alpha), fit_intercept=True, random_state=SEED)
    reg.fit(Xs_tr, y_tr)
    return reg.predict(Xs_tr), reg.predict(Xs_te)


def select_alpha_inner_cv(X_tr, y_tr, alpha_grid, inner_folds=3, metric="r2"):
    n = X_tr.shape[0]
    if n < 6 or len(alpha_grid) == 1:
        return float(alpha_grid[0])
    k = min(int(inner_folds), n)
    k = max(2, k)
    if k > n - 1:
        k = max(2, n - 1)
    if k < 2:
        return float(alpha_grid[0])
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    best_alpha = float(alpha_grid[0])
    best_score = -np.inf if metric == "r2" else np.inf
    for a in alpha_grid:
        fold_scores = []
        for tr_i, va_i in kf.split(np.arange(n)):
            _, pred_va = fit_predict_ridge(X_tr[tr_i], y_tr[tr_i], X_tr[va_i], alpha=float(a))
            yva = y_tr[va_i]
            if metric == "mse":
                s = float(np.mean((yva - pred_va) ** 2))
            else:
                ss_res = float(np.sum((yva - pred_va) ** 2))
                ss_tot = float(np.sum((yva - np.mean(yva)) ** 2))
                s = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            fold_scores.append(s)
        mean_s = float(np.mean(fold_scores))
        if metric == "mse":
            if mean_s < best_score:
                best_score = mean_s
                best_alpha = float(a)
        else:
            if mean_s > best_score:
                best_score = mean_s
                best_alpha = float(a)
    return best_alpha


# =============================================================================
# NEW: Power Analysis for latent feature count justification
# =============================================================================

def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD version) between two groups."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd < 1e-12:
        return 0.0
    return float(abs(np.mean(x1) - np.mean(x2)) / pooled_sd)


def power_two_sample_t(
    n1: int, n2: int, d: float, alpha: float = 0.05, alternative: str = "two-sided"
) -> float:
    """
    Post-hoc power for two-sample t-test given sample sizes and Cohen's d.
    
    Uses the noncentral t-distribution:
        ncp = d * sqrt(n1*n2 / (n1+n2))
        df  = n1 + n2 - 2
        power = P(T > t_crit | ncp) for one-sided, adjusted for two-sided.
    """
    if d < 1e-12:
        return alpha  # No effect => power = alpha (Type I error rate)
    
    df = n1 + n2 - 2
    ncp = d * np.sqrt(n1 * n2 / (n1 + n2))  # Noncentrality parameter
    
    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        # Power = P(|T| > t_crit | ncp)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    
    return float(np.clip(power, 0, 1))


def required_n_per_group(d: float, alpha: float = 0.05, power_target: float = 0.80) -> int:
    """
    Binary search for minimum n per group to achieve target power.
    For a two-sample t-test with equal groups.
    """
    if d < 1e-12:
        return 99999  # Undefined / enormous

    for n in range(3, 5000):
        p = power_two_sample_t(n, n, d, alpha=alpha, alternative="two-sided")
        if p >= power_target:
            return n
    return 5000  # Upper bound reached


def run_power_analysis(
    proxy_features: np.ndarray,
    y: np.ndarray,
    feat_names: List[str],
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> pd.DataFrame:
    """
    Post-hoc power analysis for each proxy feature.
    
    For each feature:
      1. Compute Cohen's d between ALS (y=1) and CTRL (y=0).
      2. Compute achieved power given actual group sizes.
      3. Compute required N per group for 80% power.
      4. Flag whether the feature is adequately powered.
    
    This helps justify how many latent features (top_k) are worth keeping:
    only features with adequate effect sizes are informative at our sample size.
    """
    idx_als = np.where(y == 1)[0]
    idx_ctrl = np.where(y == 0)[0]
    n_als, n_ctrl = len(idx_als), len(idx_ctrl)
    
    rows = []
    for j, fname in enumerate(feat_names):
        vals = proxy_features[:, j]
        ok = np.isfinite(vals)
        
        x_als = vals[np.intersect1d(idx_als, np.where(ok)[0])]
        x_ctrl = vals[np.intersect1d(idx_ctrl, np.where(ok)[0])]
        
        d = cohens_d(x_als, x_ctrl)
        
        # Classify effect size
        if d >= 0.8:
            effect_cat = "large"
        elif d >= 0.5:
            effect_cat = "medium"
        elif d >= 0.2:
            effect_cat = "small"
        else:
            effect_cat = "negligible"
        
        achieved_power = power_two_sample_t(len(x_als), len(x_ctrl), d, alpha=alpha)
        n_required = required_n_per_group(d, alpha=alpha, power_target=power_target)
        
        # Two-sample t-test (Welch's)
        if len(x_als) >= 2 and len(x_ctrl) >= 2 and np.std(x_als) > 1e-12:
            t_stat, t_pval = stats.ttest_ind(x_als, x_ctrl, equal_var=False)
        else:
            t_stat, t_pval = 0.0, 1.0
        
        # Mann-Whitney U (non-parametric alternative)
        if len(x_als) >= 2 and len(x_ctrl) >= 2:
            u_stat, u_pval = stats.mannwhitneyu(x_als, x_ctrl, alternative="two-sided")
        else:
            u_stat, u_pval = 0.0, 1.0
        
        rows.append({
            "feature": fname,
            "n_als": len(x_als),
            "n_ctrl": len(x_ctrl),
            "mean_als": float(np.mean(x_als)) if len(x_als) > 0 else np.nan,
            "mean_ctrl": float(np.mean(x_ctrl)) if len(x_ctrl) > 0 else np.nan,
            "cohens_d": d,
            "effect_category": effect_cat,
            "achieved_power": achieved_power,
            "n_per_group_for_80pct": n_required,
            "adequately_powered": achieved_power >= power_target,
            "welch_t": float(t_stat),
            "welch_p": float(t_pval),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(u_pval),
        })
    
    df = pd.DataFrame(rows).sort_values("cohens_d", ascending=False).reset_index(drop=True)
    return df


def summarize_power_analysis(power_df: pd.DataFrame, top_k: int) -> str:
    """Generate a textual summary for the power analysis results."""
    n_features = len(power_df)
    n_powered = int(power_df["adequately_powered"].sum())
    n_large = int((power_df["effect_category"] == "large").sum())
    n_medium = int((power_df["effect_category"] == "medium").sum())
    n_small = int((power_df["effect_category"] == "small").sum())
    n_negl = int((power_df["effect_category"] == "negligible").sum())
    
    lines = [
        f"  Total proxy features: {n_features} (from top_k={top_k} dims)",
        f"  Adequately powered (>=80%): {n_powered}/{n_features}",
        f"  Effect sizes: {n_large} large, {n_medium} medium, {n_small} small, {n_negl} negligible",
    ]
    
    if n_powered < n_features:
        underpowered = power_df[~power_df["adequately_powered"]]
        median_needed = int(underpowered["n_per_group_for_80pct"].median())
        lines.append(f"  Underpowered features median N needed per group: {median_needed}")
    
    if n_powered > 0:
        powered_feats = power_df[power_df["adequately_powered"]]["feature"].tolist()
        lines.append(f"  Powered features: {', '.join(powered_feats[:10])}" +
                     ("..." if len(powered_feats) > 10 else ""))
    
    # Recommendation
    if n_powered == 0:
        lines.append("  RECOMMENDATION: No features adequately powered at current N. "
                      "Consider reducing top_k or increasing sample size.")
    elif n_powered < n_features // 2:
        lines.append(f"  RECOMMENDATION: Only {n_powered}/{n_features} features powered. "
                     f"Consider reducing top_k to ~{max(1, n_powered // 3)} dims "
                     f"(yielding ~{max(1, n_powered)} proxy features).")
    else:
        lines.append(f"  RECOMMENDATION: {n_powered}/{n_features} features adequately powered. "
                     f"Current top_k seems justified.")
    
    return "\n".join(lines)


# =============================================================================
# NEW: Mutual Information (MI) based feature selection
# =============================================================================

def compute_mi_with_target(X: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    """
    Compute MI(feature_j, y) for each feature using sklearn's KNN-based MI estimator.
    Returns array of shape [n_features].
    """
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mi = mutual_info_classif(
        X_clean, y, discrete_features=False,
        n_neighbors=n_neighbors, random_state=SEED
    )
    return mi


def compute_mi_pairwise(X: np.ndarray, n_neighbors: int = 3) -> np.ndarray:
    """
    Compute pairwise MI between all features: MI(f_i, f_j).
    Uses the continuous MI estimator from sklearn (treating each feature as target).
    
    Returns symmetric matrix [n_features, n_features].
    """
    from sklearn.feature_selection import mutual_info_regression
    
    n_feat = X.shape[1]
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mi_matrix = np.zeros((n_feat, n_feat), dtype=float)
    
    for j in range(n_feat):
        # MI of all features with feature j (treated as continuous target)
        target_j = X_clean[:, j]
        if np.std(target_j) < 1e-12:
            continue
        # Exclude feature j from X to avoid MI(j,j) bias, then reinsert
        mi_vals = mutual_info_regression(
            X_clean, target_j, discrete_features=False,
            n_neighbors=n_neighbors, random_state=SEED
        )
        mi_matrix[:, j] = mi_vals
    
    # Symmetrize (average of MI(i,j) and MI(j,i))
    mi_matrix = (mi_matrix + mi_matrix.T) / 2.0
    return mi_matrix


def select_features_mrmr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_features: int = 10,
    n_neighbors: int = 3,
    mi_redundancy_thresh: float = 0.0,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selection using MI.
    
    Algorithm:
    1. Compute MI(feature, label) for all features -> relevance scores.
    2. Greedily select features that maximize relevance while minimizing
       average MI with already-selected features (redundancy).
    
    Score for candidate f given selected set S:
        mRMR_score(f) = MI(f, y) - (1/|S|) * sum_{s in S} MI(f, s)
    
    Args:
        X_train: Training features [n_samples, n_features]
        y_train: Training labels [n_samples]
        max_features: Max features to select
        n_neighbors: KNN parameter for MI estimation
        mi_redundancy_thresh: If > 0, also hard-threshold: skip if avg MI
                              with selected features exceeds this.
    
    Returns:
        (selected_indices, mi_relevance, mi_pairwise_matrix)
    """
    n_feat = X_train.shape[1]
    if n_feat == 0:
        return [], np.array([]), np.array([[]])
    
    max_features = min(max_features, n_feat)
    
    # Step 1: Relevance = MI(feature, label)
    mi_relevance = compute_mi_with_target(X_train, y_train, n_neighbors=n_neighbors)
    
    # Step 2: Pairwise MI between features (for redundancy)
    mi_pairwise = compute_mi_pairwise(X_train, n_neighbors=n_neighbors)
    
    # Step 3: Greedy mRMR selection
    selected: List[int] = []
    remaining = set(range(n_feat))
    
    for step in range(max_features):
        best_idx = -1
        best_score = -np.inf
        
        for cand in remaining:
            relevance = mi_relevance[cand]
            
            if len(selected) == 0:
                redundancy = 0.0
            else:
                redundancy = float(np.mean([mi_pairwise[cand, s] for s in selected]))
            
            # Hard threshold: skip if too redundant with selected set
            if mi_redundancy_thresh > 0 and len(selected) > 0:
                if redundancy > mi_redundancy_thresh:
                    continue
            
            score = relevance - redundancy
            
            if score > best_score:
                best_score = score
                best_idx = cand
        
        if best_idx == -1:
            break  # All remaining candidates too redundant
        
        selected.append(best_idx)
        remaining.discard(best_idx)
    
    return selected, mi_relevance, mi_pairwise


def select_features_fanova_mi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fanova_k: int = 50,
    mi_max_features: int = 10,
    n_neighbors: int = 3,
    mi_redundancy_thresh: float = 0.5,
    min_keep: int = 5,
) -> Tuple[List[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Hybrid: fANOVA pre-filter (top fanova_k by F-statistic),
    then mRMR on the reduced set for MI-based redundancy pruning.
    
    Returns:
        (selected_indices_in_original_space, mi_relevance, mi_pairwise)
    """
    n_feat = X_train.shape[1]
    if n_feat == 0:
        return [], None, None
    
    # Step 1: fANOVA pre-filter
    var = np.var(X_train, axis=0)
    F = np.zeros(n_feat, dtype=float)
    ok = var > 1e-12
    if ok.any():
        F_ok, _ = f_classif(X_train[:, ok], y_train)
        F[ok] = np.nan_to_num(F_ok, nan=0.0)
    
    order = np.argsort(-F)
    prefilter_idx = order[:min(fanova_k, n_feat)]
    
    # Step 2: mRMR on the pre-filtered subset
    X_sub = X_train[:, prefilter_idx]
    selected_in_sub, mi_rel, mi_pair = select_features_mrmr(
        X_sub, y_train,
        max_features=mi_max_features,
        n_neighbors=n_neighbors,
        mi_redundancy_thresh=mi_redundancy_thresh,
    )
    
    # Map back to original indices
    selected_original = [int(prefilter_idx[i]) for i in selected_in_sub]
    
    if len(selected_original) < min_keep:
        # Fall back: take top min_keep from fANOVA
        selected_original = [int(i) for i in prefilter_idx[:min(min_keep, len(prefilter_idx))]]
    
    return sorted(set(selected_original)), mi_rel, mi_pair


# =============================================================================
# Original fANOVA feature selection (from V13)
# =============================================================================

def fanova_scores(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[1]
    F = np.zeros(n, dtype=float)
    p = np.ones(n, dtype=float)
    var = np.var(X, axis=0)
    ok = var > 1e-12
    if ok.any():
        F_ok, p_ok = f_classif(X[:, ok], y)
        F[ok] = F_ok
        p[ok] = p_ok
    return F, p


def select_features_fanova_collinear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fanova_k: int = 50,
    collinear_thresh: float = 0.95,
    min_keep: int = 5,
) -> List[int]:
    if X_train.shape[1] == 0:
        return []
    F, _ = fanova_scores(X_train, y_train)
    order = np.argsort(-F)
    fanova_k = max(1, int(fanova_k))
    order = order[:min(fanova_k, order.size)]

    keep: List[int] = []
    for idx in order:
        idx = int(idx)
        if len(keep) == 0:
            keep.append(idx)
            continue
        x = X_train[:, idx]
        is_ok = True
        for j in keep:
            r = np.corrcoef(x, X_train[:, j])[0, 1]
            if np.isnan(r):
                continue
            if abs(r) > float(collinear_thresh):
                is_ok = False
                break
        if is_ok:
            keep.append(idx)

    if len(keep) < int(min_keep):
        keep = [int(i) for i in order[:min(int(min_keep), order.size)]]
    keep = sorted(set(keep))
    return keep


# =============================================================================
# Unified feature selector dispatch
# =============================================================================

def select_features_dispatch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mode: str,
    fanova_k: int = 50,
    collinear_thresh: float = 0.95,
    min_keep: int = 5,
    mi_max_features: int = 10,
    mi_n_neighbors: int = 3,
    mi_redundancy_thresh: float = 0.5,
) -> List[int]:
    """
    Dispatch to the appropriate feature selection method.
    Returns selected feature indices.
    """
    if mode == "fanova":
        return select_features_fanova_collinear(
            X_train, y_train, fanova_k=fanova_k,
            collinear_thresh=collinear_thresh, min_keep=min_keep,
        )
    elif mode == "mi":
        selected, _, _ = select_features_mrmr(
            X_train, y_train, max_features=mi_max_features,
            n_neighbors=mi_n_neighbors,
            mi_redundancy_thresh=mi_redundancy_thresh,
        )
        if len(selected) < min_keep:
            # Fallback: fANOVA top-k
            selected = select_features_fanova_collinear(
                X_train, y_train, fanova_k=min_keep,
                collinear_thresh=1.0, min_keep=min_keep,
            )
        return sorted(set(selected))
    elif mode == "fanova_mi":
        selected, _, _ = select_features_fanova_mi(
            X_train, y_train, fanova_k=fanova_k,
            mi_max_features=mi_max_features,
            n_neighbors=mi_n_neighbors,
            mi_redundancy_thresh=mi_redundancy_thresh,
            min_keep=min_keep,
        )
        return selected
    else:
        raise ValueError(f"Unknown feature_selection_mode: {mode}")


# =============================================================================
# Evaluation helpers (unchanged from V13)
# =============================================================================

def summarize_binary_from_scores(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "acc": float(acc), "bal_acc": float(bal), "auc": float(auc),
        "sens": float(sens), "spec": float(spec),
    }


def jaccard(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# =============================================================================
# Argument parser
# =============================================================================

def build_argparser():
    p = argparse.ArgumentParser(
        description="Proxy Latent Feature Pipeline V14 (power analysis + MI feature selection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--arch_file", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--tp_dirs", type=str, required=True, help="Comma-separated: out_0h,out_6h,out_24h")
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--top_k_per_fold", type=int, default=10)
    p.add_argument(
        "--dim_selection_mode", type=str, default="union",
        choices=["union", "reference"],
    )

    # ROI mode (from V13)
    p.add_argument("--roi_mode", type=str, default="loocv", choices=["loocv", "global"])
    p.add_argument("--roi_percentile", type=float, default=90.0)
    p.add_argument("--mse_max", type=float, default=0.1)
    p.add_argument("--mse_percentile", type=float, default=50.0)
    p.add_argument("--alpha_probe", type=float, default=1.0)

    p.add_argument("--ridge_input_mode", type=str, default="concat", choices=["concat", "delta"])
    p.add_argument("--proxy_feature_mode", choices=["per_dim", "per_dim_tp"], default="per_dim_tp")

    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--ridge_alpha_grid", type=str, default="0.1,1,10,100,1000")
    p.add_argument("--ridge_inner_folds", type=int, default=3)
    p.add_argument("--ridge_metric", type=str, default="r2", choices=["r2", "mse"])
    p.add_argument("--disable_alpha_tuning", action="store_true")

    p.add_argument("--latent_target", type=str, default="ensemble_sign", choices=["ref", "ensemble_sign"])

    p.add_argument("--skip_roi_overlap", action="store_true")
    p.add_argument("--disable_feature_selection", action="store_true")

    # Feature selection mode (V14 NEW)
    p.add_argument(
        "--feature_selection_mode", type=str, default="fanova",
        choices=["fanova", "mi", "fanova_mi"],
        help="fanova: V13 fANOVA+collinearity. mi: pure mRMR. fanova_mi: hybrid (recommended).",
    )
    p.add_argument("--fanova_k", type=int, default=50)
    p.add_argument("--collinear_thresh", type=float, default=0.95)
    p.add_argument("--min_selected_features", type=int, default=5)

    # MI-specific parameters (V14 NEW)
    p.add_argument("--mi_max_features", type=int, default=10,
                   help="Max features to keep after MI-based mRMR selection.")
    p.add_argument("--mi_n_neighbors", type=int, default=3,
                   help="KNN neighbors for MI estimation (3-5 typical).")
    p.add_argument("--mi_redundancy_thresh", type=float, default=0.5,
                   help="Hard MI redundancy cutoff. Features with avg MI > thresh to "
                        "selected set are skipped. 0 = no hard threshold.")

    # Power analysis parameters (V14 NEW)
    p.add_argument("--power_alpha", type=float, default=0.05,
                   help="Significance level for power analysis.")
    p.add_argument("--power_target", type=float, default=0.80,
                   help="Target power level (default 80%%).")
    p.add_argument("--skip_power_analysis", action="store_true",
                   help="Skip post-hoc power analysis.")

    p.add_argument("--svm_C", type=float, default=1.0)
    p.add_argument("--rf_trees", type=int, default=500)
    p.add_argument("--lr_C", type=float, default=1.0)

    return p


# =============================================================================
# Main
# =============================================================================

def main():
    args = build_argparser().parse_args()

    tp_dirs = [s.strip() for s in args.tp_dirs.split(",") if s.strip()]
    if len(tp_dirs) != 3:
        raise ValueError("tp_dirs must contain exactly 3 comma-separated directories (0h,6h,24h).")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROXY LATENT FEATURE PIPELINE - V14 (Power Analysis + MI Feature Selection)")
    print("=" * 70)
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {outdir}")
    print(f"Top K per fold: {args.top_k_per_fold} | dim_selection_mode={args.dim_selection_mode}")
    print(f"ROI mode: {args.roi_mode}")
    if args.roi_mode == "loocv":
        print("  -> ROIs recomputed per LOOCV fold on TRAIN-ONLY data (no transductive leak)")
    else:
        print("  -> ROIs computed ONCE on full dataset (V12 behaviour, transductive)")
    print(f"ROI percentile: {args.roi_percentile} (top {100-args.roi_percentile:.1f}%)")
    print(f"MSE filter: MSE < {args.mse_max} (fallback percentile={args.mse_percentile})")
    print(f"ROI probe alpha: {args.alpha_probe}")
    print(f"Ridge input mode: {args.ridge_input_mode}")
    print(f"Proxy feature mode: {args.proxy_feature_mode}")
    print(f"Latent target: {args.latent_target}")
    if args.disable_alpha_tuning:
        print(f"Alpha tuning: OFF (alpha={args.ridge_alpha})")
    else:
        print(f"Alpha tuning: ON (grid={args.ridge_alpha_grid}, inner_folds={args.ridge_inner_folds}, metric={args.ridge_metric})")
    print(f"Feature selection: {'OFF' if args.disable_feature_selection else args.feature_selection_mode}")
    if not args.disable_feature_selection:
        if args.feature_selection_mode in ("fanova", "fanova_mi"):
            print(f"  fANOVA k={args.fanova_k}, collinear_thresh={args.collinear_thresh}")
        if args.feature_selection_mode in ("mi", "fanova_mi"):
            print(f"  MI: max_features={args.mi_max_features}, n_neighbors={args.mi_n_neighbors}, "
                  f"redundancy_thresh={args.mi_redundancy_thresh}")
        print(f"  min_keep={args.min_selected_features}")
    print(f"Power analysis: {'OFF' if args.skip_power_analysis else 'ON'} "
          f"(alpha={args.power_alpha}, target={args.power_target})")

    # =========================================================================
    # Load data
    # =========================================================================
    codes_csv, y, groups = load_labels(args.labels_csv)
    codes0 = set(list_sample_codes(tp_dirs[0]))
    codes1 = set(list_sample_codes(tp_dirs[1]))
    codes2 = set(list_sample_codes(tp_dirs[2]))
    common = codes0 & codes1 & codes2

    codes = [c for c in codes_csv if c in common]
    if len(codes) != len(codes_csv):
        missing = [c for c in codes_csv if c not in common]
        sys.stderr.write(f"[WARN] {len(missing)} codes dropped (missing from one or more tp dirs).\n")
        keep_mask = np.array([c in common for c in codes_csv], dtype=bool)
        y = y[keep_mask]
        groups = groups[keep_mask]

    sample_ids = list(codes)
    N = len(codes)
    print(f"\nSamples: {N} (ALS: {int(np.sum(y==1))}, CTRL: {int(np.sum(y==0))})")

    arch_mod = import_arch_module(args.arch_file)

    print("\nLoading spectral data...")
    X_np = load_all_samples(codes, tp_dirs)
    N2, T, H, W = X_np.shape
    assert N2 == N and T == 3
    print(f"  Shape: {X_np.shape}")

    for t, tp in enumerate(["0h", "6h", "24h"]):
        flat = X_np[:, t].reshape(N, -1)
        print(f"  {tp}: mean={flat.mean():.4f}, std={flat.std():.4f}, min={flat.min():.4f}, max={flat.max():.4f}")

    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)

    # =========================================================================
    # Step 1: Load models + precompute
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Load models + precompute sqerr + compute latent targets")
    print("=" * 70)

    fold_paths = sorted([p for p in Path(args.model_dir).glob("best_model_fold*.pth")])
    if len(fold_paths) == 0:
        raise FileNotFoundError(f"No best_model_fold*.pth found in {args.model_dir}")

    models: List[nn.Module] = []
    for fp in fold_paths:
        print(f"  Loading: {fp}")
        model = arch_mod.ConvAutoencoderWithAttention(in_channels=1, latent_dim=args.latent_dim, num_classes=2).to(DEVICE)
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 1, H, W), device=DEVICE)
            _ = model(dummy)
        _load_state_dict_flexible(model, str(fp))
        model.eval()
        models.append(model)

    print("  Precomputing squared error maps...")
    sqerr_by_model = np.stack([compute_sqerr_per_sample_tp(m, X) for m in models], axis=0)

    print("  Computing latent targets (z_agg)...")
    Zs = [compute_z_agg(m, X) for m in models]
    Z_ref = Zs[0]

    if args.latent_target == "ref" or len(Zs) == 1:
        Z_target = Z_ref
    else:
        Z_aligned = [Z_ref]
        for z in Zs[1:]:
            z2 = z.copy()
            for d in range(args.latent_dim):
                a = Z_ref[:, d]
                b = z2[:, d]
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    if np.corrcoef(a, b)[0, 1] < 0:
                        z2[:, d] = -z2[:, d]
            Z_aligned.append(z2)
        Z_target = np.mean(np.stack(Z_aligned, axis=0), axis=0)

    # Dim selection
    if args.dim_selection_mode == "reference":
        dims = get_top_dims_from_model(models[0], latent_dim=args.latent_dim, top_k=args.top_k_per_fold)
    else:
        dims = get_top_dims_from_models(models, latent_dim=args.latent_dim, top_k_per_fold=args.top_k_per_fold)
    dims = [int(d) for d in dims]
    print(f"  Selected latent dims: {len(dims)}")

    # Ridge alpha grid
    alpha_grid = [float(x.strip()) for x in str(args.ridge_alpha_grid).split(",") if x.strip()]
    if not alpha_grid:
        alpha_grid = [float(args.ridge_alpha)]

    # Feature names
    TP_NAMES = ["0h", "6h", "24h"]
    if args.proxy_feature_mode == "per_dim":
        feat_names = [f"dim{d}" for d in dims]
    else:
        feat_names = [f"dim{d}_{tp}" for d in dims for tp in TP_NAMES]
    P = len(feat_names)

    # Meta
    meta: List[Dict] = []
    if args.proxy_feature_mode == "per_dim":
        for d in dims:
            meta.append({"dim": d, "tp": "all", "tp_idx": -1})
    else:
        for d in dims:
            for tp_idx, tp_name in enumerate(TP_NAMES):
                meta.append({"dim": d, "tp": tp_name, "tp_idx": tp_idx})

    # =========================================================================
    # Step 2: Global ROIs (only if --roi_mode global)
    # =========================================================================
    global_design_full = None

    if args.roi_mode == "global":
        print("\n" + "=" * 70)
        print("STEP 2: Compute GLOBAL ROIs on FULL dataset (V12 mode)")
        print("=" * 70)

        latent_rois_global = compute_global_rois(
            sqerr_by_model=sqerr_by_model, X_np=X_np, Z_target=Z_target,
            dims=dims, roi_percentile=args.roi_percentile,
            mse_max=args.mse_max, mse_percentile=args.mse_percentile,
            alpha_probe=args.alpha_probe,
        )

        roi_rows = []
        for d in dims:
            for tp in TP_NAMES:
                cnt = int(latent_rois_global[d][tp].sum())
                roi_rows.append({"latent_dim": d, "timepoint": tp, "roi_pixels": cnt, "roi_frac": cnt / (H * W)})
        pd.DataFrame(roi_rows).to_csv(outdir / "roi_info_global.csv", index=False)
        print(f"  Saved: roi_info_global.csv")

        global_design_full, _, _ = build_design_matrices(
            X_np, dims, latent_rois_global, args.proxy_feature_mode, args.ridge_input_mode
        )
        empty_count = sum(1 for Xd in global_design_full if Xd is None or Xd.shape[1] == 0)
        print(f"  Prepared {len(global_design_full)} design matrices; empty={empty_count}")
    else:
        print("\n" + "=" * 70)
        print("STEP 2: ROIs will be computed PER LOOCV FOLD on train-only data")
        print("=" * 70)
        print(f"  {N} folds × {len(dims)} dims × 3 timepoints = {N * len(dims) * 3} ROI computations")

    # =========================================================================
    # Step 3+4: LOOCV
    # =========================================================================
    classifiers = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=float(args.svm_C), probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=int(args.rf_trees), random_state=SEED,
            n_jobs=-1, class_weight="balanced_subsample",
        ),
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=float(args.lr_C),
                                       class_weight="balanced", solver="liblinear",
                                       random_state=SEED)),
        ]),
    }

    print("\n" + "=" * 70)
    print("STEP 3/4: OUTER LOOCV (nested ridge proxies + in-fold FS + classification)")
    print("=" * 70)

    loo = LeaveOneOut()
    y_score = {name: np.full(N, np.nan, dtype=float) for name in classifiers}
    y_pred = {name: np.full(N, -1, dtype=int) for name in classifiers}

    proxy_oof = np.full((N, P), np.nan, dtype=float)
    alpha_oof = np.full((N, P), np.nan, dtype=float)

    fs_counts = {fn: 0 for fn in feat_names}
    fs_rows = []
    roi_pixel_counts_per_fold = []

    for fold_idx, (train_idx, test_idx_arr) in enumerate(loo.split(np.arange(N))):
        test_idx = int(test_idx_arr[0])
        if fold_idx % 10 == 0:
            print(f"  Outer fold {fold_idx+1}/{N} (held-out index {test_idx})...")

        # ----- Determine design matrices -----
        if args.roi_mode == "loocv":
            fold_rois = compute_rois_from_subset(
                sqerr_by_model=sqerr_by_model, X_np=X_np, Z_target=Z_target,
                dims=dims, sample_idx=train_idx,
                roi_percentile=args.roi_percentile,
                mse_max=args.mse_max, mse_percentile=args.mse_percentile,
                alpha_probe=args.alpha_probe,
            )

            for d in dims:
                for tp in TP_NAMES:
                    roi_pixel_counts_per_fold.append({
                        "fold": fold_idx, "held_out": test_idx,
                        "dim": d, "tp": tp,
                        "roi_pixels": int(fold_rois[d][tp].sum()),
                    })

            fold_design, _, _ = build_design_matrices(
                X_np, dims, fold_rois, args.proxy_feature_mode, args.ridge_input_mode
            )
        else:
            fold_design = global_design_full

        # ----- Ridge proxies -----
        Xtr_feat = np.zeros((len(train_idx), P), dtype=float)
        Xte_feat = np.zeros((1, P), dtype=float)

        for p_idx in range(P):
            d = int(meta[p_idx]["dim"])
            X_full = fold_design[p_idx]
            y_tr = Z_target[train_idx, d].astype(float)

            if X_full is None or X_full.shape[1] == 0:
                const = float(np.mean(y_tr)) if len(y_tr) > 0 else 0.0
                Xtr_feat[:, p_idx] = const
                Xte_feat[0, p_idx] = const
                alpha_oof[test_idx, p_idx] = float(args.ridge_alpha)
                continue

            X_tr = X_full[train_idx, :]
            X_te = X_full[test_idx:test_idx+1, :]

            if args.disable_alpha_tuning or len(alpha_grid) == 1:
                a_sel = float(args.ridge_alpha) if args.disable_alpha_tuning else float(alpha_grid[0])
            else:
                a_sel = select_alpha_inner_cv(
                    X_tr=X_tr, y_tr=y_tr, alpha_grid=alpha_grid,
                    inner_folds=int(args.ridge_inner_folds), metric=str(args.ridge_metric),
                )

            yhat_tr, yhat_te = fit_predict_ridge(X_tr, y_tr, X_te, alpha=a_sel)
            Xtr_feat[:, p_idx] = yhat_tr
            Xte_feat[0, p_idx] = float(yhat_te[0])
            alpha_oof[test_idx, p_idx] = float(a_sel)

        proxy_oof[test_idx, :] = Xte_feat[0, :]

        # ----- In-fold feature selection (V14: dispatch based on mode) -----
        keep_idx = list(range(P))
        if not args.disable_feature_selection:
            keep_idx = select_features_dispatch(
                X_train=Xtr_feat,
                y_train=y[train_idx],
                mode=args.feature_selection_mode,
                fanova_k=int(args.fanova_k),
                collinear_thresh=float(args.collinear_thresh),
                min_keep=int(args.min_selected_features),
                mi_max_features=int(args.mi_max_features),
                mi_n_neighbors=int(args.mi_n_neighbors),
                mi_redundancy_thresh=float(args.mi_redundancy_thresh),
            )
            if len(keep_idx) == 0:
                keep_idx = list(range(P))

            kept_names = [feat_names[i] for i in keep_idx]
            fs_rows.append({
                "fold": fold_idx,
                "held_out_index": test_idx,
                "n_keep": len(keep_idx),
                "kept_features": ";".join(kept_names),
            })
            for fn in kept_names:
                fs_counts[fn] += 1

        Xtr_sel = Xtr_feat[:, keep_idx]
        Xte_sel = Xte_feat[:, keep_idx]

        # ----- Classifiers -----
        for name, clf in classifiers.items():
            clf.fit(Xtr_sel, y[train_idx])
            if hasattr(clf, "predict_proba"):
                ppos = float(clf.predict_proba(Xte_sel)[0, 1])
            else:
                s = clf.decision_function(Xte_sel)
                ppos = float(np.asarray(s).reshape(-1)[0])
            y_score[name][test_idx] = ppos
            y_pred[name][test_idx] = 1 if ppos >= 0.5 else 0

    # Sanity check
    for name in y_pred:
        if (y_pred[name] == -1).any():
            raise RuntimeError(f"Internal error: y_pred for {name} contains unset entries (-1).")

    # =========================================================================
    # Save outputs
    # =========================================================================
    df_proxy = pd.DataFrame(proxy_oof, columns=feat_names)
    df_proxy.insert(0, "sample_id", sample_ids)
    proxy_path = outdir / "proxy_feature_matrix_ridge_oof.csv"
    df_proxy.to_csv(proxy_path, index=False)
    print(f"\nSaved: {proxy_path}")

    df_alpha = pd.DataFrame(alpha_oof, columns=feat_names)
    df_alpha.insert(0, "sample_id", sample_ids)
    alpha_path = outdir / "ridge_alpha_selected_oof.csv"
    df_alpha.to_csv(alpha_path, index=False)
    print(f"Saved: {alpha_path}")

    # Feature selection artifacts
    if not args.disable_feature_selection:
        fs_counts_df = pd.DataFrame({
            "feature": list(fs_counts.keys()),
            "count": list(fs_counts.values()),
            "fraction": [c / N for c in fs_counts.values()],
        }).sort_values(["count", "feature"], ascending=[False, True])
        fs_counts_df.to_csv(outdir / "feature_selection_counts.csv", index=False)
        print(f"Saved: {outdir / 'feature_selection_counts.csv'}")

        pd.DataFrame(fs_rows).to_csv(outdir / "feature_selection_per_fold.csv", index=False)
        print(f"Saved: {outdir / 'feature_selection_per_fold.csv'}")

    # ROI diagnostics
    if args.roi_mode == "loocv" and roi_pixel_counts_per_fold:
        df_roi_folds = pd.DataFrame(roi_pixel_counts_per_fold)
        df_roi_folds.to_csv(outdir / "roi_pixel_counts_per_loocv_fold.csv", index=False)
        print(f"Saved: {outdir / 'roi_pixel_counts_per_loocv_fold.csv'}")

        summary = df_roi_folds.groupby(["dim", "tp"])["roi_pixels"].agg(["mean", "std", "min", "max"])
        summary.to_csv(outdir / "roi_pixel_counts_summary.csv")
        print(f"Saved: {outdir / 'roi_pixel_counts_summary.csv'}")

        mean_px = df_roi_folds["roi_pixels"].mean()
        std_px = df_roi_folds["roi_pixels"].std()
        print(f"  ROI pixels across folds: mean={mean_px:.0f}, std={std_px:.0f}")

    # Ridge proxy fit metrics
    fit_rows = []
    for p_idx in range(P):
        d = int(meta[p_idx]["dim"])
        tp = str(meta[p_idx]["tp"])
        y_true_z = Z_target[:, d].astype(float)
        y_hat = proxy_oof[:, p_idx].astype(float)

        ok = np.isfinite(y_true_z) & np.isfinite(y_hat)
        if ok.sum() < 3 or np.std(y_hat[ok]) < 1e-12 or np.std(y_true_z[ok]) < 1e-12:
            r, r2v, msev = np.nan, np.nan, np.nan
        else:
            r = float(np.corrcoef(y_true_z[ok], y_hat[ok])[0, 1])
            r2v = float(r2_score(y_true_z[ok], y_hat[ok]))
            msev = float(mean_squared_error(y_true_z[ok], y_hat[ok]))

        fit_rows.append({"feature": feat_names[p_idx], "dim": d, "timepoint": tp,
                         "pearson_r": r, "r2": r2v, "mse": msev})

    ridge_fit_df = pd.DataFrame(fit_rows).sort_values(["r2", "pearson_r"], ascending=[False, False])
    ridge_fit_df.to_csv(outdir / "ridge_proxy_fit_metrics_oof.csv", index=False)
    print(f"Saved: {outdir / 'ridge_proxy_fit_metrics_oof.csv'}")

    # =========================================================================
    # NEW Step 5: Post-hoc Power Analysis
    # =========================================================================
    if not args.skip_power_analysis:
        print("\n" + "=" * 70)
        print("STEP 5: POST-HOC POWER ANALYSIS")
        print("=" * 70)

        power_df = run_power_analysis(
            proxy_features=proxy_oof,
            y=y,
            feat_names=feat_names,
            alpha=args.power_alpha,
            power_target=args.power_target,
        )
        power_df.to_csv(outdir / "power_analysis.csv", index=False)
        print(f"Saved: {outdir / 'power_analysis.csv'}")

        power_summary = summarize_power_analysis(power_df, top_k=args.top_k_per_fold)
        print(power_summary)

        # Also save summary as text
        with open(outdir / "power_analysis_summary.txt", "w") as f:
            f.write(f"Power Analysis Summary (alpha={args.power_alpha}, target_power={args.power_target})\n")
            f.write("=" * 60 + "\n")
            f.write(power_summary + "\n\n")
            f.write("Per-feature details:\n")
            f.write(power_df.to_string(index=False) + "\n")
        print(f"Saved: {outdir / 'power_analysis_summary.txt'}")

    # =========================================================================
    # NEW Step 6: Global MI Redundancy Diagnostics
    # =========================================================================
    if not args.disable_feature_selection and args.feature_selection_mode in ("mi", "fanova_mi"):
        print("\n" + "=" * 70)
        print("STEP 6: MI REDUNDANCY DIAGNOSTICS (on OOF proxy features)")
        print("=" * 70)

        # MI relevance: MI(feature, label)
        mi_rel = compute_mi_with_target(proxy_oof, y, n_neighbors=args.mi_n_neighbors)
        mi_rel_df = pd.DataFrame({
            "feature": feat_names,
            "mi_with_label": mi_rel,
        }).sort_values("mi_with_label", ascending=False)
        mi_rel_df.to_csv(outdir / "mi_relevance_scores.csv", index=False)
        print(f"Saved: {outdir / 'mi_relevance_scores.csv'}")
        print("  Top MI features:")
        for _, row in mi_rel_df.head(10).iterrows():
            print(f"    {row['feature']}: MI={row['mi_with_label']:.4f}")

        # MI pairwise redundancy matrix
        mi_pair = compute_mi_pairwise(proxy_oof, n_neighbors=args.mi_n_neighbors)
        mi_pair_df = pd.DataFrame(mi_pair, index=feat_names, columns=feat_names)
        mi_pair_df.to_csv(outdir / "mi_redundancy_matrix.csv")
        print(f"Saved: {outdir / 'mi_redundancy_matrix.csv'}")

        # Average redundancy per feature
        np.fill_diagonal(mi_pair, 0)
        avg_redundancy = mi_pair.mean(axis=1)
        red_df = pd.DataFrame({
            "feature": feat_names,
            "mi_relevance": mi_rel,
            "avg_mi_redundancy": avg_redundancy,
            "mrmr_score": mi_rel - avg_redundancy,
        }).sort_values("mrmr_score", ascending=False)
        red_df.to_csv(outdir / "mi_mrmr_scores.csv", index=False)
        print(f"Saved: {outdir / 'mi_mrmr_scores.csv'}")
        print("  Top mRMR features (relevance - redundancy):")
        for _, row in red_df.head(10).iterrows():
            print(f"    {row['feature']}: mRMR={row['mrmr_score']:.4f} "
                  f"(rel={row['mi_relevance']:.4f}, red={row['avg_mi_redundancy']:.4f})")

    # =========================================================================
    # Classification summary
    # =========================================================================
    rows = []
    for name in classifiers:
        s = summarize_binary_from_scores(y_true=y, y_score=y_score[name], thr=0.5)
        rows.append({"model": name, **s})
        pd.DataFrame({
            "sample_id": sample_ids,
            "y_true": y.astype(int),
            "y_score": y_score[name],
            "y_pred": y_pred[name].astype(int),
        }).to_csv(outdir / f"predictions_{name}.csv", index=False)

    df_res = pd.DataFrame(rows)
    df_res.to_csv(outdir / "classification_results.csv", index=False)

    print("\n" + "=" * 70)
    print(f"SUMMARY (LOOCV, roi_mode={args.roi_mode}, fs_mode={args.feature_selection_mode})")
    print("=" * 70)
    print(df_res.to_string(index=False))
    print(f"\nSaved: {outdir / 'classification_results.csv'}")

    # Save config
    import json
    config = {
        "version": "v14_power_mi",
        "roi_mode": args.roi_mode,
        "n_samples": N,
        "n_als": int(np.sum(y == 1)),
        "n_ctrl": int(np.sum(y == 0)),
        "dims": dims,
        "n_dims": len(dims),
        "roi_percentile": args.roi_percentile,
        "mse_max": args.mse_max,
        "mse_percentile_fallback": args.mse_percentile,
        "proxy_feature_mode": args.proxy_feature_mode,
        "ridge_input_mode": args.ridge_input_mode,
        "ridge_alpha_grid": alpha_grid,
        "latent_target": args.latent_target,
        "dim_selection_mode": args.dim_selection_mode,
        "top_k_per_fold": args.top_k_per_fold,
        "feature_selection_mode": args.feature_selection_mode,
        "mi_max_features": args.mi_max_features,
        "mi_n_neighbors": args.mi_n_neighbors,
        "mi_redundancy_thresh": args.mi_redundancy_thresh,
        "power_alpha": args.power_alpha,
        "power_target": args.power_target,
    }
    with open(outdir / "config_v14.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPipeline complete! Results saved to: {outdir}")


if __name__ == "__main__":
    main()
