#!/usr/bin/env python3
"""
STEP 3+4 FUSED: ROI Extraction + Proxy Reconstruction with Percentile Optimization
====================================================================================

Fuses Step 3 (ROI extraction) and Step 4 (proxy reconstruction + classification)
into a single pipeline that sweeps mask_percentile to find the optimal ROI tightness.

Two percentile selection strategies (both computed for comparison):
  A) PROXY R² (fast, label-free): For each percentile, compute leave-one-out
     proxy reconstruction R² (how well the proxy recovers the true latent).
     Pick the percentile that maximizes mean R² across features.
     NO label leakage since R² only uses latent targets, not disease labels.

  B) NESTED LOOCV (leak-free, slower): Full double-nested LOOCV where the
     outer loop holds out one sample, and the inner loop (on N-1 samples)
     sweeps percentiles to pick the best by inner classification accuracy.
     The held-out sample is then evaluated using that fold's best percentile.
     Fully leak-free for both percentile selection AND classification.

Outputs:
  - percentile_sweep_r2.csv:         R² per feature per percentile
  - percentile_sweep_classification.csv: Classification metrics per percentile (quick sweep)
  - nested_loocv_results.csv:        Nested LOOCV per-fold percentile choices + predictions
  - final_classification_metrics.csv: Best results from both strategies
  - final_proxy_features.csv:        Proxy matrix at optimal percentile
  - percentile_sweep_summary.html:   Interactive dashboard

Pipeline: Step 1 -> Step OPT v2 -> Step 2 -> Step 3+4 Fused
                                                 |
                                     reads: selected_features_config.json

Usage:
  python step3_4_fused_proxy_pipeline.py \
      --tp_dirs ./out_0h,./out_6h,./out_24h \
      --labels_csv sample_labels.csv \
      --step1_dir ./step1_latent_features \
      --step2_dir ./step2_mse_maps \
      --model_dir ./5_fold_models_original \
      --opt_config ./opt_selection_results_v2/selected_features_config.json \
      --arch_file conv_autoencoder_detailed.py \
      --output_dir ./step3_4_fused_results \
      --percentiles 75,80,85,90,92,95,97
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
)
from scipy import ndimage
from scipy.stats import binom

import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TP_NAMES = ["0h", "6h", "24h"]


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path: str) -> Tuple[List[str], Dict[str, List[int]]]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config["target_features"], config["model_dims"]


def parse_target_features(feature_list: List[str]) -> Dict[str, List[int]]:
    needed = {}
    for f in feature_list:
        idx = f.rfind("_dim")
        if idx == -1:
            continue
        mname = f[:idx]
        dim_str = f[idx + 4:]
        if mname not in needed:
            needed[mname] = []
        needed[mname].append(int(dim_str))
    return needed


# =============================================================================
# Architecture Loading
# =============================================================================

def import_arch_module(arch_file: str):
    spec = importlib.util.spec_from_file_location("arch_module", arch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import architecture file: {arch_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_state_dict_flexible(model: nn.Module, path: str):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        elif k.startswith("model."):
            k = k[6:]
        new_state[k] = v
    try:
        model.load_state_dict(new_state, strict=True)
    except Exception:
        model.load_state_dict(new_state, strict=False)


# =============================================================================
# Data Loading
# =============================================================================

def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    codes = df["code"].tolist()
    groups = df["group"].values
    y = (groups == "ALS").astype(int)
    return codes, y, groups


def list_sample_codes(tp_dir: str) -> List[str]:
    codes = []
    for f in Path(tp_dir).glob("*.xlsx"):
        if f.name.startswith("~$"):
            continue
        name = f.stem
        for suffix in ["_0h", "_6h", "_24h"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        codes.append(name)
    return codes


def find_file_for_code(tp_dir: str, code: str) -> Optional[str]:
    tp_path = Path(tp_dir)
    exact_path = tp_path / f"{code}.xlsx"
    if exact_path.exists():
        return str(exact_path)
    for f in tp_path.glob("*.xlsx"):
        if f.name.startswith("~$"):
            continue
        stem = f.stem
        for suffix in ["_0h", "_6h", "_24h"]:
            if stem.endswith(suffix):
                file_code = stem[:-len(suffix)]
                if file_code == code:
                    return str(f)
                if file_code.replace('.', '_') == code:
                    return str(f)
    return None


def load_excel_as_array(filepath: str) -> np.ndarray:
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
    arr = np.array(data, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def load_all_samples(codes: List[str], tp_dirs: List[str]) -> np.ndarray:
    arrays = []
    for code in codes:
        tp_images = []
        for tp_dir in tp_dirs:
            fpath = find_file_for_code(tp_dir, code)
            if fpath is None:
                raise FileNotFoundError(f"Missing '{code}' in {tp_dir}")
            arr = load_excel_as_array(fpath)
            tp_images.append(arr)
        shapes = [a.shape for a in tp_images]
        if len(set(shapes)) > 1:
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            tp_images = [a[:min_h, :min_w] for a in tp_images]
        arrays.append(np.stack(tp_images, axis=0))
    return np.stack(arrays, axis=0)


# =============================================================================
# Importance Maps (computed ONCE, reused across percentile sweeps)
# =============================================================================

def compute_gradient_saliency(model: nn.Module, X: torch.Tensor,
                              dims: List[int]) -> Dict[int, Dict[int, np.ndarray]]:
    N, T, H, W = X.shape
    model.eval()
    grad_maps = {d: {t: np.zeros((H, W), dtype=np.float64) for t in range(T)} for d in dims}

    batch_size = min(8, N)
    n_batches = (N + batch_size - 1) // batch_size

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, N)
        B = e - s
        X_batch = X[s:e].clone().requires_grad_(True)
        X_5d = X_batch.unsqueeze(2)
        latents = [model.encoder(X_5d[:, t]) for t in range(T)]

        for d in dims:
            for t in range(T):
                target = latents[t][:, d].sum()
                model.zero_grad()
                if X_batch.grad is not None:
                    X_batch.grad.zero_()
                target.backward(retain_graph=True)
                grads = X_batch.grad.detach().abs()
                grad_maps[d][t] += grads[:, t].mean(dim=0).cpu().numpy() * B
        X_batch = X_batch.detach()

    for d in dims:
        for t in range(T):
            grad_maps[d][t] /= N
    return grad_maps


def compute_linear_probing(X: torch.Tensor, z_per_tp: Dict[str, np.ndarray],
                           dims: List[int]) -> Dict[int, Dict[int, np.ndarray]]:
    X_np = X.detach().cpu().numpy()
    N, T, H, W = X_np.shape
    linear_maps = {d: {t: np.zeros((H, W), dtype=float) for t in range(T)} for d in dims}
    scaler = StandardScaler()

    for t_idx, tp in enumerate(TP_NAMES):
        X_t = X_np[:, t_idx].reshape(N, -1)
        ok_cols = (X_t.var(axis=0) > 1e-12)
        X_t_valid = X_t[:, ok_cols]
        if X_t_valid.shape[1] == 0:
            continue
        Xs = scaler.fit_transform(X_t_valid)
        for d in dims:
            y_d = z_per_tp[tp][:, d]
            ridge = Ridge(alpha=100.0, fit_intercept=True)
            ridge.fit(Xs, y_d)
            coefs = np.abs(ridge.coef_)
            map_1d = np.zeros(H * W, dtype=float)
            map_1d[ok_cols] = coefs
            linear_maps[d][t_idx] = map_1d.reshape(H, W)
    return linear_maps


def normalize_map(m: np.ndarray) -> np.ndarray:
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-12:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def compute_consensus_maps(grad_maps: Dict, linear_maps: Dict,
                           dims: List[int], T: int = 3) -> Dict[int, Dict[int, np.ndarray]]:
    consensus = {d: {} for d in dims}
    for d in dims:
        for t in range(T):
            gm = normalize_map(grad_maps[d][t])
            lm = normalize_map(linear_maps[d][t])
            consensus[d][t] = (gm + lm) / 2.0
    return consensus


# =============================================================================
# ROI Extraction (parameterized by percentile)
# =============================================================================

def extract_roi_for_map(importance_map: np.ndarray, mse_map: np.ndarray,
                        importance_percentile: float, mse_threshold: float,
                        min_area: int = 3) -> np.ndarray:
    imp_thresh = np.percentile(importance_map, importance_percentile)
    roi_mask = (importance_map >= imp_thresh) & (mse_map < mse_threshold)
    labeled, n_comp = ndimage.label(roi_mask)
    filtered_mask = np.zeros_like(roi_mask)
    for cid in range(1, n_comp + 1):
        cmask = labeled == cid
        if cmask.sum() >= min_area:
            filtered_mask |= cmask
    return filtered_mask


def extract_all_rois_at_percentile(
    consensus_maps_all: Dict[str, Dict[int, Dict[int, np.ndarray]]],
    mse_ensemble: Dict[str, np.ndarray],
    model_dims: Dict[str, List[int]],
    percentile: float,
    mse_threshold: float,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Extract ROI masks for all models/dims/timepoints at a given percentile.
    Returns: {model_name: {dim: {tp_name: mask}}}
    """
    all_masks = {}
    for mname, dims in model_dims.items():
        if mname not in consensus_maps_all:
            continue
        all_masks[mname] = {}
        for d in dims:
            all_masks[mname][d] = {}
            for t_idx, tp_name in enumerate(TP_NAMES):
                imp_map = consensus_maps_all[mname][d][t_idx]
                mse_map = mse_ensemble[tp_name]
                mask = extract_roi_for_map(imp_map, mse_map, percentile, mse_threshold)
                all_masks[mname][d][tp_name] = mask
    return all_masks


# =============================================================================
# Ridge Proxy Helpers
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


def select_alpha_inner_cv(X_tr, y_tr, alpha_grid, inner_folds=3):
    n = X_tr.shape[0]
    if n < 6:
        return float(alpha_grid[0])
    k = min(int(inner_folds), n - 1)
    k = max(2, k)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    best_alpha = float(alpha_grid[0])
    best_score = -np.inf
    for a in alpha_grid:
        scores = []
        for tr_i, va_i in kf.split(np.arange(n)):
            _, pred_va = fit_predict_ridge(X_tr[tr_i], y_tr[tr_i], X_tr[va_i], alpha=a)
            ss_res = np.sum((y_tr[va_i] - pred_va) ** 2)
            ss_tot = np.sum((y_tr[va_i] - np.mean(y_tr[va_i])) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            scores.append(r2)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = float(a)
    return best_alpha


def calculate_binomial_power(accuracy: float, n_samples: int, alpha: float = 0.05) -> float:
    if accuracy <= 0.5:
        return 0.0
    n = int(n_samples)
    k_crit = int(binom.ppf(1.0 - alpha, n, 0.5)) + 1
    if k_crit > n:
        return 0.0
    return float(1.0 - binom.cdf(k_crit - 1, n, float(accuracy)))


# =============================================================================
# Model Bundle (lightweight: Z vectors only, masks injected per percentile)
# =============================================================================

class ModelBundle:
    """Holds latent vectors for one autoencoder fold model."""
    def __init__(self, model_name: str, top_dims: List[int],
                 z_agg: np.ndarray, z_per_tp: Dict[str, np.ndarray]):
        self.model_name = model_name
        self.top_dims = top_dims
        self.z_agg = z_agg          # [N, latent_dim]
        self.z_per_tp = z_per_tp    # {tp: [N, latent_dim]}


def load_bundles(step1_dir: Path, needed_dims: Dict[str, List[int]],
                 codes: List[str], latent_dim: int) -> List[ModelBundle]:
    """Load Z vectors for all needed models. ROI masks are handled separately."""
    ens_z_tp = {}
    for tp in TP_NAMES:
        p = step1_dir / f"ensemble_z_{tp}.csv"
        if p.exists():
            ens_z_tp[tp] = pd.read_csv(p)
    ens_z_agg_df = None
    if (step1_dir / "ensemble_z_agg.csv").exists():
        ens_z_agg_df = pd.read_csv(step1_dir / "ensemble_z_agg.csv")

    dim_cols = [f"dim{d}" for d in range(latent_dim)]
    bundles = []

    for mname, dims in needed_dims.items():
        s1_mdir = step1_dir / mname

        if (s1_mdir / "z_agg.csv").exists():
            df = pd.read_csv(s1_mdir / "z_agg.csv")
        elif ens_z_agg_df is not None:
            df = ens_z_agg_df
        else:
            raise FileNotFoundError(f"No z_agg for {mname}")

        z_codes = df["code"].astype(str).tolist()
        code_map = {c: i for i, c in enumerate(z_codes)}
        reindex = [code_map[c] for c in codes]
        z_agg = df[dim_cols].values[reindex]

        z_tp_full = {}
        for tp in TP_NAMES:
            if (s1_mdir / f"z_{tp}.csv").exists():
                df_tp = pd.read_csv(s1_mdir / f"z_{tp}.csv")
            elif tp in ens_z_tp:
                df_tp = ens_z_tp[tp]
            else:
                raise FileNotFoundError(f"Missing z_{tp} for {mname}")
            z_tp_full[tp] = df_tp[dim_cols].values[reindex]

        bundles.append(ModelBundle(mname, dims, z_agg, z_tp_full))
    return bundles


# =============================================================================
# Two-Stage Proxy (with externally provided masks)
# =============================================================================

def build_proxy_single_fold(
    X_np: np.ndarray, bundle: ModelBundle, train_idx: np.ndarray,
    test_idx: int, alpha_grid: List[float],
    masks: Dict[int, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build two-stage proxy for one LOOCV fold, one model bundle."""
    n_train = len(train_idx)
    n_dims = len(bundle.top_dims)
    proxies_tr = np.zeros((n_train, n_dims))
    proxies_te = np.zeros((1, n_dims))

    for i, d in enumerate(bundle.top_dims):
        # Stage 1: per-timepoint Ridge (ROI pixels -> z_t[d])
        s1_tr = np.zeros((n_train, 3))
        s1_te = np.zeros((1, 3))

        for t_i, tp in enumerate(TP_NAMES):
            y_d = bundle.z_per_tp[tp][:, d]
            y_tr = y_d[train_idx]
            mask = masks[d][tp]

            pixels = X_np[:, t_i].reshape(X_np.shape[0], -1)
            mask_flat = mask.ravel()
            if not mask_flat.any():
                s1_tr[:, t_i] = np.mean(y_tr)
                s1_te[:, t_i] = np.mean(y_tr)
                continue

            X_des = pixels[:, mask_flat]
            X_tr = X_des[train_idx]
            X_te = X_des[test_idx:test_idx + 1]

            a = select_alpha_inner_cv(X_tr, y_tr, alpha_grid)
            ptr, pte = fit_predict_ridge(X_tr, y_tr, X_te, a)
            s1_tr[:, t_i] = ptr
            s1_te[:, t_i] = pte[0]

        # Stage 2: aggregate (per-tp predictions -> z_agg[d])
        y_agg = bundle.z_agg[:, d]
        y_agg_tr = y_agg[train_idx]

        if s1_tr.std() < 1e-12:
            proxies_tr[:, i] = np.mean(y_agg_tr)
            proxies_te[:, i] = np.mean(y_agg_tr)
        else:
            a2 = select_alpha_inner_cv(s1_tr, y_agg_tr, alpha_grid)
            ptr, pte = fit_predict_ridge(s1_tr, y_agg_tr, s1_te, a2)
            proxies_tr[:, i] = ptr
            proxies_te[:, i] = pte[0]

    return proxies_tr, proxies_te


# =============================================================================
# Full LOOCV evaluation at a single percentile
# =============================================================================

def run_loocv_at_percentile(
    X_np: np.ndarray, y: np.ndarray, bundles: List[ModelBundle],
    needed_dims: Dict[str, List[int]], target_features: List[str],
    roi_masks: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    alpha_grid: List[float], verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Run full LOOCV proxy reconstruction + classification at a given set of ROI masks.
    Returns: (proxy_oof [N, F], results {clf_name: {metric: val}})
    """
    N = X_np.shape[0]
    F = len(target_features)

    classifiers = {
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=SEED),
        "LR": LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED),
    }

    proxy_oof = np.zeros((N, F))
    y_preds = {k: np.zeros(N, dtype=int) for k in classifiers}
    y_probs = {k: np.zeros(N) for k in classifiers}

    loo = LeaveOneOut()

    for fold, (train_idx, test_idx) in enumerate(loo.split(np.arange(N))):
        test_i = test_idx[0]

        # Build proxy matrix
        bundle_res = {}
        for b in bundles:
            masks_for_bundle = roi_masks.get(b.model_name, {})
            tr, te = build_proxy_single_fold(
                X_np, b, train_idx, test_i, alpha_grid, masks_for_bundle
            )
            bundle_res[b.model_name] = (tr, te)

        fold_X_tr = np.zeros((len(train_idx), F))
        fold_X_te = np.zeros((1, F))

        for i, fname in enumerate(target_features):
            idx = fname.rfind("_dim")
            mname = fname[:idx]
            dim = int(fname[idx + 4:])
            b_tr, b_te = bundle_res[mname]
            col_idx = needed_dims[mname].index(dim)
            fold_X_tr[:, i] = b_tr[:, col_idx]
            fold_X_te[:, i] = b_te[:, col_idx]

        proxy_oof[test_i] = fold_X_te[0]

        # Classify
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(fold_X_tr)
        X_te_s = scaler.transform(fold_X_te)

        for name, clf in classifiers.items():
            clf.fit(X_tr_s, y[train_idx])
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_te_s)[0, 1]
            else:
                prob = clf.decision_function(X_te_s)[0]
            y_probs[name][test_i] = prob
            y_preds[name][test_i] = int(prob >= 0.5)

    # Compute metrics
    results = {}
    for name in classifiers:
        acc = accuracy_score(y, y_preds[name])
        bal_acc = balanced_accuracy_score(y, y_preds[name])
        auc = roc_auc_score(y, y_probs[name])
        power = calculate_binomial_power(acc, N)
        cm = confusion_matrix(y, y_preds[name])
        tn, fp, fn, tp_ = cm.ravel()
        sens = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[name] = {
            "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc,
            "sensitivity": sens, "specificity": spec, "power": power,
        }
    return proxy_oof, results


def compute_proxy_r2(proxy_oof: np.ndarray, bundles: List[ModelBundle],
                     target_features: List[str],
                     needed_dims: Dict[str, List[int]]) -> List[Dict]:
    """Compute R² between proxy and true latent for each feature."""
    rows = []
    for i, fname in enumerate(target_features):
        idx = fname.rfind("_dim")
        mname = fname[:idx]
        dim = int(fname[idx + 4:])
        for b in bundles:
            if b.model_name == mname:
                true_z = b.z_agg[:, dim]
                break
        proxy_z = proxy_oof[:, i]
        ss_res = np.sum((true_z - proxy_z) ** 2)
        ss_tot = np.sum((true_z - true_z.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        corr = np.corrcoef(true_z, proxy_z)[0, 1] if np.std(proxy_z) > 1e-12 else 0.0
        rows.append({"feature": fname, "r2": r2, "pearson_r": corr})
    return rows


# =============================================================================
# Strategy B: Nested LOOCV with per-fold percentile selection
# =============================================================================

def run_nested_loocv(
    X_np: np.ndarray, y: np.ndarray, bundles: List[ModelBundle],
    needed_dims: Dict[str, List[int]], target_features: List[str],
    consensus_maps_all: Dict[str, Dict[int, Dict[int, np.ndarray]]],
    mse_ensemble: Dict[str, np.ndarray],
    percentiles: List[float], mse_threshold: float,
    alpha_grid: List[float],
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Fully nested LOOCV: outer fold holds out 1 sample, inner LOOCV on N-1
    samples sweeps percentiles, picks the best, then evaluates the held-out.
    """
    N = X_np.shape[0]
    F = len(target_features)

    classifiers_spec = {
        "SVM": lambda: SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
        "RF": lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=SEED),
        "LR": lambda: LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED),
    }

    proxy_oof = np.zeros((N, F))
    y_preds = {k: np.zeros(N, dtype=int) for k in classifiers_spec}
    y_probs = {k: np.zeros(N) for k in classifiers_spec}
    fold_records = []

    outer_loo = LeaveOneOut()

    for outer_fold, (outer_train, outer_test) in enumerate(outer_loo.split(np.arange(N))):
        outer_test_i = outer_test[0]
        if outer_fold % 5 == 0:
            print(f"  Nested LOOCV outer fold {outer_fold + 1}/{N}...")

        # --- Inner LOOCV: sweep percentiles on outer_train ---
        inner_best_pct = percentiles[len(percentiles) // 2]  # default fallback
        inner_best_score = -np.inf

        for pct in percentiles:
            # Extract ROIs at this percentile
            roi_masks_pct = extract_all_rois_at_percentile(
                consensus_maps_all, mse_ensemble, needed_dims, pct, mse_threshold
            )

            # Inner LOOCV on outer_train
            inner_loo = LeaveOneOut()
            inner_preds_lr = []
            inner_true = []

            for inner_train_rel, inner_test_rel in inner_loo.split(np.arange(len(outer_train))):
                inner_train_abs = outer_train[inner_train_rel]
                inner_test_abs_i = outer_train[inner_test_rel[0]]

                # Build proxy
                bundle_res = {}
                for b in bundles:
                    masks_b = roi_masks_pct.get(b.model_name, {})
                    tr, te = build_proxy_single_fold(
                        X_np, b, inner_train_abs, inner_test_abs_i, alpha_grid, masks_b
                    )
                    bundle_res[b.model_name] = (tr, te)

                inn_X_tr = np.zeros((len(inner_train_abs), F))
                inn_X_te = np.zeros((1, F))
                for i, fname in enumerate(target_features):
                    idx = fname.rfind("_dim")
                    mname = fname[:idx]
                    dim = int(fname[idx + 4:])
                    b_tr, b_te = bundle_res[mname]
                    col_idx = needed_dims[mname].index(dim)
                    inn_X_tr[:, i] = b_tr[:, col_idx]
                    inn_X_te[:, i] = b_te[:, col_idx]

                # Quick eval with LR only (speed)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(inn_X_tr)
                X_te_s = scaler.transform(inn_X_te)
                clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=SEED)
                clf.fit(X_tr_s, y[inner_train_abs])
                pred = clf.predict(X_te_s)[0]
                inner_preds_lr.append(pred)
                inner_true.append(y[inner_test_abs_i])

            inner_acc = accuracy_score(inner_true, inner_preds_lr)
            if inner_acc > inner_best_score:
                inner_best_score = inner_acc
                inner_best_pct = pct

        # --- Outer evaluation at best percentile ---
        roi_masks_best = extract_all_rois_at_percentile(
            consensus_maps_all, mse_ensemble, needed_dims, inner_best_pct, mse_threshold
        )

        bundle_res = {}
        for b in bundles:
            masks_b = roi_masks_best.get(b.model_name, {})
            tr, te = build_proxy_single_fold(
                X_np, b, outer_train, outer_test_i, alpha_grid, masks_b
            )
            bundle_res[b.model_name] = (tr, te)

        fold_X_tr = np.zeros((len(outer_train), F))
        fold_X_te = np.zeros((1, F))
        for i, fname in enumerate(target_features):
            idx = fname.rfind("_dim")
            mname = fname[:idx]
            dim = int(fname[idx + 4:])
            b_tr, b_te = bundle_res[mname]
            col_idx = needed_dims[mname].index(dim)
            fold_X_tr[:, i] = b_tr[:, col_idx]
            fold_X_te[:, i] = b_te[:, col_idx]

        proxy_oof[outer_test_i] = fold_X_te[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(fold_X_tr)
        X_te_s = scaler.transform(fold_X_te)

        for name, clf_fn in classifiers_spec.items():
            clf = clf_fn()
            clf.fit(X_tr_s, y[outer_train])
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_te_s)[0, 1]
            else:
                prob = clf.decision_function(X_te_s)[0]
            y_probs[name][outer_test_i] = prob
            y_preds[name][outer_test_i] = int(prob >= 0.5)

        fold_records.append({
            "outer_fold": outer_fold,
            "held_out_idx": outer_test_i,
            "selected_percentile": inner_best_pct,
            "inner_best_acc": inner_best_score,
        })

    # Aggregate results
    results = {}
    for name in classifiers_spec:
        acc = accuracy_score(y, y_preds[name])
        bal_acc = balanced_accuracy_score(y, y_preds[name])
        auc = roc_auc_score(y, y_probs[name])
        power = calculate_binomial_power(acc, N)
        cm = confusion_matrix(y, y_preds[name])
        tn, fp, fn, tp_ = cm.ravel()
        sens = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[name] = {
            "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc,
            "sensitivity": sens, "specificity": spec, "power": power,
        }

    fold_df = pd.DataFrame(fold_records)
    return proxy_oof, results, fold_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 3+4 Fused: ROI + Proxy with Percentile Optimization"
    )
    parser.add_argument("--tp_dirs", type=str, required=True)
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--step2_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--arch_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./step3_4_fused_results")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--mse_threshold", type=float, default=0.5)
    parser.add_argument("--percentiles", type=str, default="75,80,85,90,92,95,97",
                        help="Comma-separated percentiles to sweep")
    parser.add_argument("--opt_config", type=str, default=None,
                        help="Path to selected_features_config.json from Step OPT v2")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="CSV with columns: code, group")
    parser.add_argument("--target_features", type=str, default=None,
                        help="Comma-separated feature names (fallback)")
    parser.add_argument("--skip_nested", action="store_true",
                        help="Skip nested LOOCV (Strategy B) to save time")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    percentiles = [float(p) for p in args.percentiles.split(",")]
    alpha_grid = [0.1, 1.0, 10.0, 100.0]

    # --- Load config ---
    if args.opt_config and Path(args.opt_config).exists():
        TARGET_FEATURES, needed_dims = load_config(args.opt_config)
        import shutil
        shutil.copy2(args.opt_config, outdir / "source_opt_config.json")
    elif args.target_features:
        TARGET_FEATURES = [f.strip() for f in args.target_features.split(",")]
        needed_dims = parse_target_features(TARGET_FEATURES)
    else:
        raise ValueError("Must provide --opt_config or --target_features")

    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("STEP 3+4 FUSED: ROI + PROXY WITH PERCENTILE OPTIMIZATION")
    print("=" * 70)
    print(f"Target features:    {TARGET_FEATURES}")
    print(f"Percentile sweep:   {percentiles}")
    print(f"MSE threshold:      {args.mse_threshold}")
    print(f"Device:             {device}")
    print(f"Skip nested LOOCV:  {args.skip_nested}")

    # --- Load data ---
    print("\nLoading data...")
    codes, y, groups = load_labels(args.labels_csv)
    tp_dirs_list = tp_dirs
    found_codes = set(list_sample_codes(tp_dirs_list[0]))
    for d in tp_dirs_list[1:]:
        found_codes &= set(list_sample_codes(d))

    keep_mask = [c in found_codes for c in codes]
    codes = [c for c, k in zip(codes, keep_mask) if k]
    y = y[np.array(keep_mask)]
    groups = groups[np.array(keep_mask)]
    N = len(codes)
    n_als = int(y.sum())
    n_ctrl = N - n_als
    print(f"Samples: {N} (ALS={n_als}, CTRL={n_ctrl})")

    X_np = load_all_samples(codes, tp_dirs_list)
    _, T, H, W = X_np.shape
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
    print(f"EEM shape: [{N}, {T}, {H}, {W}]")

    # --- Load MSE maps ---
    print("\nLoading MSE maps...")
    step2_dir = Path(args.step2_dir)
    mse_ensemble = {}
    for t_name in TP_NAMES:
        mse_path = step2_dir / f"mse_ensemble_{t_name}.npy"
        if not mse_path.exists():
            raise FileNotFoundError(f"Missing: {mse_path}")
        mse_ensemble[t_name] = np.load(mse_path)

    # --- Load Z vectors ---
    print("Loading latent feature bundles...")
    bundles = load_bundles(Path(args.step1_dir), needed_dims, codes, args.latent_dim)

    # --- Load autoencoder models & compute importance maps (ONCE) ---
    print("\nComputing importance maps (gradient saliency + linear probing)...")
    arch_mod = import_arch_module(args.arch_file)
    arch_class = getattr(arch_mod, 'ConvAutoencoderWithAttention', None)
    if arch_class is None:
        import inspect
        for name, obj in inspect.getmembers(arch_mod, inspect.isclass):
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                arch_class = obj
                break
    if arch_class is None:
        raise RuntimeError("Could not find autoencoder class")

    consensus_maps_all: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    model_dir = Path(args.model_dir)

    for mname, target_dims in needed_dims.items():
        m_step1_dir = Path(args.step1_dir) / mname

        possible_paths = [
            model_dir / mname / "model.pth",
            model_dir / f"{mname}.pth",
            model_dir / f"model_fold{mname[-1]}.pth",
        ]
        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = p
                break
        if model_path is None:
            print(f"  [WARN] Skipping {mname}: no .pth found")
            continue

        print(f"  Model: {mname} ({model_path.name})")
        model = arch_class(in_channels=1, latent_dim=args.latent_dim, num_classes=2).to(device)
        _load_state_dict_flexible(model, str(model_path))
        model.eval()

        # Load Z for linear probing
        z_per_tp = {}
        for tp in TP_NAMES:
            df_tp = pd.read_csv(m_step1_dir / f"z_{tp}.csv")
            z_codes = df_tp["code"].astype(str).tolist()
            code_map = {c: i for i, c in enumerate(z_codes)}
            reindex = [code_map[c] for c in codes]
            dim_cols = [f"dim{d}" for d in range(args.latent_dim)]
            z_per_tp[tp] = df_tp[dim_cols].values[reindex]

        t0 = time.time()
        grad_maps = compute_gradient_saliency(model, X_tensor, target_dims)
        linear_maps = compute_linear_probing(X_tensor, z_per_tp, target_dims)
        consensus = compute_consensus_maps(grad_maps, linear_maps, target_dims, T=3)
        consensus_maps_all[mname] = consensus
        print(f"    Done in {time.time() - t0:.1f}s")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================================
    # STRATEGY A: Percentile Sweep (R² + Classification at each percentile)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STRATEGY A: Full Percentile Sweep")
    print("=" * 70)

    sweep_r2_rows = []
    sweep_clf_rows = []

    for pct in percentiles:
        print(f"\n--- Percentile {pct}% ---")
        t0 = time.time()

        roi_masks = extract_all_rois_at_percentile(
            consensus_maps_all, mse_ensemble, needed_dims, pct, args.mse_threshold
        )

        # Report ROI sizes
        for mname in roi_masks:
            for d in roi_masks[mname]:
                for tp in TP_NAMES:
                    n_px = roi_masks[mname][d][tp].sum()
                    print(f"  {mname} dim{d} {tp}: {n_px} px "
                          f"({100 * n_px / (H * W):.1f}%)")

        # Run LOOCV
        proxy_oof, results = run_loocv_at_percentile(
            X_np, y, bundles, needed_dims, TARGET_FEATURES, roi_masks, alpha_grid
        )

        # R² per feature
        r2_list = compute_proxy_r2(proxy_oof, bundles, TARGET_FEATURES, needed_dims)
        mean_r2 = np.mean([r["r2"] for r in r2_list])

        for r in r2_list:
            sweep_r2_rows.append({"percentile": pct, **r})

        # Classification results
        for clf_name, metrics in results.items():
            sweep_clf_rows.append({"percentile": pct, "classifier": clf_name, **metrics})

        elapsed = time.time() - t0
        best_clf = max(results.keys(), key=lambda k: results[k]["auc"])
        print(f"  Mean R²={mean_r2:.3f} | Best AUC={results[best_clf]['auc']:.3f} "
              f"({best_clf}) | {elapsed:.1f}s")

    sweep_r2_df = pd.DataFrame(sweep_r2_rows)
    sweep_clf_df = pd.DataFrame(sweep_clf_rows)
    sweep_r2_df.to_csv(outdir / "percentile_sweep_r2.csv", index=False)
    sweep_clf_df.to_csv(outdir / "percentile_sweep_classification.csv", index=False)

    # Find best percentiles
    r2_by_pct = sweep_r2_df.groupby("percentile")["r2"].mean()
    best_pct_r2 = r2_by_pct.idxmax()

    auc_by_pct = sweep_clf_df.groupby("percentile")["auc"].max()
    best_pct_auc = auc_by_pct.idxmax()

    print(f"\n  Best percentile by R²:  {best_pct_r2} (mean R²={r2_by_pct[best_pct_r2]:.3f})")
    print(f"  Best percentile by AUC: {best_pct_auc} (max AUC={auc_by_pct[best_pct_auc]:.3f})")

    # =========================================================================
    # STRATEGY B: Nested LOOCV (optional)
    # =========================================================================
    nested_results = None
    nested_fold_df = None
    nested_proxy_oof = None

    if not args.skip_nested:
        print("\n" + "=" * 70)
        print("STRATEGY B: Nested LOOCV (per-fold percentile selection)")
        print("=" * 70)
        print(f"  Outer: {N} LOOCV folds")
        print(f"  Inner: {N - 1} LOOCV folds × {len(percentiles)} percentiles")
        print(f"  Estimated time: ~{N * (N-1) * len(percentiles) * 0.03:.0f}s")

        t0 = time.time()
        nested_proxy_oof, nested_results, nested_fold_df = run_nested_loocv(
            X_np, y, bundles, needed_dims, TARGET_FEATURES,
            consensus_maps_all, mse_ensemble,
            percentiles, args.mse_threshold, alpha_grid,
        )
        elapsed = time.time() - t0
        print(f"\n  Nested LOOCV completed in {elapsed:.1f}s")

        nested_fold_df.to_csv(outdir / "nested_loocv_fold_details.csv", index=False)

        # Percentile selection distribution
        pct_counts = nested_fold_df["selected_percentile"].value_counts().sort_index()
        print(f"\n  Percentile selection distribution:")
        for pct_val, cnt in pct_counts.items():
            print(f"    {pct_val}%: chosen in {cnt}/{N} folds ({100 * cnt / N:.0f}%)")

        for clf_name, metrics in nested_results.items():
            print(f"\n  {clf_name} (nested): Acc={metrics['accuracy']:.3f} "
                  f"BalAcc={metrics['balanced_accuracy']:.3f} "
                  f"AUC={metrics['auc']:.3f} Power={metrics['power']:.3f}")

    # =========================================================================
    # Save final results at best percentile (Strategy A)
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL OUTPUTS")
    print("=" * 70)

    # Re-run at best R²-selected percentile to save proxy matrix
    best_pct = best_pct_r2
    print(f"\nSaving final proxy at best-R² percentile = {best_pct}%")

    roi_masks_final = extract_all_rois_at_percentile(
        consensus_maps_all, mse_ensemble, needed_dims, best_pct, args.mse_threshold
    )
    proxy_oof_final, results_final = run_loocv_at_percentile(
        X_np, y, bundles, needed_dims, TARGET_FEATURES, roi_masks_final, alpha_grid
    )
    r2_final = compute_proxy_r2(proxy_oof_final, bundles, TARGET_FEATURES, needed_dims)

    # Save proxy matrix
    proxy_df = pd.DataFrame(proxy_oof_final, columns=TARGET_FEATURES)
    proxy_df.insert(0, "code", codes)
    proxy_df.insert(1, "group", groups)
    proxy_df.to_csv(outdir / "final_proxy_features.csv", index=False)

    # Save ROI masks at best percentile
    masks_dict = {}
    for mname in roi_masks_final:
        m_outdir = outdir / "roi_masks" / mname
        m_outdir.mkdir(parents=True, exist_ok=True)
        for d in roi_masks_final[mname]:
            for tp in TP_NAMES:
                mask = roi_masks_final[mname][d][tp]
                np.save(m_outdir / f"roi_mask_dim{d}_{tp}.npy", mask)
                masks_dict[f"{mname}_dim{d}_{tp}"] = mask
    np.savez_compressed(outdir / "roi_all_masks.npz", **masks_dict)

    # Save consolidated results
    final_rows = []

    # Strategy A results (best by R²)
    for clf_name, metrics in results_final.items():
        final_rows.append({
            "strategy": "A_best_r2",
            "percentile": best_pct,
            "classifier": clf_name,
            **metrics,
        })

    # Strategy A results (best by AUC)
    if best_pct_auc != best_pct:
        roi_masks_auc = extract_all_rois_at_percentile(
            consensus_maps_all, mse_ensemble, needed_dims, best_pct_auc, args.mse_threshold
        )
        _, results_auc = run_loocv_at_percentile(
            X_np, y, bundles, needed_dims, TARGET_FEATURES, roi_masks_auc, alpha_grid
        )
        for clf_name, metrics in results_auc.items():
            final_rows.append({
                "strategy": "A_best_auc",
                "percentile": best_pct_auc,
                "classifier": clf_name,
                **metrics,
            })

    # Strategy B results
    if nested_results is not None:
        for clf_name, metrics in nested_results.items():
            final_rows.append({
                "strategy": "B_nested_loocv",
                "percentile": "per_fold",
                "classifier": clf_name,
                **metrics,
            })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(outdir / "final_classification_metrics.csv", index=False)

    # R² at best percentile
    pd.DataFrame(r2_final).to_csv(outdir / "final_proxy_reconstruction_quality.csv", index=False)

    # ROI summary
    roi_summary = []
    for mname in roi_masks_final:
        for d in roi_masks_final[mname]:
            for tp in TP_NAMES:
                n_px = int(roi_masks_final[mname][d][tp].sum())
                roi_summary.append({
                    "model": mname, "dim": d, "timepoint": tp,
                    "percentile": best_pct,
                    "roi_pixels": n_px, "total_pixels": H * W,
                    "roi_fraction": n_px / (H * W),
                })
    pd.DataFrame(roi_summary).to_csv(outdir / "roi_summary.csv", index=False)

    # Save config
    config_out = {
        "pipeline_version": "step3_4_fused_v1",
        "target_features": TARGET_FEATURES,
        "model_dims": {m: [int(d) for d in dims] for m, dims in needed_dims.items()},
        "percentiles_swept": percentiles,
        "mse_threshold": args.mse_threshold,
        "best_percentile_by_r2": float(best_pct_r2),
        "best_percentile_by_auc": float(best_pct_auc),
        "n_samples": N,
        "n_als": n_als,
        "n_ctrl": n_ctrl,
    }
    with open(outdir / "pipeline_config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nStrategy A — Percentile Sweep:")
    print(f"  Best by R²:  {best_pct_r2}% (mean R²={r2_by_pct[best_pct_r2]:.3f})")
    print(f"  Best by AUC: {best_pct_auc}% (max AUC={auc_by_pct[best_pct_auc]:.3f})")

    print(f"\nFinal metrics at {best_pct}% (R²-selected):")
    for clf_name, metrics in results_final.items():
        print(f"  {clf_name}: Acc={metrics['accuracy']:.3f} "
              f"BalAcc={metrics['balanced_accuracy']:.3f} "
              f"AUC={metrics['auc']:.3f} "
              f"Sens={metrics['sensitivity']:.3f} "
              f"Spec={metrics['specificity']:.3f} "
              f"Power={metrics['power']:.3f}")

    if nested_results:
        print(f"\nStrategy B — Nested LOOCV:")
        for clf_name, metrics in nested_results.items():
            print(f"  {clf_name}: Acc={metrics['accuracy']:.3f} "
                  f"BalAcc={metrics['balanced_accuracy']:.3f} "
                  f"AUC={metrics['auc']:.3f} "
                  f"Power={metrics['power']:.3f}")

    print(f"\nOutputs saved to {outdir}")


if __name__ == "__main__":
    main()
