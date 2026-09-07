#!/usr/bin/env python3
"""
STEP 2 v3: Encoder-Weight-Based Proxy Latent Pipeline
=======================================================

FIXES OVER v2:
  PCA on 39 samples with 131K features produces 38 components that span the
  entire sample space (var_explained=1.0). No actual reduction occurs. The
  proxy R² was no better than noise.

NEW APPROACH — THREE COMPLEMENTARY STRATEGIES:

  STRATEGY A: "FC-projection"  (the encoder already knows the answer)
    The FC weight vector W_d for dim d IS the exact linear mapping from conv
    features to z_d. We don't need Ridge to rediscover it. Instead:
      1. Project conv features onto W_d: score_t = conv_feats[:,t,:] @ W_d + b_d
      2. This gives [N, 3] scores (one per timepoint)
      3. Learn optimal TP combination via attention weights or simple Ridge on
         these 3 scores -> z_target_d
    This is 39 samples × 3 features. Clean, interpretable, no overfitting.

  STRATEGY B: "FC-neighborhood projection"
    Expand beyond the single W_d direction by also using nearby FC rows
    (top-K correlated dims). This captures related latent structure:
      1. Find top K dims whose FC weight vectors correlate most with W_d
      2. Project conv features onto each -> [N, 3, K+1] scores
      3. Ridge on these (3*(K+1)) features -> z_target_d
    Moderate dimensionality (e.g., 39 × 30 with K=9).

  STRATEGY C: "Raw pixel baseline"
    For comparison: Ridge from raw EEM pixels (flattened) -> z_target_d.
    Uses the same PCA+Ridge LOOCV, but on the original images rather than
    conv features. This tells us whether the conv layers add value.

  STRATEGY D: "PLS regression" (supervised dim reduction)
    Partial Least Squares finds components that maximize covariance between
    X and z_target — unlike PCA which maximizes variance in X alone.
    PLS components are by construction relevant to the target.

All strategies use fold-specific conv features and fully leak-free LOOCV.
Permutation tests validate each strategy.

USAGE:
  python step2_encoder_weight_proxy_v3.py \
    --arch_file conv_autoencoder_detailed.py \
    --model_dir ./5_fold_models_original \
    --tp_dirs ./out_0h,./out_6h,./out_24h \
    --labels_csv sample_labels.csv \
    --step1_dir ./step1_latent_features \
    --opt_config ./opt_selection_results_v3/selected_features_config.json \
    --output_dir ./step2_encoder_proxy_v3 \
    --n_pls 10 --fc_neighborhood_k 9 --n_permutations 200
"""

import os
import sys
import argparse
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, r2_score, mean_squared_error
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TP_NAMES = ["0h", "6h", "24h"]


# =============================================================================
# Architecture import
# =============================================================================
def import_arch_module(arch_file: str):
    spec = importlib.util.spec_from_file_location("arch_module", arch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import architecture file: {arch_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Data loading
# =============================================================================
def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    codes = df["code"].astype(str).tolist()
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
    exact = tp_path / f"{code}.xlsx"
    if exact.exists():
        return str(exact)
    for f in tp_path.glob("*.xlsx"):
        if f.name.startswith("~$"):
            continue
        stem = f.stem
        for suffix in ["_0h", "_6h", "_24h"]:
            if stem.endswith(suffix):
                file_code = stem[:-len(suffix)]
                if file_code == code or file_code.replace('.', '_') == code:
                    return str(f)
    return None

def load_excel_as_array(filepath: str) -> np.ndarray:
    from openpyxl import load_workbook
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
                raise FileNotFoundError(f"No file for code '{code}' in {tp_dir}")
            arr = load_excel_as_array(fpath)
            tp_images.append(arr)
        shapes = [a.shape for a in tp_images]
        if len(set(shapes)) > 1:
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            tp_images = [a[:min_h, :min_w] for a in tp_images]
        arrays.append(np.stack(tp_images, axis=0))
    X = np.stack(arrays, axis=0)
    print(f"  Loaded {X.shape[0]} samples, shape per sample: {X.shape[1:]}")
    return X


# =============================================================================
# Model loading
# =============================================================================
def _load_state_dict_flexible(model: nn.Module, path: str):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    try:
        model.load_state_dict(sd, strict=True)
        return
    except RuntimeError:
        pass
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k.replace("module.", "")] = v
    try:
        model.load_state_dict(new_sd, strict=True)
        return
    except RuntimeError:
        pass
    model_sd = model.state_dict()
    matched = OrderedDict()
    for mk, mv in model_sd.items():
        for sk, sv in sd.items():
            if sv.shape == mv.shape and mk.split(".")[-1] == sk.split(".")[-1]:
                matched[mk] = sv
                break
        else:
            matched[mk] = mv
    model.load_state_dict(matched, strict=False)

def load_model(arch_mod, model_path: str, latent_dim: int, X_dummy: torch.Tensor):
    model = arch_mod.ConvAutoencoderWithAttention(
        in_channels=1, latent_dim=latent_dim, num_classes=2
    )
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(X_dummy[:1].unsqueeze(2).to(DEVICE))
    _load_state_dict_flexible(model, model_path)
    model.eval()
    return model


# =============================================================================
# Feature extraction
# =============================================================================
def extract_conv_features(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """X: [N, 3, H, W] -> [N, 3, flatten_size]"""
    model.eval()
    N, T, H, W = X.shape
    X_5d = X.unsqueeze(2).to(DEVICE)
    feats = []
    with torch.no_grad():
        for t in range(T):
            x = X_5d[:, t]
            x = model.encoder.conv1(x)
            x = model.encoder.conv2(x)
            x = model.encoder.conv3(x)
            x = model.encoder.conv4(x)
            feats.append(x.view(x.size(0), -1).cpu().numpy())
    return np.stack(feats, axis=1)

def extract_fc_weights(model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    W = model.encoder.fc.weight.data.cpu().numpy()
    b = model.encoder.fc.bias.data.cpu().numpy()
    return W, b

def extract_attention_weights(model: nn.Module) -> Dict[str, np.ndarray]:
    return {
        "attn_w1": model.attention.attention_net[0].weight.data.cpu().numpy(),
        "attn_b1": model.attention.attention_net[0].bias.data.cpu().numpy(),
        "attn_w2": model.attention.attention_net[2].weight.data.cpu().numpy(),
        "attn_b2": model.attention.attention_net[2].bias.data.cpu().numpy(),
    }

def compute_z_agg_analytical(conv_feats, W, b, attn_params):
    N, T, F = conv_feats.shape
    z_per_tp = np.zeros((N, T, W.shape[0]))
    for t in range(T):
        z_per_tp[:, t, :] = conv_feats[:, t, :] @ W.T + b
    w1, b1 = attn_params["attn_w1"], attn_params["attn_b1"]
    w2, b2 = attn_params["attn_w2"], attn_params["attn_b2"]
    scores = np.zeros((N, T))
    for t in range(T):
        h = np.tanh(z_per_tp[:, t, :] @ w1.T + b1)
        scores[:, t] = (h @ w2.T + b2).squeeze(-1)
    scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    attn = scores_exp / scores_exp.sum(axis=1, keepdims=True)
    z_agg = sum(attn[:, t:t+1] * z_per_tp[:, t, :] for t in range(T))
    return z_agg, attn


# =============================================================================
# Helpers
# =============================================================================
def _oof_stats(proxy_oof, z_target):
    """Compute OOF fit statistics."""
    ok = np.isfinite(proxy_oof) & np.isfinite(z_target)
    if ok.sum() < 3 or np.std(proxy_oof[ok]) < 1e-12 or np.std(z_target[ok]) < 1e-12:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(z_target[ok], proxy_oof[ok])[0, 1])
    r2 = float(r2_score(z_target[ok], proxy_oof[ok]))
    mse = float(mean_squared_error(z_target[ok], proxy_oof[ok]))
    return r, r2, mse


# =============================================================================
# STRATEGY A: FC-projection (exact encoder direction)
# =============================================================================
def strategy_fc_projection(
    conv_feats: np.ndarray,   # [N, 3, F] fold-specific
    W_d: np.ndarray,          # [F,] FC weight row for dim d
    b_d: float,               # FC bias for dim d
    z_target: np.ndarray,     # [N,]
    alpha_grid: List[float],
    attn_weights: Optional[np.ndarray] = None,  # [N, 3] from model
) -> Tuple[np.ndarray, Dict]:
    """
    Project conv features onto the KNOWN FC direction for this dim.
    This gives exactly 3 features (one per timepoint).
    Then Ridge learns the optimal TP combination.

    Returns: (proxy_oof [N,], info_dict)
    """
    N = conv_feats.shape[0]

    # Step 1: Project onto FC direction -> [N, 3]
    scores = np.zeros((N, 3))
    for t in range(3):
        scores[:, t] = conv_feats[:, t, :] @ W_d + b_d

    # Step 2: Ridge LOOCV on these 3 features -> z_target
    loo = LeaveOneOut()
    proxy_oof = np.full(N, np.nan)
    for train_idx, test_idx in loo.split(scores):
        X_tr, X_te = scores[train_idx], scores[test_idx]
        z_tr = z_target[train_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        reg = RidgeCV(alphas=alpha_grid, fit_intercept=True)
        reg.fit((X_tr - mu) / sd, z_tr)
        proxy_oof[test_idx] = reg.predict((X_te - mu) / sd)

    r, r2, mse = _oof_stats(proxy_oof, z_target)

    # Per-TP correlation with z_target
    tp_corr = {}
    for t in range(3):
        if np.std(scores[:, t]) > 1e-12:
            tp_corr[TP_NAMES[t]] = float(np.corrcoef(z_target, scores[:, t])[0, 1])
        else:
            tp_corr[TP_NAMES[t]] = np.nan

    # Also compute attention-weighted proxy (no Ridge, pure model weights)
    if attn_weights is not None:
        z_attn = np.sum(scores * attn_weights, axis=1)
        r_attn = float(np.corrcoef(z_target, z_attn)[0, 1]) if np.std(z_attn) > 1e-12 else np.nan
    else:
        r_attn = np.nan

    return proxy_oof, {
        "r": r, "r2": r2, "mse": mse,
        "n_features": 3,
        "tp_corr": tp_corr,
        "r_attn_weighted": r_attn,
    }


# =============================================================================
# STRATEGY B: FC-neighborhood projection
# =============================================================================
def strategy_fc_neighborhood(
    conv_feats: np.ndarray,   # [N, 3, F]
    W_full: np.ndarray,       # [latent_dim, F] full FC weight matrix
    b_full: np.ndarray,       # [latent_dim]
    dim_d: int,
    z_target: np.ndarray,     # [N,]
    alpha_grid: List[float],
    K: int = 9,
) -> Tuple[np.ndarray, Dict]:
    """
    Project onto dim d's FC direction + K nearest neighbor directions.
    This gives 3*(K+1) features.

    Returns: (proxy_oof [N,], info_dict)
    """
    N, T, F = conv_feats.shape
    latent_dim = W_full.shape[0]

    # Find top-K dims whose FC weight vectors are most similar to W_d
    W_d = W_full[dim_d]
    W_d_norm = W_d / (np.linalg.norm(W_d) + 1e-12)

    # Cosine similarity of all FC rows with W_d
    norms = np.linalg.norm(W_full, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    W_normed = W_full / norms
    cosine_sim = W_normed @ W_d_norm  # [latent_dim,]
    cosine_sim[dim_d] = -np.inf  # exclude self
    neighbor_dims = np.argsort(-np.abs(cosine_sim))[:K]

    all_dims = [dim_d] + list(neighbor_dims)
    n_proj = len(all_dims)

    # Project conv features onto these dims -> [N, 3, n_proj]
    scores = np.zeros((N, T, n_proj))
    for j, dd in enumerate(all_dims):
        for t in range(T):
            scores[:, t, j] = conv_feats[:, t, :] @ W_full[dd] + b_full[dd]

    # Flatten: [N, 3*n_proj]
    X_design = scores.reshape(N, -1)

    # Ridge LOOCV
    loo = LeaveOneOut()
    proxy_oof = np.full(N, np.nan)
    for train_idx, test_idx in loo.split(X_design):
        X_tr, X_te = X_design[train_idx], X_design[test_idx]
        z_tr = z_target[train_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        reg = RidgeCV(alphas=alpha_grid, fit_intercept=True)
        reg.fit((X_tr - mu) / sd, z_tr)
        proxy_oof[test_idx] = reg.predict((X_te - mu) / sd)

    r, r2, mse = _oof_stats(proxy_oof, z_target)

    return proxy_oof, {
        "r": r, "r2": r2, "mse": mse,
        "n_features": X_design.shape[1],
        "neighbor_dims": [int(x) for x in neighbor_dims],
    }


# =============================================================================
# STRATEGY C: Raw pixel baseline
# =============================================================================
def strategy_raw_pixel_baseline(
    X_raw: np.ndarray,        # [N, 3, H, W] raw images
    z_target: np.ndarray,     # [N,]
    alpha_grid: List[float],
    n_pls: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """
    PLS from raw pixels -> z_target. Fair baseline to see if conv adds value.
    Uses PLS (supervised) since PCA failed for the same N>>p reasons.
    """
    N, T, H, W = X_raw.shape
    # Flatten per TP and stack: [N, 3*H*W]
    X_flat = X_raw.reshape(N, -1)
    n_comp = min(n_pls, N - 2)

    loo = LeaveOneOut()
    proxy_oof = np.full(N, np.nan)
    for train_idx, test_idx in loo.split(X_flat):
        X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
        z_tr = z_target[train_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        X_tr_s = (X_tr - mu) / sd
        X_te_s = (X_te - mu) / sd

        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X_tr_s, z_tr)
        proxy_oof[test_idx] = pls.predict(X_te_s).ravel()

    r, r2, mse = _oof_stats(proxy_oof, z_target)
    return proxy_oof, {"r": r, "r2": r2, "mse": mse, "n_pls": n_comp,
                        "n_pixels": X_flat.shape[1]}


# =============================================================================
# STRATEGY D: PLS on conv features (supervised dim reduction)
# =============================================================================
def strategy_pls_conv(
    conv_feats: np.ndarray,   # [N, 3, F]
    z_target: np.ndarray,     # [N,]
    n_pls: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """
    PLS regression from conv features -> z_target.
    PLS finds directions that maximize covariance(X, z) unlike PCA.
    """
    N, T, F = conv_feats.shape
    X_flat = conv_feats.reshape(N, -1)  # [N, 3*F]
    n_comp = min(n_pls, N - 2)

    loo = LeaveOneOut()
    proxy_oof = np.full(N, np.nan)
    for train_idx, test_idx in loo.split(X_flat):
        X_tr, X_te = X_flat[train_idx], X_flat[test_idx]
        z_tr = z_target[train_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        X_tr_s = (X_tr - mu) / sd
        X_te_s = (X_te - mu) / sd

        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X_tr_s, z_tr)
        proxy_oof[test_idx] = pls.predict(X_te_s).ravel()

    r, r2, mse = _oof_stats(proxy_oof, z_target)
    return proxy_oof, {"r": r, "r2": r2, "mse": mse, "n_pls": n_comp}


# =============================================================================
# STRATEGY E: Full analytical z_agg (FC + attention, zero fitting)
# =============================================================================
def strategy_analytical_z(
    conv_feats: np.ndarray,        # [N, 3, F] fold-specific
    W_full: np.ndarray,            # [latent_dim, F]
    b_full: np.ndarray,            # [latent_dim]
    attn_params: Dict[str, np.ndarray],
    dim_d: int,
    z_target: np.ndarray,          # [N,]
) -> Tuple[np.ndarray, Dict]:
    """
    Pure encoder reconstruction: FC + attention, zero fitting parameters.

    This IS what the model computes for dim d:
      1. z_t = conv_feats[:,t,:] @ W.T + b   (all 512 dims, needed for attention)
      2. attention scores from z_per_tp using attention network
      3. z_agg = attention-weighted sum of z_per_tp
      4. Extract dim d from z_agg

    To prevent data leakage, sign-alignment and scaling to z_target 
    are computed strictly within a LOOCV loop.

    Returns: (proxy [N,], info_dict)
    """
    from sklearn.model_selection import LeaveOneOut
    
    # 1. Full analytical z_agg for this fold (Deterministic, no leakage here)
    z_agg, attn_weights = compute_z_agg_analytical(conv_feats, W_full, b_full, attn_params)

    # Extract dim d
    z_fold_d = z_agg[:, dim_d]  # [N,]

    N_samples = len(z_target)
    z_proxy = np.zeros(N_samples)
    
    loo = LeaveOneOut()
    sign_flips = [] # Track folds to report if it consistently flipped

    # 2. Strict LOOCV Loop for Scaling and Alignment
    for train_idx, test_idx in loo.split(z_fold_d):
        z_train_feat = z_fold_d[train_idx]
        z_test_feat = z_fold_d[test_idx]
        z_train_targ = z_target[train_idx]
        
        # Sign alignment strictly on N-1 training samples
        if np.std(z_train_feat) > 1e-12 and np.std(z_train_targ) > 1e-12:
            r_train = float(np.corrcoef(z_train_targ, z_train_feat)[0, 1])
            flip_mult = -1.0 if r_train < 0 else 1.0
        else:
            flip_mult = 1.0
            
        sign_flips.append(flip_mult == -1.0)
        
        z_train_feat_aligned = z_train_feat * flip_mult
        z_test_feat_aligned = z_test_feat * flip_mult
        
        # Scaling (Affine transform) strictly on N-1 training samples
        feat_mean = z_train_feat_aligned.mean()
        feat_std = z_train_feat_aligned.std()
        targ_mean = z_train_targ.mean()
        targ_std = z_train_targ.std()
        
        if feat_std > 1e-12:
            z_proxy[test_idx] = (z_test_feat_aligned - feat_mean) / feat_std * targ_std + targ_mean
        else:
            z_proxy[test_idx] = z_test_feat_aligned

    # 3. Calculate Honest Out-Of-Fold Stats
    r, r2, mse = _oof_stats(z_proxy, z_target)

    # Global raw correlation (strictly for reporting diagnostics, NOT used in predictions)
    if np.std(z_fold_d) > 1e-12 and np.std(z_target) > 1e-12:
        r_raw = float(np.corrcoef(z_target, z_fold_d)[0, 1])
    else:
        r_raw = np.nan

    # Also compute per-TP contributions for diagnostics (deterministic, no fit)
    N_samples, T, F = conv_feats.shape
    z_per_tp_d = np.zeros((N_samples, T))
    for t in range(T):
        z_per_tp_d[:, t] = conv_feats[:, t, :] @ W_full[dim_d] + b_full[dim_d]
        
    tp_corr = {}
    for t in range(T):
        if np.std(z_per_tp_d[:, t]) > 1e-12:
            tp_corr[TP_NAMES[t]] = float(np.corrcoef(z_target, z_per_tp_d[:, t])[0, 1])
        else:
            tp_corr[TP_NAMES[t]] = np.nan

    # Attention weight statistics
    attn_mean = attn_weights.mean(axis=0)  # [3,]
    consistently_flipped = all(sign_flips) if sign_flips else False

    return z_proxy, {
        "r": r, "r2": r2, "mse": mse,
        "r_raw_before_sign": r_raw,
        "sign_flipped": consistently_flipped,
        "n_features": 0,  # zero free parameters
        "tp_corr": tp_corr,
        "attn_mean_weights": {TP_NAMES[t]: float(attn_mean[t]) for t in range(T)},
    }

# =============================================================================
# Permutation test (generic)
# =============================================================================
def permutation_test_proxy(
    run_fn,           # callable(z_target) -> (proxy_oof, info)
    z_target: np.ndarray,
    n_permutations: int,
    label: str = "",
) -> Tuple[float, float, np.ndarray]:
    """Generic permutation test for any proxy strategy."""
    rng = np.random.RandomState(SEED)
    _, info_obs = run_fn(z_target)
    obs_r2 = info_obs["r2"]

    null_r2 = np.full(n_permutations, np.nan)
    desc = f"    Perm {label}" if label else "    Perm test"
    for i in tqdm(range(n_permutations), desc=desc, leave=False):
        z_perm = rng.permutation(z_target)
        _, info_perm = run_fn(z_perm)
        null_r2[i] = info_perm["r2"]

    valid = null_r2[np.isfinite(null_r2)]
    if len(valid) > 0 and np.isfinite(obs_r2):
        p_val = (np.sum(valid >= obs_r2) + 1) / (len(valid) + 1)
    else:
        p_val = np.nan
    return obs_r2, p_val, null_r2


# =============================================================================
# Classification
# =============================================================================
def run_classification_loocv(X: np.ndarray, y: np.ndarray) -> Tuple[List[Dict], Dict, Dict]:
    classifiers = {
        "SVM": SkPipeline([("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, probability=True,
                        class_weight="balanced", random_state=SEED))]),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED,
                                      class_weight="balanced_subsample"),
        "LR": SkPipeline([("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced",
                                       solver="liblinear", random_state=SEED))]),
    }
    loo = LeaveOneOut()
    N = len(y)
    preds = {k: np.zeros(N, dtype=int) for k in classifiers}
    scores = {k: np.zeros(N) for k in classifiers}
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        for name, clf in classifiers.items():
            clf.fit(X_tr, y_tr)
            preds[name][test_idx] = clf.predict(X_te)
            try:
                if hasattr(clf, "predict_proba"):
                    scores[name][test_idx] = clf.predict_proba(X_te)[0, 1]
                else:
                    scores[name][test_idx] = clf.named_steps["clf"].predict_proba(X_te)[0, 1]
            except Exception:
                scores[name][test_idx] = 0.5
    results = []
    for name in classifiers:
        acc = accuracy_score(y, preds[name])
        try:
            auc = roc_auc_score(y, scores[name])
        except Exception:
            auc = np.nan
        results.append({"model": name, "accuracy": acc, "auc": auc,
                         "n_correct": int((preds[name] == y).sum()), "n_total": N})
    return results, preds, scores


# =============================================================================
# Parse config
# =============================================================================
def parse_selected_features(config: dict, fold_paths: List[Path]) -> List[Dict]:
    fold_name_to_idx = {fp.stem: i for i, fp in enumerate(fold_paths)}
    features = []
    for feat_name in config.get("target_features", []):
        idx = feat_name.rfind("_dim")
        if idx == -1:
            continue
        fold_name = feat_name[:idx]
        dim = int(feat_name[idx + 4:])
        fold_idx = fold_name_to_idx.get(fold_name)
        if fold_idx is None:
            for fn, fi in fold_name_to_idx.items():
                if fn.endswith(fold_name.split("fold")[-1]):
                    fold_idx = fi
                    break
        if fold_idx is None:
            print(f"  [WARN] Cannot map '{fold_name}' -> skipping {feat_name}")
            continue
        features.append({"feature": feat_name, "fold_idx": fold_idx,
                         "fold_name": fold_name, "dim": dim})
    return features

def parse_target_dims_manual(dims_str: str) -> List[Dict]:
    return [{"feature": f"fold1_dim{int(d.strip())}", "fold_idx": 0,
             "fold_name": "fold1", "dim": int(d.strip())}
            for d in dims_str.split(",")]

def ensemble_sign_align(all_z_agg: List[np.ndarray]) -> np.ndarray:
    ref = all_z_agg[0]
    aligned = [ref.copy()]
    for z in all_z_agg[1:]:
        z2 = z.copy()
        for d in range(z.shape[1]):
            if np.std(ref[:, d]) > 1e-12 and np.std(z2[:, d]) > 1e-12:
                if np.corrcoef(ref[:, d], z2[:, d])[0, 1] < 0:
                    z2[:, d] *= -1
        aligned.append(z2)
    return np.mean(np.stack(aligned), axis=0)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 2 v3: FC-projection + neighborhood + PLS proxy pipeline"
    )
    parser.add_argument("--arch_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tp_dirs", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--step1_dir", type=str, default=None)
    parser.add_argument("--opt_config", type=str, default=None)
    parser.add_argument("--target_dims", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./step2_encoder_proxy_v3")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--fc_neighborhood_k", type=int, default=9,
                        help="Number of neighbor FC directions for strategy B")
    parser.add_argument("--n_pls", type=int, default=10,
                        help="PLS components for strategies C and D")
    parser.add_argument("--alpha_grid", type=str, default="0.01,0.1,1,10,100,1000,10000")
    parser.add_argument("--n_permutations", type=int, default=200,
                        help="Permutations per strategy per dim (0 to skip)")
    parser.add_argument("--force_strategy", type=str, default=None,
                        choices=["A", "B", "C", "D", "E"],
                        help="Force a specific strategy for all features instead of auto-selecting. "
                             "E=analytical z_agg (recommended: zero fitting, pure encoder output)")
    parser.add_argument("--export_inference", action="store_true", default=False,
                        help="Export a standalone inference bundle (.pt) for new-sample prediction. "
                             "Includes model weights, normalization params, and trained classifiers.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]
    alpha_grid = [float(a) for a in args.alpha_grid.split(",")]

    print("=" * 70)
    print("STEP 2 v3: FC-Projection + Neighborhood + PLS Proxy Pipeline")
    print("=" * 70)
    print(f"FC neighborhood K: {args.fc_neighborhood_k}")
    print(f"PLS components: {args.n_pls}")
    print(f"Permutations: {args.n_permutations}")
    if args.force_strategy:
        print(f"FORCED STRATEGY: {args.force_strategy}")

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("\nLoading data...")
    codes, y, groups = load_labels(args.labels_csv)
    common = set(list_sample_codes(tp_dirs[0])) & set(list_sample_codes(tp_dirs[1])) & set(list_sample_codes(tp_dirs[2]))
    keep = [c in common for c in codes]
    codes = [c for c, k in zip(codes, keep) if k]
    y = y[np.array(keep)]
    groups = groups[np.array(keep)]
    X_np = load_all_samples(codes, tp_dirs)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    N = X_np.shape[0]
    print(f"  Samples: {N} (ALS={int(y.sum())}, CTRL={N - int(y.sum())})")

    # =========================================================================
    # 2. Load models
    # =========================================================================
    print("\nLoading models...")
    arch_mod = import_arch_module(args.arch_file)
    model_dir = Path(args.model_dir)
    fold_paths = sorted(model_dir.glob("best_model_fold*.pth"))
    if not fold_paths:
        fold_paths = sorted(model_dir.glob("best_model_fold_*.pth"))
    M = len(fold_paths)
    print(f"  Found {M} fold models")

    all_conv_feats = []
    all_fc_W = []
    all_fc_b = []
    all_attn_params = []
    all_z_agg = []

    for fold_idx, fp in enumerate(fold_paths):
        print(f"  Fold {fold_idx + 1}: {fp.name}")
        model = load_model(arch_mod, str(fp), args.latent_dim, X_tensor)
        conv_feats = extract_conv_features(model, X_tensor)
        all_conv_feats.append(conv_feats)
        W, b = extract_fc_weights(model)
        all_fc_W.append(W)
        all_fc_b.append(b)
        attn_params = extract_attention_weights(model)
        all_attn_params.append(attn_params)
        z_agg, _ = compute_z_agg_analytical(conv_feats, W, b, attn_params)
        all_z_agg.append(z_agg)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # 3. Z_target
    # =========================================================================
    print("\nBuilding Z_target...")
    Z_target = ensemble_sign_align(all_z_agg)
    if args.step1_dir:
        z_path = Path(args.step1_dir) / "ensemble_z_agg.csv"
        if z_path.exists():
            df_z = pd.read_csv(z_path)
            dim_cols = [c for c in df_z.columns if c.startswith("dim")]
            Z_target = df_z[dim_cols].values
            print(f"  Loaded from step1 ({Z_target.shape})")

    # =========================================================================
    # 4. Parse selected features
    # =========================================================================
    if args.opt_config:
        with open(args.opt_config) as f:
            opt_config = json.load(f)
        selected_features = parse_selected_features(opt_config, fold_paths)
    elif args.target_dims:
        selected_features = parse_target_dims_manual(args.target_dims)
    else:
        raise ValueError("Must provide --opt_config or --target_dims")

    print(f"\nSelected features ({len(selected_features)}):")
    for sf in selected_features:
        print(f"  {sf['feature']}  ->  fold {sf['fold_idx'] + 1}, dim {sf['dim']}")

    # =========================================================================
    # 5. Run all strategies for each feature
    # =========================================================================
    print("\n" + "=" * 70)
    print("PROXY STRATEGIES")
    print("=" * 70)

    all_results = []
    # Store best proxy per feature for final classification
    best_proxy_per_feature = {}

    for sf in selected_features:
        fi = sf["fold_idx"]
        d = sf["dim"]
        z_d = Z_target[:, d]
        conv_this = all_conv_feats[fi]
        W_full = all_fc_W[fi]
        b_full = all_fc_b[fi]
        W_d = W_full[d]
        b_d = b_full[d]

        # Compute fold-specific attention weights for this feature
        _, attn_w = compute_z_agg_analytical(conv_this, W_full, b_full, all_attn_params[fi])

        print(f"\n{'─' * 70}")
        print(f"  {sf['feature']} (fold {fi+1}, dim {d}, z_std={np.std(z_d):.4f})")
        print(f"{'─' * 70}")

        entry = {"feature": sf["feature"], "fold": fi + 1, "dim": d,
                 "z_std": float(np.std(z_d))}

        # --- Strategy A: FC-projection (3 features) ---
        proxy_a, info_a = strategy_fc_projection(
            conv_this, W_d, b_d, z_d, alpha_grid, attn_w
        )
        print(f"  [A] FC-projection (3 feats):  r={info_a['r']:.4f}, R²={info_a['r2']:.4f}")
        print(f"      Per-TP corr: {info_a['tp_corr']}")
        print(f"      Attn-weighted (no Ridge): r={info_a['r_attn_weighted']:.4f}")
        entry.update({f"A_{k}": v for k, v in info_a.items()
                       if k not in ("tp_corr",)})
        for tp, r_tp in info_a["tp_corr"].items():
            entry[f"A_tp_{tp}"] = r_tp

        # --- Strategy B: FC-neighborhood (3*(K+1) features) ---
        proxy_b, info_b = strategy_fc_neighborhood(
            conv_this, W_full, b_full, d, z_d, alpha_grid, args.fc_neighborhood_k
        )
        print(f"  [B] FC-neighborhood ({info_b['n_features']} feats): r={info_b['r']:.4f}, R²={info_b['r2']:.4f}")
        print(f"      Neighbor dims: {info_b['neighbor_dims']}")
        entry.update({f"B_{k}": v for k, v in info_b.items()
                       if k != "neighbor_dims"})
        entry["B_neighbor_dims"] = str(info_b["neighbor_dims"])

        # --- Strategy C: Raw pixel PLS baseline ---
        proxy_c, info_c = strategy_raw_pixel_baseline(
            X_np, z_d, alpha_grid, args.n_pls
        )
        print(f"  [C] Raw pixel PLS ({info_c['n_pls']} comp): r={info_c['r']:.4f}, R²={info_c['r2']:.4f}")
        entry.update({f"C_{k}": v for k, v in info_c.items()})

        # --- Strategy D: PLS on conv features ---
        proxy_d, info_d = strategy_pls_conv(
            conv_this, z_d, args.n_pls
        )
        print(f"  [D] Conv PLS ({info_d['n_pls']} comp):      r={info_d['r']:.4f}, R²={info_d['r2']:.4f}")
        entry.update({f"D_{k}": v for k, v in info_d.items()})

        # --- Strategy E: Full analytical z_agg (zero fitting) ---
        proxy_e, info_e = strategy_analytical_z(
            conv_this, W_full, b_full, all_attn_params[fi], d, z_d
        )
        print(f"  [E] Analytical z_agg (0 params): r={info_e['r']:.4f}, R²={info_e['r2']:.4f}")
        print(f"      Raw r (before sign-align): {info_e['r_raw_before_sign']:.4f}, flipped: {info_e['sign_flipped']}")
        print(f"      Attn weights: {info_e['attn_mean_weights']}")
        print(f"      Per-TP corr: {info_e['tp_corr']}")
        entry.update({f"E_{k}": v for k, v in info_e.items()
                       if k not in ("tp_corr", "attn_mean_weights")})
        for tp, r_tp in info_e["tp_corr"].items():
            entry[f"E_tp_{tp}"] = r_tp
        for tp, aw in info_e["attn_mean_weights"].items():
            entry[f"E_attn_{tp}"] = aw

        # --- Pick best strategy ---
        strategies = {
            "A_fc_proj": (proxy_a, info_a["r"]),
            "B_fc_nbr": (proxy_b, info_b["r"]),
            "C_pixel_pls": (proxy_c, info_c["r"]),
            "D_conv_pls": (proxy_d, info_d["r"]),
            "E_analytical": (proxy_e, info_e["r"]),
        }
        best_name = max(strategies, key=lambda k: strategies[k][1] if np.isfinite(strategies[k][1]) else -np.inf)

        # Override with forced strategy if requested
        force_map = {"A": "A_fc_proj", "B": "B_fc_nbr", "C": "C_pixel_pls",
                     "D": "D_conv_pls", "E": "E_analytical"}
        if args.force_strategy and force_map[args.force_strategy] in strategies:
            forced = force_map[args.force_strategy]
            print(f"  (auto-best: {best_name} r={strategies[best_name][1]:.4f}, "
                  f"forced: {forced} r={strategies[forced][1]:.4f})")
            best_name = forced

        best_proxy = strategies[best_name][0]
        best_r = strategies[best_name][1]
        best_proxy_per_feature[sf["feature"]] = best_proxy
        entry["best_strategy"] = best_name
        entry["best_r"] = best_r
        print(f"  >>> Best: {best_name} (r={best_r:.4f})")

        # --- Permutation test on best strategy ---
        if args.n_permutations > 0:
            # Build a closure for the best strategy
            # Use default args to capture current loop variables by value
            if best_name == "A_fc_proj":
                run_fn = lambda zt, _cf=conv_this, _wd=W_d, _bd=b_d, _ag=alpha_grid, _aw=attn_w: \
                    strategy_fc_projection(_cf, _wd, _bd, zt, _ag, _aw)
            elif best_name == "B_fc_nbr":
                run_fn = lambda zt, _cf=conv_this, _wf=W_full, _bf=b_full, _d=d, _ag=alpha_grid, _k=args.fc_neighborhood_k: \
                    strategy_fc_neighborhood(_cf, _wf, _bf, _d, zt, _ag, _k)
            elif best_name == "C_pixel_pls":
                run_fn = lambda zt, _xn=X_np, _ag=alpha_grid, _np=args.n_pls: \
                    strategy_raw_pixel_baseline(_xn, zt, _ag, _np)
            elif best_name == "E_analytical":
                run_fn = lambda zt, _cf=conv_this, _wf=W_full, _bf=b_full, _ap=all_attn_params[fi], _d=d: \
                    strategy_analytical_z(_cf, _wf, _bf, _ap, _d, zt)
            else:
                run_fn = lambda zt, _cf=conv_this, _np=args.n_pls: \
                    strategy_pls_conv(_cf, zt, _np)

            obs_r2, perm_p, null_r2 = permutation_test_proxy(
                run_fn, z_d, args.n_permutations, label=best_name
            )
            null_valid = null_r2[np.isfinite(null_r2)]
            null_mean = float(null_valid.mean()) if len(null_valid) > 0 else np.nan
            sig = "✓" if (np.isfinite(perm_p) and perm_p < 0.05) else "✗"
            print(f"  Permutation ({best_name}): p={perm_p:.4f} {sig}, "
                  f"null_mean_R²={null_mean:.4f}")
            entry.update({"perm_p": perm_p, "perm_null_mean_r2": null_mean})
            np.save(outdir / f"perm_null_{sf['feature']}_{best_name}.npy", null_r2)

        all_results.append(entry)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / "proxy_fit_quality.csv", index=False)

    # =========================================================================
    # 6. Build final proxy matrix and classify
    # =========================================================================
    print("\n" + "=" * 70)
    print("CLASSIFICATION (LOOCV)")
    print("=" * 70)

    proxy_final = np.column_stack([
        best_proxy_per_feature[sf["feature"]] for sf in selected_features
    ])
    proxy_cols = [sf["feature"] for sf in selected_features]
    df_proxy = pd.DataFrame(proxy_final, columns=proxy_cols)
    df_proxy.insert(0, "code", codes)
    df_proxy.insert(1, "group", groups)
    df_proxy.to_csv(outdir / "proxy_latent_matrix.csv", index=False)

    clf_results, clf_preds, clf_scores = run_classification_loocv(proxy_final, y)
    clf_df = pd.DataFrame(clf_results)
    clf_df.to_csv(outdir / "proxy_classification_loocv.csv", index=False)
    print("Proxy features:")
    print(clf_df.to_string(index=False))

    # True z reference
    dim_indices = [sf["dim"] for sf in selected_features]
    clf_true, _, _ = run_classification_loocv(Z_target[:, dim_indices], y)
    clf_true_df = pd.DataFrame(clf_true)
    clf_true_df.to_csv(outdir / "true_z_classification_loocv.csv", index=False)
    print("\nTrue z_agg (reference):")
    print(clf_true_df.to_string(index=False))

    for name in clf_preds:
        pd.DataFrame({"code": codes, "group": groups, "y_true": y,
                       "y_pred": clf_preds[name], "y_score": clf_scores[name]
                       }).to_csv(outdir / f"predictions_proxy_{name}.csv", index=False)

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for _, row in results_df.iterrows():
        p_str = f", perm_p={row.get('perm_p', 'N/A')}" if "perm_p" in row else ""
        print(f"  {row['feature']}: best={row['best_strategy']} r={row['best_r']:.4f}{p_str}")
    best_clf = clf_df.loc[clf_df["auc"].idxmax()]
    print(f"\nClassification: {best_clf['model']} acc={best_clf['accuracy']:.3f}, auc={best_clf['auc']:.3f}")

    manifest = {
        "pipeline_version": "step2_v3",
        "strategies": ["A_fc_projection", "B_fc_neighborhood", "C_raw_pixel_pls", "D_conv_pls", "E_analytical_z"],
        "features": [{"feature": sf["feature"], "fold": sf["fold_idx"]+1, "dim": sf["dim"],
                       "best_strategy": results_df.iloc[i]["best_strategy"],
                       "best_r": float(results_df.iloc[i]["best_r"])}
                      for i, sf in enumerate(selected_features)],
        "fc_neighborhood_k": args.fc_neighborhood_k,
        "n_pls": args.n_pls,
        "n_permutations": args.n_permutations,
        "n_samples": N, "n_folds": M,
        "classification": clf_df.to_dict(orient="records"),
    }
    with open(outdir / "step2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nAll outputs saved to: {outdir}")

    # =========================================================================
    # 8. Export inference bundle (optional)
    # =========================================================================
    if args.export_inference:
        from proxy_latent_inference import export_inference_bundle
        bundle_path = str(outdir / "inference_bundle.pt")
        export_inference_bundle(
            arch_file=args.arch_file,
            fold_paths=fold_paths,
            selected_features=selected_features,
            all_conv_feats=all_conv_feats,
            all_fc_W=all_fc_W,
            all_fc_b=all_fc_b,
            all_attn_params=all_attn_params,
            Z_target=Z_target,
            latent_dim=args.latent_dim,
            output_path=bundle_path,
            train_codes=codes,
            train_labels=y,
            export_classifier=True,
        )


if __name__ == "__main__":
    main()
