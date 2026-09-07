#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chirality_direct_pipeline.py
============================

Direct chirality-guided feature extraction for ALS/CTRL classification.

Instead of using an autoencoder to discover ROIs, this pipeline extracts
spectral features directly from 12 known DNA chirality positions in the
EEM matrix. These positions are biologically grounded and invariant to
matrix size.

FEATURE EXTRACTION:
  For each of 12 chirality positions:
    - Extract a small patch (configurable radius) centered on the position
    - Aggregate the patch into scalar features using multiple statistics:
        * mean: average intensity in patch
        * max:  peak intensity in patch
        * std:  variability within patch
    - Repeat for each of 3 timepoints (0h, 6h, 24h)

FEATURE MODES:
  - "separate":  12 positions × 3 timepoints = 36 features per statistic
  - "delta":     12 positions × 2 deltas (6h-0h, 24h-0h) = 24 features per stat
  - "all":       separate + delta = 60 features per statistic
  - "ratio":     12 positions × 2 ratios (6h/0h, 24h/0h) = 24 features per stat

STATISTICS (--stats):
  - "mean":      patch mean only
  - "mean,max":  mean + max (doubles feature count)
  - "mean,max,std": all three (triples feature count)

CLASSIFICATION:
  - Strict LOOCV (no leakage at any step)
  - In-fold feature selection: fANOVA + collinearity pruning
  - SVM (linear), Random Forest, Logistic Regression

Run example:
  python chirality_direct_pipeline.py ^
    --tp_dirs "out_0h,out_6h,out_24h" ^
    --labels_csv "sample_labels.csv" ^
    --output_dir "./chirality_results" ^
    --patch_radius 3 ^
    --feature_mode all ^
    --stats mean,max,std ^
    --fanova_k 15 ^
    --collinear_thresh 0.90
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# =============================================================================
# DNA Chirality Positions (from Coordinates_DNA.txt)
# =============================================================================

CHIRALITY_POSITIONS = [
    {"name": "8,3",  "emission_nm": 973.98,  "excitation_nm": 673.94},
    {"name": "6,5",  "emission_nm": 987.82,  "excitation_nm": 577.12},
    {"name": "7,5",  "emission_nm": 1047.81, "excitation_nm": 653.32},
    {"name": "10,2", "emission_nm": 1080.60, "excitation_nm": 745.92},
    {"name": "9,4",  "emission_nm": 1131.96, "excitation_nm": 731.39},
    {"name": "8,4",  "emission_nm": 1130.34, "excitation_nm": 599.78},
    {"name": "7,6",  "emission_nm": 1138.19, "excitation_nm": 659.79},
    {"name": "8,6",  "emission_nm": 1200.03, "excitation_nm": 727.40},
    {"name": "8,7",  "emission_nm": 1288.27, "excitation_nm": 740.87},
    {"name": "9,5",  "emission_nm": 1262.98, "excitation_nm": 685.15},
    {"name": "10,3", "emission_nm": 1267.70, "excitation_nm": 648.97},
    {"name": "10,5", "emission_nm": 1282.97, "excitation_nm": 801.23},
]


# =============================================================================
# Data Loading
# =============================================================================

def load_spectral_axes(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract emission (row) and excitation (column) wavelength axes from Excel."""
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    header = rows[0]
    excitation = []
    for v in header[1:]:
        if v is None:
            continue
        s = str(v).replace("Excitation_", "").replace("nm", "").strip()
        s = re.sub(r"^0+(\d)", r"\1", s)  # strip leading zeros
        try:
            excitation.append(float(s))
        except ValueError:
            pass

    emission = []
    for r in rows[1:]:
        if r[0] is None:
            continue
        try:
            emission.append(float(r[0]))
        except (ValueError, TypeError):
            pass

    return np.array(emission), np.array(excitation)


def load_excel_as_array(filepath: str) -> np.ndarray:
    """Load Excel EEM data as min-max normalized numpy array."""
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
                row_vals.append(0.0)
        if row_vals:
            data.append(row_vals)

    arr = np.array(data, dtype=np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def load_all_samples(codes: List[str], tp_dirs: List[str]) -> np.ndarray:
    """Load all samples across 3 timepoints. Returns [N, 3, H, W]."""
    X_list = []
    for code in codes:
        tp_mats = []
        for tp_dir in tp_dirs:
            fp = os.path.join(tp_dir, f"{code}.xlsx")
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing: {fp}")
            tp_mats.append(load_excel_as_array(fp))
        X_list.append(np.stack(tp_mats, axis=0))
    return np.stack(X_list, axis=0)


# =============================================================================
# Chirality Position Mapping
# =============================================================================

def map_chirality_to_indices(
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    max_em_error_nm: float = 5.0,
    max_exc_error_nm: float = 10.0,
) -> List[Dict]:
    """
    Map chirality positions (in nm) to pixel indices in the EEM grid.
    Returns list of dicts with 'name', 'em_idx', 'exc_idx', 'em_nm', 'exc_nm', 'em_error', 'exc_error'.
    """
    mapped = []
    for pos in CHIRALITY_POSITIONS:
        em_idx = int(np.argmin(np.abs(emission_axis - pos["emission_nm"])))
        exc_idx = int(np.argmin(np.abs(excitation_axis - pos["excitation_nm"])))
        em_err = abs(emission_axis[em_idx] - pos["emission_nm"])
        exc_err = abs(excitation_axis[exc_idx] - pos["excitation_nm"])

        if em_err > max_em_error_nm or exc_err > max_exc_error_nm:
            print(f"  WARNING: chirality {pos['name']} mapping error too large "
                  f"(em={em_err:.1f}nm, exc={exc_err:.1f}nm). Skipping.")
            continue

        mapped.append({
            "name": pos["name"],
            "em_idx": em_idx,
            "exc_idx": exc_idx,
            "em_nm": float(emission_axis[em_idx]),
            "exc_nm": float(excitation_axis[exc_idx]),
            "em_target_nm": pos["emission_nm"],
            "exc_target_nm": pos["excitation_nm"],
            "em_error_nm": float(em_err),
            "exc_error_nm": float(exc_err),
        })
    return mapped


def extract_patch(matrix: np.ndarray, em_idx: int, exc_idx: int, radius: int) -> np.ndarray:
    """Extract a (2r+1) x (2r+1) patch centered at (em_idx, exc_idx), clamped to bounds."""
    H, W = matrix.shape
    em_lo = max(0, em_idx - radius)
    em_hi = min(H - 1, em_idx + radius)
    exc_lo = max(0, exc_idx - radius)
    exc_hi = min(W - 1, exc_idx + radius)
    return matrix[em_lo:em_hi + 1, exc_lo:exc_hi + 1]


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_chirality_features(
    X_np: np.ndarray,           # [N, 3, H, W]
    chirality_map: List[Dict],
    patch_radius: int,
    feature_mode: str,          # "separate", "delta", "all", "ratio"
    stats: List[str],           # e.g. ["mean", "max", "std"]
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract chirality-guided features for all samples.

    Returns:
        features: [N, n_features]
        feature_names: list of feature name strings
    """
    N, T, H, W = X_np.shape
    tp_names = ["0h", "6h", "24h"]
    n_pos = len(chirality_map)

    # Step 1: extract patch statistics for each sample × position × timepoint
    # Shape: [N, n_pos, 3, n_stats]
    stat_funcs = {
        "mean": lambda p: float(np.mean(p)),
        "max": lambda p: float(np.max(p)),
        "std": lambda p: float(np.std(p)),
        "median": lambda p: float(np.median(p)),
        "sum": lambda p: float(np.sum(p)),
        "skew": lambda p: float(_safe_skew(p)),
    }

    used_stats = [s.strip() for s in stats if s.strip() in stat_funcs]
    if not used_stats:
        used_stats = ["mean"]

    # raw_vals[i, pos, tp, stat_idx]
    raw_vals = np.zeros((N, n_pos, 3, len(used_stats)), dtype=np.float64)

    for i in range(N):
        for p_idx, pos in enumerate(chirality_map):
            for t in range(3):
                patch = extract_patch(X_np[i, t], pos["em_idx"], pos["exc_idx"], patch_radius)
                for s_idx, stat_name in enumerate(used_stats):
                    raw_vals[i, p_idx, t, s_idx] = stat_funcs[stat_name](patch)

    # Step 2: build feature matrix based on mode
    features = []
    names = []

    for s_idx, stat_name in enumerate(used_stats):
        if feature_mode in ("separate", "all"):
            # 12 positions × 3 timepoints
            for p_idx, pos in enumerate(chirality_map):
                for t, tp in enumerate(tp_names):
                    names.append(f"ch{pos['name']}_{tp}_{stat_name}")
                    features.append(raw_vals[:, p_idx, t, s_idx])

        if feature_mode in ("delta", "all"):
            # 12 positions × 2 deltas (6h-0h, 24h-0h)
            for p_idx, pos in enumerate(chirality_map):
                d6 = raw_vals[:, p_idx, 1, s_idx] - raw_vals[:, p_idx, 0, s_idx]
                d24 = raw_vals[:, p_idx, 2, s_idx] - raw_vals[:, p_idx, 0, s_idx]
                names.append(f"ch{pos['name']}_d6h_{stat_name}")
                features.append(d6)
                names.append(f"ch{pos['name']}_d24h_{stat_name}")
                features.append(d24)

        if feature_mode == "ratio":
            # 12 positions × 2 ratios (6h/0h, 24h/0h)
            for p_idx, pos in enumerate(chirality_map):
                v0 = raw_vals[:, p_idx, 0, s_idx]
                v6 = raw_vals[:, p_idx, 1, s_idx]
                v24 = raw_vals[:, p_idx, 2, s_idx]
                # Safe ratio: avoid division by zero
                r6 = np.where(np.abs(v0) > 1e-10, v6 / v0, 0.0)
                r24 = np.where(np.abs(v0) > 1e-10, v24 / v0, 0.0)
                names.append(f"ch{pos['name']}_r6h_{stat_name}")
                features.append(r6)
                names.append(f"ch{pos['name']}_r24h_{stat_name}")
                features.append(r24)

    X_feat = np.column_stack(features) if features else np.zeros((N, 0))
    return X_feat, names


def _safe_skew(arr):
    """Compute skewness, returning 0 for degenerate arrays."""
    flat = arr.flatten()
    if len(flat) < 3 or np.std(flat) < 1e-12:
        return 0.0
    m = np.mean(flat)
    s = np.std(flat)
    return float(np.mean(((flat - m) / s) ** 3))


# =============================================================================
# Feature Selection
# =============================================================================

def select_features_fanova_collinear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fanova_k: int = 15,
    collinear_thresh: float = 0.90,
    min_keep: int = 3,
) -> List[int]:
    """Select top fANOVA features, then prune by collinearity."""
    n_feat = X_train.shape[1]
    if n_feat == 0:
        return []

    # Compute F-scores
    var = np.var(X_train, axis=0)
    ok = var > 1e-12
    F = np.zeros(n_feat)
    if ok.any():
        F_ok, _ = f_classif(X_train[:, ok], y_train)
        F[ok] = np.nan_to_num(F_ok, nan=0.0)

    order = np.argsort(-F)
    order = order[:min(fanova_k, len(order))]

    # Collinearity pruning
    keep = []
    for idx in order:
        idx = int(idx)
        if not keep:
            keep.append(idx)
            continue
        x = X_train[:, idx]
        redundant = False
        for j in keep:
            r = np.corrcoef(x, X_train[:, j])[0, 1]
            if np.isnan(r):
                continue
            if abs(r) > collinear_thresh:
                redundant = True
                break
        if not redundant:
            keep.append(idx)

    if len(keep) < min_keep:
        keep = [int(i) for i in order[:min(min_keep, len(order))]]

    return sorted(set(keep))


# =============================================================================
# Evaluation
# =============================================================================

def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List[str],
    feature_names: List[str],
    fanova_k: int = 15,
    collinear_thresh: float = 0.90,
    min_keep: int = 3,
    disable_fs: bool = False,
    svm_C: float = 1.0,
    lr_C: float = 1.0,
    rf_trees: int = 500,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Run strict LOOCV classification.

    Returns:
        results_df:   classification metrics per model
        predictions:  dict of model_name -> DataFrame with per-sample predictions
        fs_counts_df: feature selection frequency
        fs_per_fold:  per-fold feature selection details
    """
    N, P = X.shape

    classifiers = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=svm_C, probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]),
        "RF": RandomForestClassifier(
            n_estimators=rf_trees, random_state=SEED,
            n_jobs=-1, class_weight="balanced_subsample",
        ),
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=lr_C,
                                       class_weight="balanced", solver="liblinear",
                                       random_state=SEED)),
        ]),
    }

    loo = LeaveOneOut()
    y_score = {name: np.full(N, np.nan) for name in classifiers}
    y_pred = {name: np.full(N, -1, dtype=int) for name in classifiers}

    fs_counts = {fn: 0 for fn in feature_names}
    fs_rows = []

    for fold_idx, (train_idx, test_idx_arr) in enumerate(loo.split(np.arange(N))):
        test_idx = int(test_idx_arr[0])

        Xtr = X[train_idx]
        Xte = X[test_idx:test_idx + 1]
        ytr = y[train_idx]

        # In-fold feature selection
        if not disable_fs and P > 3:
            keep = select_features_fanova_collinear(
                Xtr, ytr,
                fanova_k=fanova_k,
                collinear_thresh=collinear_thresh,
                min_keep=min_keep,
            )
            if not keep:
                keep = list(range(P))
        else:
            keep = list(range(P))

        kept_names = [feature_names[i] for i in keep]
        fs_rows.append({
            "fold": fold_idx,
            "held_out": test_idx,
            "n_keep": len(keep),
            "kept_features": ";".join(kept_names),
        })
        for fn in kept_names:
            fs_counts[fn] += 1

        Xtr_sel = Xtr[:, keep]
        Xte_sel = Xte[:, keep]

        # Classify
        for name, clf in classifiers.items():
            clf.fit(Xtr_sel, ytr)
            if hasattr(clf, "predict_proba"):
                ppos = float(clf.predict_proba(Xte_sel)[0, 1])
            elif hasattr(clf, "decision_function"):
                ppos = float(clf.decision_function(Xte_sel)[0])
            else:
                ppos = float(clf.predict(Xte_sel)[0])
            y_score[name][test_idx] = ppos
            y_pred[name][test_idx] = 1 if ppos >= 0.5 else 0

    # Compile results
    results = []
    predictions = {}
    for name in classifiers:
        yp = y_pred[name]
        ys = y_score[name]

        acc = accuracy_score(y, yp)
        bal = balanced_accuracy_score(y, yp)
        cm = confusion_matrix(y, yp, labels=[0, 1])
        tn, fp, fn_val, tp = cm.ravel()
        sens = tp / (tp + fn_val) if (tp + fn_val) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        try:
            auc = roc_auc_score(y, ys)
        except:
            auc = np.nan

        results.append({
            "model": name,
            "acc": acc, "bal_acc": bal, "auc": auc,
            "sens": sens, "spec": spec,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn_val),
        })

        predictions[name] = pd.DataFrame({
            "sample_id": sample_ids,
            "y_true": y,
            "y_score": ys,
            "y_pred": yp,
        })

    results_df = pd.DataFrame(results)

    fs_counts_df = pd.DataFrame({
        "feature": list(fs_counts.keys()),
        "count": list(fs_counts.values()),
        "fraction": [c / N for c in fs_counts.values()],
    }).sort_values(["count", "feature"], ascending=[False, True])

    fs_per_fold = pd.DataFrame(fs_rows)

    return results_df, predictions, fs_counts_df, fs_per_fold


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Chirality-guided direct feature pipeline for ALS/CTRL classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--tp_dirs", type=str, required=True,
                    help="Comma-separated: out_0h,out_6h,out_24h")
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--chirality_file", type=str, default=None,
                    help="Optional path to Coordinates_DNA.txt (uses built-in if not specified)")

    p.add_argument("--patch_radius", type=int, default=3,
                    help="Patch radius around each chirality position (default=3 -> 7x7)")
    p.add_argument("--feature_mode", type=str, default="all",
                    choices=["separate", "delta", "all", "ratio"],
                    help="Feature construction mode")
    p.add_argument("--stats", type=str, default="mean,max,std",
                    help="Comma-separated patch statistics (mean,max,std,median,sum,skew)")

    p.add_argument("--fanova_k", type=int, default=15)
    p.add_argument("--collinear_thresh", type=float, default=0.90)
    p.add_argument("--min_selected_features", type=int, default=3)
    p.add_argument("--disable_feature_selection", action="store_true")

    p.add_argument("--svm_C", type=float, default=1.0)
    p.add_argument("--lr_C", type=float, default=1.0)
    p.add_argument("--rf_trees", type=int, default=500)

    p.add_argument("--sweep_radii", action="store_true",
                    help="Run classification for radii 0,1,2,3,5 and report comparison")

    args = p.parse_args()

    tp_dirs = [s.strip() for s in args.tp_dirs.split(",") if s.strip()]
    if len(tp_dirs) != 3:
        raise ValueError("Need exactly 3 timepoint directories (0h, 6h, 24h)")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats_list = [s.strip() for s in args.stats.split(",") if s.strip()]

    print("=" * 70)
    print("CHIRALITY-GUIDED DIRECT FEATURE PIPELINE")
    print("=" * 70)
    print(f"Output: {outdir}")
    print(f"Patch radius: {args.patch_radius} ({2*args.patch_radius+1}x{2*args.patch_radius+1})")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Statistics: {stats_list}")
    print(f"Feature selection: {'OFF' if args.disable_feature_selection else 'ON'}")
    if not args.disable_feature_selection:
        print(f"  fANOVA k={args.fanova_k}, collinear_thresh={args.collinear_thresh}")

    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    codes = labels_df["code"].astype(str).tolist()
    groups = labels_df["group"].tolist()
    y = np.array([1 if str(g).strip().lower() in ("als", "1") else 0 for g in groups], dtype=int)
    N = len(codes)
    print(f"\nSamples: {N} (ALS: {y.sum()}, CTRL: {N - y.sum()})")

    # Load data
    print("Loading spectral data...")
    X_np = load_all_samples(codes, tp_dirs)
    print(f"  Shape: {X_np.shape}")

    # Get spectral axes from first file
    first_file = os.path.join(tp_dirs[0], f"{codes[0]}.xlsx")
    emission_axis, excitation_axis = load_spectral_axes(first_file)
    print(f"  Emission: {len(emission_axis)} points [{emission_axis[0]:.1f} - {emission_axis[-1]:.1f}] nm")
    print(f"  Excitation: {len(excitation_axis)} points [{excitation_axis[0]:.1f} - {excitation_axis[-1]:.1f}] nm")

    # Map chirality positions
    chirality_map = map_chirality_to_indices(emission_axis, excitation_axis)
    print(f"\nMapped {len(chirality_map)}/{len(CHIRALITY_POSITIONS)} chirality positions:")
    for pos in chirality_map:
        print(f"  [{pos['name']}] em={pos['em_nm']:.1f}nm (err={pos['em_error_nm']:.1f}), "
              f"exc={pos['exc_nm']:.1f}nm (err={pos['exc_error_nm']:.1f}), "
              f"idx=({pos['em_idx']},{pos['exc_idx']})")

    # Save mapping
    pd.DataFrame(chirality_map).to_csv(outdir / "chirality_mapping.csv", index=False)

    # =========================================================================
    # Single run or sweep
    # =========================================================================
    if args.sweep_radii:
        print("\n" + "=" * 70)
        print("RADIUS SWEEP MODE")
        print("=" * 70)

        radii = [0, 1, 2, 3, 5]
        sweep_results = []

        for radius in radii:
            print(f"\n--- Radius {radius} ({2*radius+1}x{2*radius+1}) ---")
            X_feat, feat_names = extract_chirality_features(
                X_np, chirality_map, radius, args.feature_mode, stats_list
            )
            print(f"  Features: {X_feat.shape[1]}")

            results_df, predictions, fs_counts_df, fs_per_fold = run_loocv(
                X_feat, y, codes, feat_names,
                fanova_k=args.fanova_k,
                collinear_thresh=args.collinear_thresh,
                min_keep=args.min_selected_features,
                disable_fs=args.disable_feature_selection,
                svm_C=args.svm_C,
                lr_C=args.lr_C,
                rf_trees=args.rf_trees,
            )

            for _, row in results_df.iterrows():
                sweep_results.append({
                    "radius": radius,
                    "patch_size": f"{2*radius+1}x{2*radius+1}",
                    "n_features": X_feat.shape[1],
                    **row.to_dict(),
                })

            print(results_df[["model", "acc", "bal_acc", "auc", "sens", "spec"]].to_string(index=False))

        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(outdir / "radius_sweep_results.csv", index=False)
        print(f"\n{'=' * 70}")
        print("RADIUS SWEEP SUMMARY")
        print(f"{'=' * 70}")
        print(sweep_df[["radius", "patch_size", "model", "acc", "bal_acc", "auc"]].to_string(index=False))

    else:
        # Single run
        print(f"\n{'=' * 70}")
        print("EXTRACTING CHIRALITY FEATURES")
        print(f"{'=' * 70}")

        X_feat, feat_names = extract_chirality_features(
            X_np, chirality_map, args.patch_radius, args.feature_mode, stats_list
        )
        print(f"  Total features: {X_feat.shape[1]}")
        print(f"  Feature names (first 10): {feat_names[:10]}")

        # Save feature matrix
        df_feat = pd.DataFrame(X_feat, columns=feat_names)
        df_feat.insert(0, "sample_id", codes)
        df_feat.insert(1, "group", groups)
        df_feat.insert(2, "label", y)
        df_feat.to_csv(outdir / "chirality_feature_matrix.csv", index=False)
        print(f"  Saved: chirality_feature_matrix.csv")

        # Feature correlation with labels (quick check)
        print(f"\n  Top features by univariate F-score (full dataset, informational only):")
        var = np.var(X_feat, axis=0)
        ok = var > 1e-12
        if ok.any():
            F, pvals = f_classif(X_feat[:, ok], y)
            ok_names = [feat_names[i] for i in range(len(feat_names)) if ok[i]]
            top_idx = np.argsort(-F)[:15]
            for rank, idx in enumerate(top_idx):
                print(f"    {rank+1}. {ok_names[idx]:>35s}  F={F[idx]:.2f}  p={pvals[idx]:.4f}")

        # Classification
        print(f"\n{'=' * 70}")
        print("LOOCV CLASSIFICATION")
        print(f"{'=' * 70}")

        results_df, predictions, fs_counts_df, fs_per_fold = run_loocv(
            X_feat, y, codes, feat_names,
            fanova_k=args.fanova_k,
            collinear_thresh=args.collinear_thresh,
            min_keep=args.min_selected_features,
            disable_fs=args.disable_feature_selection,
            svm_C=args.svm_C,
            lr_C=args.lr_C,
            rf_trees=args.rf_trees,
        )

        # Save results
        results_df.to_csv(outdir / "classification_results.csv", index=False)
        for name, pred_df in predictions.items():
            pred_df.to_csv(outdir / f"predictions_{name}.csv", index=False)
        fs_counts_df.to_csv(outdir / "feature_selection_counts.csv", index=False)
        fs_per_fold.to_csv(outdir / "feature_selection_per_fold.csv", index=False)

        print(f"\n{'=' * 70}")
        print("RESULTS")
        print(f"{'=' * 70}")
        print(results_df[["model", "acc", "bal_acc", "auc", "sens", "spec"]].to_string(index=False))

        # Top selected features
        print(f"\nTop 10 most frequently selected features:")
        for _, row in fs_counts_df.head(10).iterrows():
            print(f"  {row['feature']:>40s}  selected {row['count']}/{N} folds ({row['fraction']:.1%})")

    # Save config
    config = {
        "version": "chirality_direct_v1",
        "n_samples": N,
        "n_als": int(y.sum()),
        "n_ctrl": int(N - y.sum()),
        "n_chirality_positions": len(chirality_map),
        "patch_radius": args.patch_radius,
        "feature_mode": args.feature_mode,
        "stats": stats_list,
        "fanova_k": args.fanova_k,
        "collinear_thresh": args.collinear_thresh,
        "sweep_radii": args.sweep_radii,
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAll results saved to: {outdir}")


if __name__ == "__main__":
    main()
