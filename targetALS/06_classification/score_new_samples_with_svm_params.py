#!/usr/bin/env python3
"""
Score new samples using exported linear SVM parameters (weights/intercept + thresholds).

Inputs:
  --proxy_csv         CSV with columns: subject (optional), group (optional), and proxy_dim_* features
  --svm_params_json   JSON produced by fit_final_linear_svm_and_export_params.py (svm_linear_params.json)
Output:
  --out_csv           Scored CSV with decision scores and predicted labels

Example (PowerShell):
  python score_new_samples_with_svm_params.py `
    --proxy_csv "...\new_proxy_latents.csv" `
    --svm_params_json "...\svm_linear_params.json" `
    --out_csv "...\svm_scores.csv"
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


def _read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_columns(df: pd.DataFrame, cols: List[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} required column(s) in {context}: {missing}\n"
            f"Available columns (first 30): {list(df.columns)[:30]}"
        )


def _predict_from_scores(scores: np.ndarray, thr: float, pos_group: str, neg_group: str) -> np.ndarray:
    # decision >= thr -> positive class
    return np.where(scores >= thr, pos_group, neg_group)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_csv", required=True, help="Proxy-latents CSV (with proxy_dim_* columns).")
    ap.add_argument("--svm_params_json", required=True, help="Exported SVM params JSON (svm_linear_params.json).")
    ap.add_argument("--out_csv", required=True, help="Where to write scored CSV.")
    ap.add_argument("--score_col", default="svm_score", help="Name of decision score column.")
    ap.add_argument("--strict", action="store_true", help="If set, error if any feature has NaN after parsing.")
    args = ap.parse_args()

    proxy_csv = Path(args.proxy_csv)
    params_json = Path(args.svm_params_json)
    out_csv = Path(args.out_csv)

    df = pd.read_csv(proxy_csv)

    # Support either "subject" or "code" as identifier.
    if "subject" not in df.columns and "code" in df.columns:
        df = df.rename(columns={"code": "subject"})

    params = _read_json(str(params_json))

    feature_cols = params.get("feature_cols", None)
    coef_w = params.get("coef_w", None)
    intercept_b = params.get("intercept_b", None)

    if feature_cols is None or coef_w is None or intercept_b is None:
        raise ValueError(
            "svm_params_json is missing one of required keys: feature_cols, coef_w, intercept_b"
        )

    feature_cols = list(feature_cols)
    w = np.asarray(coef_w, dtype=float).reshape(-1)
    b = float(intercept_b)

    if w.shape[0] != len(feature_cols):
        raise ValueError(
            f"Length mismatch: len(coef_w)={w.shape[0]} but len(feature_cols)={len(feature_cols)}"
        )

    _ensure_columns(df, feature_cols, context=str(proxy_csv))

    # Class names
    mapping = params.get("mapping", {})
    pos_group = params.get("positive_group", None)
    if pos_group is None:
        # fallback: assume label=1 is positive if mapping provided
        inv = {v: k for k, v in mapping.items()} if mapping else {}
        pos_group = inv.get(1, "POS")
    # Determine negative group name
    if mapping and pos_group in mapping:
        neg_group = [k for k in mapping.keys() if k != pos_group]
        neg_group = neg_group[0] if len(neg_group) == 1 else "NEG"
    else:
        neg_group = "NEG"

    thr0 = float(params.get("threshold_thr0", 0.0))
    thropt = float(params.get("threshold_operating", thr0))

    X = df[feature_cols].to_numpy(dtype=float)

    if args.strict and np.isnan(X).any():
        # identify first offending cell for debugging
        i, j = np.argwhere(np.isnan(X))[0]
        raise ValueError(
            f"NaN found in feature matrix at row={i}, col='{feature_cols[j]}'. "
            f"Check input proxy CSV parsing."
        )

    scores = X @ w + b

    df_out = df.copy()
    df_out[args.score_col] = scores
    df_out["pred_thr0"] = _predict_from_scores(scores, thr0, pos_group=pos_group, neg_group=neg_group)
    df_out["pred_operating"] = _predict_from_scores(scores, thropt, pos_group=pos_group, neg_group=neg_group)
    df_out["thr0"] = thr0
    df_out["thr_operating"] = thropt

    # Put key columns first if present
    front = [c for c in ["subject", "group", args.score_col, "pred_thr0", "pred_operating", "thr0", "thr_operating"] if c in df_out.columns]
    rest = [c for c in df_out.columns if c not in front]
    df_out = df_out[front + rest]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    print(f"Loaded proxy rows: {len(df):d}")
    print(f"Features used (n={len(feature_cols)}): {feature_cols}")
    print(f"Decision score: score = w·x + b,  b={b:.6g}")
    print(f"Thresholds: thr0={thr0:.6g}, operating={thropt:.6g}  (pos='{pos_group}', neg='{neg_group}')")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
