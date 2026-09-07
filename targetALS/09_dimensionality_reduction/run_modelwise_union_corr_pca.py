#!/usr/bin/env python3
"""
run_modelwise_union_corr_pca.py

For each *model file* (e.g., combined_union0.csv ... combined_union4.csv), where
latent columns are wide and *indexed by timepoint suffix* like:

  dim0_0h, dim1_0h, ..., dim511_0h,
  dim0_6h, ..., dim511_6h,
  dim0_24h, ..., dim511_24h

this script:
1) Splits the wide file into per-timepoint latent matrices.
2) Correlates latent dims with clinical outcomes (unadjusted + covariate-adjusted),
   using Pearson + Spearman.
3) Computes BH-FDR (Benjamini–Hochberg) within each (model × timepoint × method × adjusted)
   across (dims × outcomes).
4) Runs PCA per timepoint (unadjusted + covariate-adjusted residual PCA),
   correlates PCs with outcomes + global BH across (PC×outcome), and optional LOOCV prediction.

Outputs: ONE Excel per model (so 5 files for 5 models).

Dependencies:
  pip install pandas numpy scipy scikit-learn openpyxl matplotlib

Example (Windows):
  python run_modelwise_union_corr_pca.py ^
    --input_glob "C:\\path\\to\\combined_union*.csv" ^
    --early_slope "C:\\path\\early_slope.xlsx" ^
    --covariates "C:\\path\\clinical_covariates.xlsx" ^
    --out_dir "C:\\path\\out_modelwise"

Or explicitly:
  python run_modelwise_union_corr_pca.py \
    --model_csv "model0=C:/.../combined_union0.csv" \
    --model_csv "model1=C:/.../combined_union1.csv" \
    --model_csv "model2=C:/.../combined_union2.csv" \
    --model_csv "model3=C:/.../combined_union3.csv" \
    --model_csv "model4=C:/.../combined_union4.csv" \
    --early_slope C:/.../early_slope.xlsx \
    --covariates C:/.../clinical_covariates.xlsx \
    --out_dir C:/.../out_modelwise
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt


# -----------------------------
# Stats utilities
# -----------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def corr_vec(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    if method == "pearson":
        return stats.pearsonr(x, y)
    if method == "spearman":
        return stats.spearmanr(x, y)
    raise ValueError(f"Unknown method: {method}")


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residualize y on covariates X (adds intercept)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    Xc = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


def coerce_yes_no(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1", "taking", "on"):
        return 1.0
    if s in ("no", "n", "false", "0", "none", "not taking", "off"):
        return 0.0
    return np.nan


def coerce_sex(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("male", "m"):
        return 1.0
    if s in ("female", "f"):
        return 0.0
    return np.nan


# -----------------------------
# Column parsing (wide union files)
# -----------------------------
DIM_TP_RE = re.compile(r"^dim(\d+)_(.+)$")


def detect_timepoints_and_dims(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns:
      timepoints: sorted list of timepoint strings (as they appear in suffix, e.g. '0h','6h','24h')
      tp_to_cols: dict timepoint -> list of corresponding columns in df
    """
    tp_to_cols: Dict[str, List[Tuple[int, str]]] = {}
    for c in df.columns:
        m = DIM_TP_RE.match(str(c))
        if not m:
            continue
        dim_idx = int(m.group(1))
        tp = m.group(2)
        tp_to_cols.setdefault(tp, []).append((dim_idx, c))

    # sort cols within each timepoint by dim index
    tp_to_cols_sorted: Dict[str, List[str]] = {}
    for tp, pairs in tp_to_cols.items():
        pairs.sort(key=lambda x: x[0])
        tp_to_cols_sorted[tp] = [c for _, c in pairs]

    # sort timepoints nicely if they look like "0h/6h/24h"
    def tp_key(t: str):
        mm = re.match(r"^(\d+)\s*h$", t.strip().lower())
        return (0, int(mm.group(1))) if mm else (1, t)

    timepoints = sorted(tp_to_cols_sorted.keys(), key=tp_key)
    return timepoints, tp_to_cols_sorted


def build_timepoint_df(df_wide: pd.DataFrame, id_col: str, group_col: Optional[str], tp: str, cols: List[str]) -> pd.DataFrame:
    """Extract one timepoint into a standard latent dataframe: code, group, dim0..dimN."""
    out = df_wide[[id_col] + ([group_col] if group_col and group_col in df_wide.columns else []) + cols].copy()
    # rename dimK_tp -> dimK
    rename = {}
    for c in cols:
        m = DIM_TP_RE.match(str(c))
        if m:
            rename[c] = f"dim{m.group(1)}"
    out = out.rename(columns=rename)
    out["timepoint"] = tp
    return out


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    id_col: str
    group_col: Optional[str]
    pos_group: str
    clinical_cols: List[str]
    cov_cols_raw: List[str]
    n_pcs: int
    loocv_ks: List[int]


# -----------------------------
# Merging clinical + covariates
# -----------------------------
def merge_with_clinical(
    latent_df: pd.DataFrame,
    early_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    cfg: Config,
    model_name: str,
) -> pd.DataFrame:
    df = latent_df.copy()
    df[cfg.id_col] = df[cfg.id_col].astype(str)

    early = early_df[[cfg.id_col] + cfg.clinical_cols].copy()
    early[cfg.id_col] = early[cfg.id_col].astype(str)

    cov = cov_df[[cfg.id_col] + cfg.cov_cols_raw].copy()
    cov[cfg.id_col] = cov[cfg.id_col].astype(str)

    m = df.merge(early, on=cfg.id_col, how="inner").merge(cov, on=cfg.id_col, how="left")
    m["model"] = model_name

    # numeric covariates
    m["_age"] = pd.to_numeric(m["Age Years"], errors="coerce")
    m["_sex_male"] = m["Sex"].apply(coerce_sex)
    m["_edaravone"] = m["Taking Radicava Edaravone"].apply(coerce_yes_no)
    m["_riluzole"] = m["Taking Rilutek Riluzole"].apply(coerce_yes_no)

    return m


def latent_dim_cols(df: pd.DataFrame) -> List[str]:
    """Return dim0..dimN columns (sorted)."""
    cols = []
    for c in df.columns:
        if re.fullmatch(r"dim\d+", str(c)):
            cols.append(c)
    cols.sort(key=lambda x: int(x.replace("dim", "")))
    return cols


def als_subset_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    if not cfg.group_col or cfg.group_col not in df.columns:
        return np.ones(len(df), dtype=bool)
    return (df[cfg.group_col].astype(str).str.upper() == cfg.pos_group.upper()).to_numpy()


# -----------------------------
# Correlations (latent dims vs outcomes)
# -----------------------------
def compute_latent_outcome_corr(m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cov_num_cols = ["_age", "_sex_male", "_edaravone", "_riluzole"]
    dims = latent_dim_cols(m)

    rows = []
    for method in ("pearson", "spearman"):
        for adjusted in (False, True):
            pvals = []
            tmp = []

            for d in dims:
                x = pd.to_numeric(m[d], errors="coerce").to_numpy()
                for clin in cfg.clinical_cols:
                    y = pd.to_numeric(m[clin], errors="coerce").to_numpy()

                    mask = np.isfinite(x) & np.isfinite(y)
                    if adjusted:
                        C = m[cov_num_cols].to_numpy(dtype=float)
                        mask = mask & np.all(np.isfinite(C), axis=1)
                        if mask.sum() < 5:
                            r = np.nan
                            p = np.nan
                            n = int(mask.sum())
                        else:
                            xr = residualize(x[mask], C[mask])
                            yr = residualize(y[mask], C[mask])
                            r, p = corr_vec(xr, yr, method)
                            n = int(mask.sum())
                    else:
                        if mask.sum() < 5:
                            r = np.nan
                            p = np.nan
                            n = int(mask.sum())
                        else:
                            r, p = corr_vec(x[mask], y[mask], method)
                            n = int(mask.sum())

                    tmp.append(
                        {
                            "model": m["model"].iloc[0],
                            "timepoint": m["timepoint"].iloc[0],
                            "method": method,
                            "adjusted": adjusted,
                            "latent_dim": d,
                            "clinical_var": clin,
                            "r": float(r) if np.isfinite(r) else np.nan,
                            "p": float(p) if np.isfinite(p) else np.nan,
                            "n": n,
                        }
                    )
                    pvals.append(p if np.isfinite(p) else 1.0)

            q = bh_qvalues(np.array(pvals, dtype=float))
            for rec, qi in zip(tmp, q):
                rec["q"] = float(qi)
                rows.append(rec)

    return pd.DataFrame(rows)


# -----------------------------
# PCA + PC correlations + LOOCV
# -----------------------------
def pc_analysis_per_timepoint(
    m: pd.DataFrame,
    cfg: Config,
    clinical_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PCA on ALS-only (if group exists) within this timepoint:
      - unadjusted PCA on centered X
      - cov_adj PCA on residualized X (on covariates)
    Correlate PCs with outcomes and compute:
      - q_within_outcome: BH across PCs within each outcome
      - q_global: BH across all (PC×outcome) within each (adjustment×method) block
    LOOCV regression: predict outcome from first k PCs.
    """
    cov_num_cols = ["_age", "_sex_male", "_edaravone", "_riluzole"]
    dims = latent_dim_cols(m)

    df = m.copy()
    df = df[als_subset_mask(df, cfg)].copy()

    X = df[dims].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    C = df[cov_num_cols].to_numpy(dtype=float)
    Y = {name: pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for name, col in clinical_map.items()}

    # unadj PCA on complete X
    mask_X = np.all(np.isfinite(X), axis=1)
    X_u = X[mask_X]
    X_u_c = StandardScaler(with_mean=True, with_std=False).fit_transform(X_u)
    pca_u = PCA(n_components=min(cfg.n_pcs, X_u_c.shape[0], X_u_c.shape[1]))
    PCs_u = pca_u.fit_transform(X_u_c)
    var_u = pca_u.explained_variance_ratio_

    # cov_adj PCA on complete X and C
    mask_XC = mask_X & np.all(np.isfinite(C), axis=1)
    X_c = X[mask_XC]
    C_c = C[mask_XC]
    X_res = np.empty_like(X_c)
    for j in range(X_c.shape[1]):
        X_res[:, j] = residualize(X_c[:, j], C_c)
    X_res_c = StandardScaler(with_mean=True, with_std=False).fit_transform(X_res)
    pca_c = PCA(n_components=min(cfg.n_pcs, X_res_c.shape[0], X_res_c.shape[1]))
    PCs_c = pca_c.fit_transform(X_res_c)
    var_c = pca_c.explained_variance_ratio_

    corr_rows = []
    for adj_label, PCs, var, mask, do_y_resid in [
        ("unadj", PCs_u, var_u, mask_X, False),
        ("cov_adj", PCs_c, var_c, mask_XC, True),
    ]:
        for out_name, y_all in Y.items():
            y_sub = y_all[mask]

            if do_y_resid:
                y_finite = np.isfinite(y_sub)
                if y_finite.sum() < 5:
                    continue
                y_res = residualize(y_sub[y_finite], C_c[y_finite])
                PCs_use = PCs[y_finite]
                y_use = y_res
            else:
                y_finite = np.isfinite(y_sub)
                if y_finite.sum() < 5:
                    continue
                PCs_use = PCs[y_finite]
                y_use = y_sub[y_finite]

            for method in ("pearson", "spearman"):
                pvals = []
                tmp = []
                for k in range(PCs_use.shape[1]):
                    pc = PCs_use[:, k]
                    msk = np.isfinite(pc) & np.isfinite(y_use)
                    if msk.sum() < 5:
                        r = np.nan
                        p = np.nan
                        n = int(msk.sum())
                    else:
                        r, p = corr_vec(pc[msk], y_use[msk], method)
                        n = int(msk.sum())
                    tmp.append((k + 1, r, p, n))
                    pvals.append(p if np.isfinite(p) else 1.0)

                q_within = bh_qvalues(np.array(pvals, dtype=float))
                for (pc_idx, r, p, n), qi in zip(tmp, q_within):
                    corr_rows.append(
                        {
                            "model": m["model"].iloc[0],
                            "timepoint": m["timepoint"].iloc[0],
                            "adjustment": adj_label,
                            "method": method,
                            "outcome": out_name,
                            "PC": pc_idx,
                            "explained_var": float(var[pc_idx - 1]) if pc_idx - 1 < len(var) else np.nan,
                            "r": float(r) if np.isfinite(r) else np.nan,
                            "p": float(p) if np.isfinite(p) else np.nan,
                            "q_within_outcome": float(qi),
                            "n": n,
                        }
                    )

    pc_corr = pd.DataFrame(corr_rows)
    pc_corr["q_global"] = np.nan
    for (adj, method), idxs in pc_corr.groupby(["adjustment", "method"]).groups.items():
        pvals = pc_corr.loc[idxs, "p"].to_numpy(dtype=float)
        pvals = np.where(np.isfinite(pvals), pvals, 1.0)
        pc_corr.loc[idxs, "q_global"] = bh_qvalues(pvals)

    # LOOCV regression
    reg_rows = []
    loo = LeaveOneOut()
    for adj_label, PCs, var, mask, do_y_resid in [
        ("unadj", PCs_u, var_u, mask_X, False),
        ("cov_adj", PCs_c, var_c, mask_XC, True),
    ]:
        for out_name, y_all in Y.items():
            y_sub = y_all[mask]
            y_finite = np.isfinite(y_sub)
            if y_finite.sum() < 8:
                continue
            if do_y_resid:
                yv = residualize(y_sub[y_finite], C_c[y_finite])
            else:
                yv = y_sub[y_finite]
            PCv = PCs[y_finite]

            n = len(yv)
            sst = ((yv - yv.mean()) ** 2).sum()

            for k in cfg.loocv_ks:
                k_eff = int(min(k, PCv.shape[1], max(1, n - 2)))
                Xk = PCv[:, :k_eff]
                y_pred = np.zeros(n)

                for train, test in loo.split(Xk):
                    model = LinearRegression()
                    model.fit(Xk[train], yv[train])
                    y_pred[test[0]] = model.predict(Xk[test])[0]

                sse = ((yv - y_pred) ** 2).sum()
                r2 = 1 - sse / sst if sst > 0 else np.nan
                r_pred = stats.pearsonr(yv, y_pred)[0] if n > 2 else np.nan
                rmse = float(np.sqrt(np.mean((yv - y_pred) ** 2)))

                reg_rows.append(
                    {
                        "model": m["model"].iloc[0],
                        "timepoint": m["timepoint"].iloc[0],
                        "adjustment": adj_label,
                        "outcome": out_name,
                        "k_PCs": k_eff,
                        "n": n,
                        "LOOCV_R2": float(r2) if np.isfinite(r2) else np.nan,
                        "corr_obs_pred": float(r_pred) if np.isfinite(r_pred) else np.nan,
                        "RMSE": rmse,
                        "PCs_explained_var_cum": float(np.sum(var[:k_eff])) if k_eff <= len(var) else np.nan,
                    }
                )

    pc_reg = pd.DataFrame(reg_rows)

    notes = pd.DataFrame(
        [
            {
                "model": m["model"].iloc[0],
                "timepoint": m["timepoint"].iloc[0],
                "n_rows_merged": int(len(m)),
                "n_ALS_used": int(len(df)),
                "n_complete_X": int(mask_X.sum()),
                "n_complete_XC": int(mask_XC.sum()),
                "PC1_var_unadj": float(var_u[0]) if len(var_u) else np.nan,
                "PC1_var_cov_adj": float(var_c[0]) if len(var_c) else np.nan,
            }
        ]
    )
    return notes, pc_corr, pc_reg


def make_pc_heatmap(pc_corr_all: pd.DataFrame, out_png: str, pcs: int = 5):
    """Heatmap panel: rows=PC1..PC5, cols=outcomes, for each timepoint and adjustment, Pearson r."""
    tps = pc_corr_all["timepoint"].unique().tolist()
    tps.sort()
    fig, axes = plt.subplots(2, len(tps), figsize=(4 * len(tps), 6), constrained_layout=True)

    for j, tp in enumerate(tps):
        for i, adj in enumerate(["unadj", "cov_adj"]):
            ax = axes[i, j] if len(tps) > 1 else axes[i]
            sub = pc_corr_all[
                (pc_corr_all["timepoint"] == tp)
                & (pc_corr_all["adjustment"] == adj)
                & (pc_corr_all["method"] == "pearson")
                & (pc_corr_all["PC"] <= pcs)
            ]
            if sub.empty:
                ax.axis("off")
                continue
            mat = (
                sub.pivot_table(index="PC", columns="outcome", values="r", aggfunc="first")
                .reindex(index=list(range(1, pcs + 1)))
            )
            im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", vmin=-1, vmax=1)
            ax.set_title(f"{tp} • {adj} • Pearson r")
            ax.set_xticks(np.arange(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(pcs))
            ax.set_yticklabels([f"PC{k}" for k in range(1, pcs + 1)])

    fig.colorbar(im, ax=axes.ravel().tolist() if len(tps) > 1 else axes.ravel().tolist(), shrink=0.7)
    fig.savefig(out_png, dpi=200)


# -----------------------------
# Main
# -----------------------------
def parse_kv_arg(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"Expected key=path, got: {s}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_csv", action="append", help="modelName=path.csv (repeatable)")
    ap.add_argument("--input_glob", help="glob for model csvs, e.g. combined_union*.csv")
    ap.add_argument("--early_slope", required=True, help="early_slope.xlsx")
    ap.add_argument("--covariates", required=True, help="clinical_covariates.xlsx")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--id_col", default="code")
    ap.add_argument("--group_col", default="group")
    ap.add_argument("--pos_group", default="ALS")

    # clinical columns (match your files)
    ap.add_argument("--nfl_col", default="Nfl Concentration Pg Per Ml")
    ap.add_argument("--alsfrs_total_col", default="Total Alsfrsr Score")
    ap.add_argument("--alsfrs_slope_col", default="Early Slope (pts/mo)")

    # covariate columns (match your files)
    ap.add_argument("--age_col", default="Age Years")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--edaravone_col", default="Taking Radicava Edaravone")
    ap.add_argument("--riluzole_col", default="Taking Rilutek Riluzole")

    ap.add_argument("--n_pcs", type=int, default=10)
    ap.add_argument("--loocv_ks", default="1,2,3,5,10")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # collect model files
    model_files: List[Tuple[str, str]] = []
    if args.model_csv:
        model_files.extend([parse_kv_arg(s) for s in args.model_csv])
    if args.input_glob:
        for p in sorted(glob.glob(args.input_glob)):
            # infer model name from filename, e.g. combined_union3.csv -> model3
            base = os.path.basename(p)
            m = re.search(r"(\d+)", base)
            name = f"model{m.group(1)}" if m else os.path.splitext(base)[0]
            model_files.append((name, p))
    if not model_files:
        raise SystemExit("Provide --model_csv (repeatable) or --input_glob")

    # read clinical files
    early_df = pd.read_excel(args.early_slope)
    cov_df = pd.read_excel(args.covariates)

    # ensure id col exists
    if args.id_col not in early_df.columns and "code" in early_df.columns:
        early_df = early_df.rename(columns={"code": args.id_col})
    if args.id_col not in cov_df.columns and "code" in cov_df.columns:
        cov_df = cov_df.rename(columns={"code": args.id_col})

    cfg = Config(
        id_col=args.id_col,
        group_col=args.group_col,
        pos_group=args.pos_group,
        clinical_cols=[args.alsfrs_slope_col, args.alsfrs_total_col, args.nfl_col],
        cov_cols_raw=[args.age_col, args.sex_col, args.edaravone_col, args.riluzole_col],
        n_pcs=args.n_pcs,
        loocv_ks=[int(x) for x in args.loocv_ks.split(",") if x.strip()],
    )

    clinical_map = {
        "NFL": args.nfl_col,
        "ALSFRS_total": args.alsfrs_total_col,
        "ALSFRS_slope": args.alsfrs_slope_col,
    }

    for model_name, csv_path in model_files:
        wide = pd.read_csv(csv_path)

        # drop group_col if missing
        group_col = cfg.group_col if cfg.group_col in wide.columns else None
        cfg_local = Config(**{**cfg.__dict__, "group_col": group_col})

        # detect timepoints
        timepoints, tp_to_cols = detect_timepoints_and_dims(wide)
        if not timepoints:
            raise SystemExit(f"No dim*_TIMEPOINT columns found in {csv_path}")

        # per-timepoint results
        all_corr = []
        all_pc_notes = []
        all_pc_corr = []
        all_pc_reg = []

        # Excel output per model
        out_xlsx = os.path.join(args.out_dir, f"{model_name}_corr_and_pca.xlsx")
        heat_png = os.path.join(args.out_dir, f"{model_name}_pc_heatmap.png")

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            # model-level notes
            model_notes = pd.DataFrame([{
                "model": model_name,
                "source_csv": csv_path,
                "n_rows_wide": int(len(wide)),
                "n_cols_wide": int(wide.shape[1]),
                "timepoints_detected": ", ".join(timepoints),
                "group_col_used": group_col or "",
            }])
            model_notes.to_excel(writer, sheet_name="model_notes", index=False)

            for tp in timepoints:
                tp_df = build_timepoint_df(wide, cfg_local.id_col, cfg_local.group_col, tp, tp_to_cols[tp])
                merged = merge_with_clinical(tp_df, early_df, cov_df, cfg_local, model_name)

                # correlations
                corr_df = compute_latent_outcome_corr(merged, cfg_local)
                all_corr.append(corr_df)

                # save per-timepoint corr sheets
                sheet_all = f"{tp}_corr_all"
                corr_df.to_excel(writer, sheet_name=sheet_all[:31], index=False)

                top = corr_df[(corr_df["q"] <= 0.1) & corr_df["p"].notna()].sort_values(["q", "p"]).head(200)
                top.to_excel(writer, sheet_name=f"{tp}_corr_top_q0p1"[:31], index=False)

                # PCA + PC corr/reg
                notes, pc_corr, pc_reg = pc_analysis_per_timepoint(merged, cfg_local, clinical_map)
                all_pc_notes.append(notes)
                all_pc_corr.append(pc_corr)
                all_pc_reg.append(pc_reg)

                notes.to_excel(writer, sheet_name=f"{tp}_pca_notes"[:31], index=False)
                pc_corr.to_excel(writer, sheet_name=f"{tp}_pc_corr"[:31], index=False)
                pc_reg.to_excel(writer, sheet_name=f"{tp}_pc_loocv"[:31], index=False)

                top_pc = pc_corr[(pc_corr["q_global"] <= 0.1) & pc_corr["p"].notna()].sort_values(["q_global", "p"]).head(200)
                top_pc.to_excel(writer, sheet_name=f"{tp}_pc_top_q0p1"[:31], index=False)

            # combined sheets
            corr_all = pd.concat(all_corr, ignore_index=True)
            pc_notes_all = pd.concat(all_pc_notes, ignore_index=True) if all_pc_notes else pd.DataFrame()
            pc_corr_all = pd.concat(all_pc_corr, ignore_index=True) if all_pc_corr else pd.DataFrame()
            pc_reg_all = pd.concat(all_pc_reg, ignore_index=True) if all_pc_reg else pd.DataFrame()

            corr_all.to_excel(writer, sheet_name="ALL_timepoints_corr", index=False)
            pc_notes_all.to_excel(writer, sheet_name="ALL_timepoints_pca_notes", index=False)
            pc_corr_all.to_excel(writer, sheet_name="ALL_timepoints_pc_corr", index=False)
            pc_reg_all.to_excel(writer, sheet_name="ALL_timepoints_pc_loocv", index=False)

        # heatmap across timepoints for this model
        try:
            pc_corr_all = pd.concat(all_pc_corr, ignore_index=True) if all_pc_corr else pd.DataFrame()
            if not pc_corr_all.empty:
                make_pc_heatmap(pc_corr_all, heat_png, pcs=min(5, cfg_local.n_pcs))
        except Exception as e:
            print(f"[warn] heatmap failed for {model_name}: {e}")

        print(f"[ok] wrote {out_xlsx}")
        if os.path.exists(heat_png):
            print(f"[ok] wrote {heat_png}")


if __name__ == "__main__":
    main()
