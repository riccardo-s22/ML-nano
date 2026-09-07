#!/usr/bin/env python3
"""
run_latent_clinical_corr_and_pca.py

- Correlate latent vectors with clinical outcomes (unadjusted + covariate-adjusted),
  for each latent dataset (e.g., 0h/6h/24h or z_agg1..5).
- Compute BH-FDR (Benjamini–Hochberg) within each:
    (dataset × method × adjusted) across all latent_dims × outcomes.
- PCA on latent space (ALS-only by default if group column exists) and correlate PCs with outcomes
  (unadjusted PCA + covariate-adjusted PCA), plus LOOCV prediction using PCs.

Dependencies:
  pip install pandas numpy scipy scikit-learn openpyxl matplotlib

Example:
  python run_latent_clinical_corr_and_pca.py \
    --latent "0h=/mnt/data/ensemble_z_0h.csv" \
    --latent "6h=/mnt/data/ensemble_z_6h.csv" \
    --latent "24h=/mnt/data/ensemble_z_24h.csv" \
    --early_slope /mnt/data/early_slope.xlsx \
    --covariates /mnt/data/clinical_covariates.xlsx \
    --out_dir /mnt/data/results_corr_pca

Notes:
- Covariate adjustment is done via residualization:
    x_res = x - Xcov*beta_x ; y_res = y - Xcov*beta_y ; correlate(x_res, y_res)
- If controls lack covariates, adjusted analyses automatically drop them (complete-case).
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt


# -----------------------------
# Utilities
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


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualize y on covariates X.
    X is (n,k) without intercept. We add intercept internally.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    Xc = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return y - Xc @ beta


def corr_vec(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    if method == "pearson":
        return stats.pearsonr(x, y)
    if method == "spearman":
        return stats.spearmanr(x, y)
    raise ValueError(f"Unknown method: {method}")


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


def detect_latent_cols(df: pd.DataFrame, id_col: str, group_col: Optional[str]) -> List[str]:
    """Prefer dim\d+ columns if present; otherwise use numeric columns excluding id/group."""
    dim_cols = [c for c in df.columns if re.fullmatch(r"dim\d+", str(c))]
    if dim_cols:
        return dim_cols
    ignore = {id_col}
    if group_col:
        ignore.add(group_col)
    out = []
    for c in df.columns:
        if c in ignore:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if np.isfinite(v).sum() > 0:
            out.append(c)
    return out


# -----------------------------
# Core computations
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


def prepare_merged(
    latent_df: pd.DataFrame,
    early_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    cfg: Config,
    dataset_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge latent with early_slope and covariates; add numeric covariates columns."""
    df = latent_df.copy()
    df[cfg.id_col] = df[cfg.id_col].astype(str)

    early = early_df[[cfg.id_col] + cfg.clinical_cols].copy()
    early[cfg.id_col] = early[cfg.id_col].astype(str)

    cov = cov_df[[cfg.id_col] + cfg.cov_cols_raw].copy()
    cov[cfg.id_col] = cov[cfg.id_col].astype(str)

    m = df.merge(early, on=cfg.id_col, how="inner").merge(cov, on=cfg.id_col, how="left")
    m["dataset"] = dataset_name

    # numeric covariates
    m["_age"] = pd.to_numeric(m["Age Years"], errors="coerce")
    m["_sex_male"] = m["Sex"].apply(coerce_sex)
    m["_edaravone"] = m["Taking Radicava Edaravone"].apply(coerce_yes_no)
    m["_riluzole"] = m["Taking Rilutek Riluzole"].apply(coerce_yes_no)

    cov_num_cols = ["_age", "_sex_male", "_edaravone", "_riluzole"]

    latent_cols = detect_latent_cols(m, cfg.id_col, cfg.group_col)
    return m, latent_cols


def compute_latent_clinical_correlations(
    m: pd.DataFrame,
    latent_cols: List[str],
    cfg: Config,
) -> pd.DataFrame:
    """
    For each method × adjusted:
      correlate every latent dim with each clinical outcome,
      BH-FDR across (len(latent_cols) * len(clinical_cols)) tests.
    """
    cov_num_cols = ["_age", "_sex_male", "_edaravone", "_riluzole"]
    rows = []

    for method in ("pearson", "spearman"):
        for adjusted in (False, True):
            pvals = []
            tmp = []

            for lc in latent_cols:
                x = pd.to_numeric(m[lc], errors="coerce").to_numpy()
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
                            "dataset": m["dataset"].iloc[0],
                            "method": method,
                            "adjusted": adjusted,
                            "latent_var": lc,
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


def als_mask(m: pd.DataFrame, cfg: Config) -> np.ndarray:
    if not cfg.group_col or cfg.group_col not in m.columns:
        # If no group column, just use all rows
        return np.ones(len(m), dtype=bool)
    return (m[cfg.group_col].astype(str).str.upper() == cfg.pos_group.upper()).to_numpy()


def pc_level_analysis(
    m: pd.DataFrame,
    latent_cols: List[str],
    cfg: Config,
    clinical_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PCA (unadjusted + covariate-adjusted) on ALS subset (if group exists).
    Then correlate PC1..PCn with outcomes (Pearson+Spearman),
    BH within-outcome across PCs and BH global across all (PC×outcome) per (adjustment×method).
    Also LOOCV regression: predict each outcome from first k PCs.
    """
    cov_num_cols = ["_age", "_sex_male", "_edaravone", "_riluzole"]

    m2 = m.copy()
    m2 = m2[als_mask(m2, cfg)].copy()

    # matrix X
    X = m2[latent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    C = m2[cov_num_cols].to_numpy(dtype=float)

    # outcomes
    Y = {name: pd.to_numeric(m2[col], errors="coerce").to_numpy(dtype=float) for name, col in clinical_map.items()}

    # Unadjusted PCA mask: complete X
    mask_X = np.all(np.isfinite(X), axis=1)
    X_u = X[mask_X]
    X_u_c = StandardScaler(with_mean=True, with_std=False).fit_transform(X_u)
    pca_u = PCA(n_components=min(cfg.n_pcs, X_u_c.shape[0], X_u_c.shape[1]))
    PCs_u = pca_u.fit_transform(X_u_c)
    var_u = pca_u.explained_variance_ratio_

    # Cov-adjusted PCA mask: complete X and C
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

    # Correlations
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
                    # record empties
                    for method in ("pearson", "spearman"):
                        for k in range(PCs.shape[1]):
                            corr_rows.append(
                                {
                                    "dataset": m["dataset"].iloc[0],
                                    "adjustment": adj_label,
                                    "method": method,
                                    "outcome": out_name,
                                    "PC": k + 1,
                                    "explained_var": float(var[k]) if k < len(var) else np.nan,
                                    "r": np.nan,
                                    "p": np.nan,
                                    "q_within_outcome": 1.0,
                                    "n": int(y_finite.sum()),
                                }
                            )
                    continue

                y_res = residualize(y_sub[y_finite], C_c[y_finite])

                for method in ("pearson", "spearman"):
                    pvals = []
                    tmp = []
                    for k in range(PCs.shape[1]):
                        pc = PCs[:, k][y_finite]
                        msk = np.isfinite(pc) & np.isfinite(y_res)
                        if msk.sum() < 5:
                            r = np.nan
                            p = np.nan
                            n = int(msk.sum())
                        else:
                            r, p = corr_vec(pc[msk], y_res[msk], method)
                            n = int(msk.sum())
                        tmp.append((k + 1, r, p, n))
                        pvals.append(p if np.isfinite(p) else 1.0)
                    q = bh_qvalues(np.array(pvals, dtype=float))
                    for (pc_idx, r, p, n), qi in zip(tmp, q):
                        corr_rows.append(
                            {
                                "dataset": m["dataset"].iloc[0],
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
            else:
                # unadjusted: correlate directly, handle missing y
                for method in ("pearson", "spearman"):
                    pvals = []
                    tmp = []
                    for k in range(PCs.shape[1]):
                        pc = PCs[:, k]
                        msk = np.isfinite(pc) & np.isfinite(y_sub)
                        if msk.sum() < 5:
                            r = np.nan
                            p = np.nan
                            n = int(msk.sum())
                        else:
                            r, p = corr_vec(pc[msk], y_sub[msk], method)
                            n = int(msk.sum())
                        tmp.append((k + 1, r, p, n))
                        pvals.append(p if np.isfinite(p) else 1.0)
                    q = bh_qvalues(np.array(pvals, dtype=float))
                    for (pc_idx, r, p, n), qi in zip(tmp, q):
                        corr_rows.append(
                            {
                                "dataset": m["dataset"].iloc[0],
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

    corr_df = pd.DataFrame(corr_rows)

    # Global q across all (PC×outcome) within each (adjustment×method) block
    corr_df["q_global"] = np.nan
    for (adj, method), idxs in corr_df.groupby(["adjustment", "method"]).groups.items():
        pvals = corr_df.loc[idxs, "p"].to_numpy(dtype=float)
        pvals = np.where(np.isfinite(pvals), pvals, 1.0)
        corr_df.loc[idxs, "q_global"] = bh_qvalues(pvals)

    # LOOCV regression: predict each outcome from first k PCs
    reg_rows = []
    for adj_label, PCs, var, mask, do_y_resid in [
        ("unadj", PCs_u, var_u, mask_X, False),
        ("cov_adj", PCs_c, var_c, mask_XC, True),
    ]:
        for out_name, y_all in Y.items():
            y_sub = y_all[mask]
            if do_y_resid:
                y_finite = np.isfinite(y_sub)
                if y_finite.sum() < 8:
                    continue
                yv = residualize(y_sub[y_finite], C_c[y_finite])
                PCv = PCs[y_finite]
            else:
                y_finite = np.isfinite(y_sub)
                if y_finite.sum() < 8:
                    continue
                yv = y_sub[y_finite]
                PCv = PCs[y_finite]

            n = len(yv)
            sst = ((yv - yv.mean()) ** 2).sum()
            loo = LeaveOneOut()

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
                        "dataset": m["dataset"].iloc[0],
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

    reg_df = pd.DataFrame(reg_rows)

    notes = pd.DataFrame(
        [
            {
                "dataset": m["dataset"].iloc[0],
                "n_rows_merged": int(len(m)),
                "n_subset_used_for_PCA": int(len(m2)),
                "n_complete_X": int(mask_X.sum()),
                "n_complete_XC": int(mask_XC.sum()),
                "PC1_var_unadj": float(var_u[0]) if len(var_u) else np.nan,
                "PC1_var_cov_adj": float(var_c[0]) if len(var_c) else np.nan,
            }
        ]
    )

    return notes, corr_df, reg_df


def save_pc_heatmaps(
    corr_df: pd.DataFrame,
    out_png: str,
    pcs: int = 5,
):
    """Create a simple heatmap panel: (unadj/cov_adj) × datasets for Pearson r, PC1..PC5."""
    datasets = corr_df["dataset"].unique().tolist()
    fig, axes = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 6), constrained_layout=True)

    for j, ds in enumerate(datasets):
        for i, adj in enumerate(["unadj", "cov_adj"]):
            ax = axes[i, j] if len(datasets) > 1 else axes[i]
            sub = corr_df[
                (corr_df["dataset"] == ds)
                & (corr_df["adjustment"] == adj)
                & (corr_df["method"] == "pearson")
                & (corr_df["PC"] <= pcs)
            ].copy()
            if sub.empty:
                ax.axis("off")
                continue
            mat = (
                sub.pivot_table(index="PC", columns="outcome", values="r", aggfunc="first")
                .reindex(index=list(range(1, pcs + 1)))
            )
            im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", vmin=-1, vmax=1)
            ax.set_title(f"{ds} • {adj} • Pearson r")
            ax.set_xticks(np.arange(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(pcs))
            ax.set_yticklabels([f"PC{k}" for k in range(1, pcs + 1)])

    fig.colorbar(im, ax=axes.ravel().tolist() if len(datasets) > 1 else axes.ravel().tolist(), shrink=0.7)
    fig.savefig(out_png, dpi=200)


# -----------------------------
# I/O + main
# -----------------------------
def parse_latent_arg(s: str) -> Tuple[str, str]:
    """
    Expect: name=path.csv
    Example: 0h=/path/ensemble_z_0h.csv
    """
    if "=" not in s:
        raise ValueError(f"--latent must be like name=path.csv, got: {s}")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", action="append", required=True, help="name=path.csv (repeatable)")
    ap.add_argument("--early_slope", required=True, help="early_slope.xlsx")
    ap.add_argument("--covariates", required=True, help="clinical_covariates.xlsx")
    ap.add_argument("--out_dir", required=True, help="output directory")

    ap.add_argument("--id_col", default="code")
    ap.add_argument("--group_col", default="group")
    ap.add_argument("--pos_group", default="ALS")

    # clinical columns (match your files)
    ap.add_argument("--nfl_col", default="Nfl Concentration Pg Per Ml")
    ap.add_argument("--alsfrs_total_col", default="Total Alsfrsr Score")
    ap.add_argument("--alsfrs_slope_col", default="Early Slope (pts/mo)")

    # covariates columns (match your files)
    ap.add_argument("--age_col", default="Age Years")
    ap.add_argument("--sex_col", default="Sex")
    ap.add_argument("--edaravone_col", default="Taking Radicava Edaravone")
    ap.add_argument("--riluzole_col", default="Taking Rilutek Riluzole")

    ap.add_argument("--n_pcs", type=int, default=10)
    ap.add_argument("--loocv_ks", default="1,2,3,5,10")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    latent_specs = [parse_latent_arg(s) for s in args.latent]

    early_df = pd.read_excel(args.early_slope)
    cov_df = pd.read_excel(args.covariates)

    # Ensure id_col exists in early/cov; if not, try rename from 'code'
    if args.id_col not in early_df.columns and "code" in early_df.columns:
        early_df = early_df.rename(columns={"code": args.id_col})
    if args.id_col not in cov_df.columns and "code" in cov_df.columns:
        cov_df = cov_df.rename(columns={"code": args.id_col})

    cfg = Config(
        id_col=args.id_col,
        group_col=args.group_col if args.group_col in pd.read_csv(latent_specs[0][1], nrows=1).columns else None,
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

    all_corr = []
    all_notes = []
    all_pc_corr = []
    all_pc_reg = []

    for name, path in latent_specs:
        latent_df = pd.read_csv(path)
        # If group_col missing in this dataset, drop it from cfg for this dataset
        cfg_local = cfg
        if cfg.group_col and cfg.group_col not in latent_df.columns:
            cfg_local = Config(**{**cfg.__dict__, "group_col": None})

        m, latent_cols = prepare_merged(latent_df, early_df, cov_df, cfg_local, name)

        corr_df = compute_latent_clinical_correlations(m, latent_cols, cfg_local)
        all_corr.append(corr_df)

        notes_pc, pc_corr_df, pc_reg_df = pc_level_analysis(m, latent_cols, cfg_local, clinical_map)
        all_notes.append(notes_pc)
        all_pc_corr.append(pc_corr_df)
        all_pc_reg.append(pc_reg_df)

        # Write per-dataset Excel
        out_xlsx = os.path.join(args.out_dir, f"{name}_corr_and_pca.xlsx")
        out_csv = os.path.join(args.out_dir, f"{name}_latent_clinical_corr.csv")
        out_png = os.path.join(args.out_dir, f"{name}_pc_heatmap.png")

        corr_df.to_csv(out_csv, index=False)

        top_corr = corr_df[(corr_df["q"] <= 0.1) & corr_df["p"].notna()].sort_values(["q", "p"]).head(200)
        top_pc = pc_corr_df[(pc_corr_df["q_global"] <= 0.1) & pc_corr_df["p"].notna()].sort_values(["q_global", "p"]).head(200)

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            # correlation outputs
            corr_df.to_excel(writer, sheet_name="latent_corr_all", index=False)
            top_corr.to_excel(writer, sheet_name="latent_corr_top_q0p1", index=False)
            for method in ["pearson", "spearman"]:
                for adjusted in [False, True]:
                    sheet = f"{method}_{'adj' if adjusted else 'unadj'}"
                    sub = corr_df[(corr_df.method == method) & (corr_df.adjusted == adjusted)]
                    sub.to_excel(writer, sheet_name=sheet[:31], index=False)

            # PCA + PC correlations
            notes_pc.to_excel(writer, sheet_name="pca_notes", index=False)
            pc_corr_df.to_excel(writer, sheet_name="pc_correlations", index=False)
            top_pc.to_excel(writer, sheet_name="pc_top_q_global0p1", index=False)
            pc_reg_df.to_excel(writer, sheet_name="pc_loocv_regression", index=False)

        # Heatmap for this dataset only
        try:
            save_pc_heatmaps(pc_corr_df, out_png, pcs=min(5, cfg_local.n_pcs))
        except Exception as e:
            print(f"[warn] heatmap failed for {name}: {e}")

        print(f"[ok] wrote {out_xlsx}")
        print(f"[ok] wrote {out_csv}")
        if os.path.exists(out_png):
            print(f"[ok] wrote {out_png}")

    # Combined summary (optional, useful when multiple datasets are provided)
    all_corr_df = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()
    all_notes_df = pd.concat(all_notes, ignore_index=True) if all_notes else pd.DataFrame()
    all_pc_corr_df = pd.concat(all_pc_corr, ignore_index=True) if all_pc_corr else pd.DataFrame()
    all_pc_reg_df = pd.concat(all_pc_reg, ignore_index=True) if all_pc_reg else pd.DataFrame()

    combined_xlsx = os.path.join(args.out_dir, "ALL_datasets_corr_and_pca_combined.xlsx")
    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        all_notes_df.to_excel(writer, sheet_name="pca_notes_all", index=False)
        all_corr_df.to_excel(writer, sheet_name="latent_corr_all", index=False)
        all_pc_corr_df.to_excel(writer, sheet_name="pc_correlations_all", index=False)
        all_pc_reg_df.to_excel(writer, sheet_name="pc_loocv_all", index=False)

    print(f"[ok] wrote {combined_xlsx}")


if __name__ == "__main__":
    main()
