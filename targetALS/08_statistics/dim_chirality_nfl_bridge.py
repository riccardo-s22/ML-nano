#!/usr/bin/env python3
"""
Bridge analysis between latent dimensions, chirality descriptors, and NFL.

What it does
------------
1. Merges z_agg3.csv, chirality_interp_descriptors.csv, and early_slope.xlsx on `code`
2. Computes raw Spearman/Pearson correlation between target dim(s) and all chirality descriptors
3. Computes partial correlation between target dim(s) and descriptors controlling for NFL
4. Fits NFL regression models:
      NFL ~ dim
      NFL ~ descriptor
      NFL ~ dim + descriptor
5. Writes ranked CSV tables
6. Makes scatterplots for top descriptors, with points colored by NFL

Designed for Riccardo's CNT ALS analysis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


DEFAULT_BASE = Path(r"C:\Users\riccardo-s\Documents\CNT\targetALS\PLS")
DEFAULT_Z = "z_agg3.csv"
DEFAULT_CHIR = "chirality_interp_descriptors.csv"
DEFAULT_NFL = "early_slope.xlsx"
DEFAULT_OUTDIR = "dim_chirality_nfl_bridge_results"
NFL_COL_CANDIDATES = [
    "Nfl Concentration Pg Per Ml",
    "nfl_conc",
    "nfl",
    "NFL",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relate latent dims, chirality descriptors, and NFL")
    p.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE))
    p.add_argument("--z_file", type=str, default=DEFAULT_Z)
    p.add_argument("--chir_file", type=str, default=DEFAULT_CHIR)
    p.add_argument("--nfl_file", type=str, default=DEFAULT_NFL)
    p.add_argument("--target_dims", type=str, required=True,
                   help="Comma-separated latent dim indices or names, e.g. 477,350 or dim477,dim350")
    p.add_argument("--top_n", type=int, default=12,
                   help="How many top chirality descriptors to export plots for, per dimension")
    p.add_argument("--min_n", type=int, default=6,
                   help="Minimum complete samples required for a descriptor to be tested")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    return p.parse_args()


def resolve_target_dims(raw: str, z_cols: Iterable[str]) -> List[str]:
    z_cols = set(z_cols)
    dims = []
    for tok in [x.strip() for x in raw.split(",") if x.strip()]:
        if tok in z_cols:
            dims.append(tok)
        elif tok.isdigit() and f"dim{tok}" in z_cols:
            dims.append(f"dim{tok}")
        else:
            raise ValueError(f"Target dimension '{tok}' not found in z_agg columns.")
    if not dims:
        raise ValueError("No valid target dims provided.")
    return dims


def find_nfl_col(df: pd.DataFrame) -> str:
    for col in NFL_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find NFL column. Tried: {NFL_COL_CANDIDATES}")


def rank_partial_corr(x: pd.Series, y: pd.Series, z: pd.Series) -> Tuple[float, float, int]:
    tmp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce"), "z": pd.to_numeric(z, errors="coerce")}).dropna()
    n = len(tmp)
    if n < 4:
        return np.nan, np.nan, n

    xr = tmp["x"].rank()
    yr = tmp["y"].rank()
    zr = tmp["z"].rank()

    Xz = sm.add_constant(zr)
    rx = sm.OLS(xr, Xz).fit().resid
    ry = sm.OLS(yr, Xz).fit().resid
    r, p = pearsonr(rx, ry)
    return float(r), float(p), int(n)


def fit_r2(y: pd.Series, X: pd.DataFrame):
    y_num = pd.to_numeric(y, errors="coerce")
    X_num = X.apply(pd.to_numeric, errors="coerce")
    tmp = pd.concat([y_num.rename("y"), X_num], axis=1).dropna()
    if tmp.shape[0] < 3 or tmp.iloc[:, 1:].shape[1] == 0:
        return None
    Xc = sm.add_constant(tmp.iloc[:, 1:], has_constant="add")
    if Xc.shape[0] < 3:
        return None
    return sm.OLS(tmp.iloc[:, 0], Xc).fit()


def safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> Tuple[float, float, int]:
    tmp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(tmp)
    if n < 3:
        return np.nan, np.nan, n
    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan, n
    if method == "spearman":
        r, p = spearmanr(tmp["x"], tmp["y"])
    else:
        r, p = pearsonr(tmp["x"], tmp["y"])
    return float(r), float(p), int(n)


def scatterplot(df: pd.DataFrame, xcol: str, ycol: str, colorcol: str, title: str, outfile: Path) -> None:
    d = df[[xcol, ycol, colorcol, "group"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d[colorcol] = pd.to_numeric(d[colorcol], errors="coerce")
    d = d.dropna().copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(d[xcol], d[ycol], c=d[colorcol], s=60, edgecolor="black", linewidth=0.4)

    # regression line for display only
    if d[xcol].nunique() > 1 and d[ycol].nunique() > 1:
        m, b = np.polyfit(d[xcol].values, d[ycol].values, 1)
        xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
        ax.plot(xs, m * xs + b, linewidth=1.2)

    for _, row in d.iterrows():
        grp = str(row["group"]) if pd.notna(row["group"]) else ""
        if grp.upper() == "ALS":
            ax.scatter([], [], label="ALS")
            break
    # dedupe legend entries if present later
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(colorcol)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary_plot(top_df: pd.DataFrame, dim_name: str, out_png: Path) -> None:
    if top_df.empty:
        return
    plot_df = top_df.head(10).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.42 * len(plot_df) + 1.3)))
    ax.barh(plot_df["feature"], plot_df["abs_rho_raw"])
    ax.set_xlabel("|Spearman rho| with {}".format(dim_name))
    ax.set_title(f"Top chirality descriptors linked to {dim_name}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    z_path = base / args.z_file
    chir_path = base / args.chir_file
    nfl_path = base / args.nfl_file
    outdir = base / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    z = pd.read_csv(z_path)
    chir = pd.read_csv(chir_path)
    nfl = pd.read_excel(nfl_path)

    # normalize sample keys because z/nfl use dots while chirality file uses underscores
    for df in (z, chir, nfl):
        df["code"] = df["code"].astype(str)
        df["code_clean"] = df["code"].str.replace(".", "_", regex=False)

    target_dims = resolve_target_dims(args.target_dims, z.columns)
    nfl_col = find_nfl_col(nfl)

    keep_nfl_cols = ["code", nfl_col]
    if "Patient ID" in nfl.columns:
        keep_nfl_cols.append("Patient ID")
    if "Early Slope (pts/mo)" in nfl.columns:
        keep_nfl_cols.append("Early Slope (pts/mo)")

    keep_nfl_cols.append("code_clean")
    nfl = nfl[keep_nfl_cols].copy().rename(columns={nfl_col: "nfl"})

    merged = z.merge(chir.drop(columns=["code"]), on="code_clean", how="inner").merge(
        nfl.drop(columns=["code"]), on="code_clean", how="inner"
    )
    if "group" not in merged.columns:
        merged["group"] = np.nan

    chir_cols = [c for c in chir.columns if c != "code"]

    run_log = []
    for dim_name in target_dims:
        dim_dir = outdir / dim_name
        dim_dir.mkdir(exist_ok=True)

        rows = []
        for feat in chir_cols:
            sub = merged[[dim_name, feat, "nfl"]].dropna()
            if len(sub) < args.min_n:
                continue
            rho_raw, p_raw, n1 = safe_corr(sub[dim_name], sub[feat], method="spearman")
            pear_r, pear_p, _ = safe_corr(sub[dim_name], sub[feat], method="pearson")
            rho_dim_nfl, p_dim_nfl, _ = safe_corr(sub[dim_name], sub["nfl"], method="spearman")
            rho_feat_nfl, p_feat_nfl, _ = safe_corr(sub[feat], sub["nfl"], method="spearman")
            rho_partial, p_partial, n2 = rank_partial_corr(sub[dim_name], sub[feat], sub["nfl"])

            m1 = fit_r2(sub["nfl"], sub[[dim_name]])
            m2 = fit_r2(sub["nfl"], sub[[feat]])
            m3 = fit_r2(sub["nfl"], sub[[dim_name, feat]])

            rows.append({
                "feature": feat,
                "n": n1,
                "rho_raw": rho_raw,
                "p_raw": p_raw,
                "pearson_r": pear_r,
                "pearson_p": pear_p,
                "rho_dim_vs_nfl": rho_dim_nfl,
                "p_dim_vs_nfl": p_dim_nfl,
                "rho_feature_vs_nfl": rho_feat_nfl,
                "p_feature_vs_nfl": p_feat_nfl,
                "rho_partial_given_nfl": rho_partial,
                "p_partial": p_partial,
                "R2_nfl_dim_only": float(m1.rsquared) if m1 is not None else np.nan,
                "R2_nfl_feature_only": float(m2.rsquared) if m2 is not None else np.nan,
                "R2_nfl_both": float(m3.rsquared) if m3 is not None else np.nan,
                "beta_dim_in_joint": float(m3.params.get(dim_name, np.nan)) if m3 is not None else np.nan,
                "p_dim_in_joint": float(m3.pvalues.get(dim_name, np.nan)) if m3 is not None else np.nan,
                "beta_feature_in_joint": float(m3.params.get(feat, np.nan)) if m3 is not None else np.nan,
                "p_feature_in_joint": float(m3.pvalues.get(feat, np.nan)) if m3 is not None else np.nan,
                "delta_R2_over_dim": float(m3.rsquared - m1.rsquared) if (m1 is not None and m3 is not None) else np.nan,
                "delta_R2_over_feature": float(m3.rsquared - m2.rsquared) if (m2 is not None and m3 is not None) else np.nan,
            })

        res = pd.DataFrame(rows)
        if res.empty:
            run_log.append({"dim": dim_name, "tested_features": 0})
            continue

        res["p_raw_fdr"] = multipletests(res["p_raw"].fillna(1.0), method="fdr_bh")[1]
        res["p_partial_fdr"] = multipletests(res["p_partial"].fillna(1.0), method="fdr_bh")[1]
        res["abs_rho_raw"] = res["rho_raw"].abs()
        res["abs_rho_partial"] = res["rho_partial_given_nfl"].abs()
        res = res.sort_values(["p_raw_fdr", "abs_rho_raw"], ascending=[True, False]).reset_index(drop=True)
        res.to_csv(dim_dir / f"{dim_name}_chirality_vs_nfl_bridge.csv", index=False)

        top_df = res.head(args.top_n).copy()
        top_df.to_csv(dim_dir / f"{dim_name}_top{args.top_n}_features.csv", index=False)
        summary_plot(top_df, dim_name, dim_dir / f"{dim_name}_top_features_barplot.png")

        # Make scatterplots for the top descriptors
        for i, feat in enumerate(top_df["feature"], start=1):
            title = f"{dim_name} vs {feat}\ncolored by NFL"
            scatterplot(merged, dim_name, feat, "nfl", title, dim_dir / f"{i:02d}_{dim_name}_vs_{feat}.png")

        # Per-dim text summary
        top1 = top_df.iloc[0]
        run_log.append({
            "dim": dim_name,
            "tested_features": int(len(res)),
            "best_feature": str(top1["feature"]),
            "best_rho_raw": float(top1["rho_raw"]),
            "best_p_raw_fdr": float(top1["p_raw_fdr"]),
            "best_rho_partial": float(top1["rho_partial_given_nfl"]),
            "best_p_partial_fdr": float(top1["p_partial_fdr"]),
            "best_delta_R2_over_dim": float(top1["delta_R2_over_dim"]),
        })

    run_log_df = pd.DataFrame(run_log)
    run_log_df.to_csv(outdir / "run_summary.csv", index=False)

    with open(outdir / "README_results.txt", "w", encoding="utf-8") as fh:
        fh.write("Results generated by dim_chirality_nfl_bridge.py\n\n")
        fh.write("Main table per dimension:\n")
        fh.write("  <dim>/<dim>_chirality_vs_nfl_bridge.csv\n")
        fh.write("Columns include raw dim-descriptor correlation, partial correlation controlling for NFL,\n")
        fh.write("and NFL regression R^2 values for dim only, descriptor only, and both together.\n\n")
        fh.write("Interpretation guide:\n")
        fh.write("- raw significant + partial weak: mostly shared NFL dependence\n")
        fh.write("- raw significant + partial significant: shared structure beyond NFL\n")
        fh.write("- positive delta_R2_over_dim: descriptor adds NFL information beyond the dim\n")

    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
