#!/usr/bin/env python3
"""
interdependence_pipeline_xlsx.py

Matched 0h/6h/24h Excel pipeline:
- base_dir contains subfolders: 0h, 6h, 24h
- filenames across folders are intersected (matched files)
- each .xlsx is loaded (default: first sheet) into a numeric 2D array
- computes per-file:
    * Pearson correlation (Patch Interdependence)
    * Spearman correlation (Trace Spearman)
    * Partial correlation (Trace Partial Corr) via Ledoit–Wolf covariance + precision
- aggregates MEAN + SD across matched files
- computes paired deltas vs 0h (6h-0h and 24h-0h) for Pearson:
    * delta mean + SD
    * bootstrap CI-excludes-zero mask for mean delta
    * BH-FDR significant-edge mask from paired tests on deltas
- writes PNGs to outdir

Usage (Windows PowerShell example):
python interdependence_pipeline_xlsx.py `
  --base_dir "C:\\Users\\riccardo-s\\Documents\\CNT\\targetALS\\exp5" `
  --outdir "C:\\Users\\riccardo-s\\Documents\\CNT\\targetALS\\exp5\\outputs" `
  --labels "6.5,7.5,7.6,8.3,8.4,8.6,8.7,9.4,9.5,10.2,10.3,10.5"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata, wilcoxon, ttest_1samp
from sklearn.covariance import LedoitWolf


SUPPORTED_EXTS = (".xlsx", ".xls")


# -------------------------
# File matching
# -------------------------

def list_files(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("~$"):
            out[p.name] = p
    return out


def intersect_filenames(d0: Path, d6: Path, d24: Path) -> List[str]:
    f0 = set(list_files(d0).keys())
    f6 = set(list_files(d6).keys())
    f24 = set(list_files(d24).keys())
    common = sorted(list(f0 & f6 & f24))
    if len(common) == 0:
        raise RuntimeError("No matched .xlsx filenames across 0h/6h/24h.")
    return common


# -------------------------
# Excel loading (robust numeric extraction)
# -------------------------

def _clean_numeric_df(df: pd.DataFrame) -> np.ndarray:
    """
    Convert an Excel sheet to a numeric 2D array.
    - Coerces all cells to numeric (non-numeric -> NaN)
    - Drops all-NaN rows/cols
    This works when the sheet contains headers/index labels in the first row/col.
    """
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
    arr = num.to_numpy(dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("After numeric coercion, no usable numeric data remained.")
    return arr


def load_xlsx_as_array(path: Path, sheet: Optional[str] = None) -> np.ndarray:
    """
    Load .xlsx into a 2D numeric numpy array.
    - If sheet is None: uses first sheet.
    - Robust to non-numeric headers/indices; drops them via coercion + NaN pruning.
    """
    if sheet is None or sheet.strip() == "":
        df = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    else:
        df = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    return _clean_numeric_df(df)


# -------------------------
# Metrics
# -------------------------

def _ensure_trace_shape(x: np.ndarray) -> np.ndarray:
    """
    If square: treat as matrix already (P,P).
    Else assume trace-like (T,P). If it looks like (P,T) with small first dim, transpose.
    """
    if x.shape[0] == x.shape[1]:
        return x
    if x.shape[0] <= 50 and x.shape[1] > x.shape[0]:
        return x.T
    return x


def pearson_corr_from_trace(x: np.ndarray) -> np.ndarray:
    x = _ensure_trace_shape(x)
    if x.shape[0] == x.shape[1]:
        return np.asarray(x, dtype=float)
    return np.corrcoef(x, rowvar=False)


def spearman_corr_from_trace(x: np.ndarray) -> np.ndarray:
    x = _ensure_trace_shape(x)
    if x.shape[0] == x.shape[1]:
        return np.asarray(x, dtype=float)
    xr = np.apply_along_axis(rankdata, 0, x)
    return np.corrcoef(xr, rowvar=False)


def partial_corr_from_trace(x: np.ndarray) -> np.ndarray:
    x = _ensure_trace_shape(x)
    if x.shape[0] == x.shape[1]:
        # If already square, treat as covariance-ish; this is a fallback.
        cov = np.asarray(x, dtype=float)
    else:
        lw = LedoitWolf().fit(x)
        cov = lw.covariance_

    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return np.clip(pcorr, -1.0, 1.0)


# -------------------------
# Aggregation + inference
# -------------------------

def mean_sd_stack(mats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.stack(mats, axis=0)
    mean = np.mean(stack, axis=0)
    sd = np.std(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    return mean, sd


def bootstrap_ci_excludes_zero(
    deltas: List[np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0
) -> np.ndarray:
    """Bootstrap CI for mean(delta); returns mask where CI excludes 0."""
    rng = np.random.default_rng(seed)
    stack = np.stack(deltas, axis=0)  # (N,P,P)
    n = stack.shape[0]
    boot_means = np.empty((n_boot,) + stack.shape[1:], dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(stack[idx], axis=0)
    lo = np.quantile(boot_means, alpha / 2.0, axis=0)
    hi = np.quantile(boot_means, 1.0 - alpha / 2.0, axis=0)
    mask = (lo > 0) | (hi < 0)
    np.fill_diagonal(mask, False)
    return mask.astype(float)


def bh_fdr(pvals_1d: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg rejection mask for 1D p-values."""
    p = np.asarray(pvals_1d, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    rej = np.zeros_like(p, dtype=bool)
    if np.any(below):
        kmax = int(np.max(np.where(below)[0]))
        cutoff = ranked[kmax]
        rej = p <= cutoff
    out = np.zeros_like(rej)
    out[order] = rej[order]
    return out


def significant_edges_fdr(
    deltas: List[np.ndarray],
    alpha: float = 0.05,
    test: str = "wilcoxon"
) -> np.ndarray:
    """
    Per-edge paired testing on deltas + BH-FDR; returns (P,P) mask.
    Tests whether mean/median(delta) differs from 0 across matched files.
    """
    stack = np.stack(deltas, axis=0)
    n, p, _ = stack.shape
    iu = np.triu_indices(p, k=1)

    pvals = np.ones(len(iu[0]), dtype=float)
    for k, (i, j) in enumerate(zip(iu[0], iu[1])):
        x = stack[:, i, j]
        if np.allclose(x, 0):
            pvals[k] = 1.0
            continue

        if test == "ttest":
            # one-sample test of deltas vs 0
            _, pv = ttest_1samp(x, popmean=0.0, nan_policy="omit")
        else:
            # nonparametric signed-rank vs 0
            try:
                _, pv = wilcoxon(
                    x, np.zeros_like(x),
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    mode="auto",
                )
            except ValueError:
                pv = 1.0

        pvals[k] = pv if np.isfinite(pv) else 1.0

    rej = bh_fdr(pvals, alpha=alpha)
    mask = np.zeros((p, p), dtype=float)
    mask[iu] = rej.astype(float)
    mask[(iu[1], iu[0])] = rej.astype(float)
    np.fill_diagonal(mask, 0.0)
    return mask


# -------------------------
# Plotting
# -------------------------

def _parse_labels(labels_csv: Optional[str], p: int) -> List[str]:
    if labels_csv is None or labels_csv.strip() == "":
        return [str(i) for i in range(p)]
    labels = [s.strip() for s in labels_csv.split(",") if s.strip() != ""]
    if len(labels) != p:
        raise ValueError(f"--labels has {len(labels)} entries but matrices are size {p}.")
    return labels


def symmetric_limits(mat: np.ndarray, pct: float = 99.0) -> float:
    vals = np.abs(mat[np.isfinite(mat)])
    return float(np.percentile(vals, pct)) if vals.size else 1.0


def plot_heatmap(
    mat: np.ndarray,
    labels: List[str],
    title: str,
    outpath: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    annotate: bool = False,
    fmt: str = "{:.2f}",
) -> None:
    p = mat.shape[0]
    fig, ax = plt.subplots(figsize=(10.5, 8), dpi=150)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_title(title)
    ax.set_xticks(np.arange(p))
    ax.set_yticks(np.arange(p))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if annotate:
        for i in range(p):
            for j in range(p):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def compute_all_matrices_for_file(path: Path, sheet: Optional[str]) -> Dict[str, np.ndarray]:
    x = load_xlsx_as_array(path, sheet=sheet)
    return {
        "pearson": pearson_corr_from_trace(x),
        "spearman": spearman_corr_from_trace(x),
        "partialcorr": partial_corr_from_trace(x),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help=r'Base folder containing subfolders "out_0h", "out_6h", "out_24h".')
    ap.add_argument("--outdir", required=True, help="Output folder for PNGs")
    ap.add_argument("--labels", default=None, help="Comma-separated axis labels (length P)")
    ap.add_argument("--sheet", default=None, help="Excel sheet name to read (default: first sheet)")
    ap.add_argument("--n-boot", default=2000, type=int, help="Bootstrap iterations for CI masks")
    ap.add_argument("--alpha", default=0.05, type=float, help="Alpha for CI + FDR")
    ap.add_argument("--test", default="wilcoxon", choices=["wilcoxon", "ttest"], help="Delta-edge test")
    ap.add_argument("--seed", default=0, type=int, help="Seed for bootstrap")
    args = ap.parse_args()

    base = Path(args.base_dir)
    d0 = base / "out_0h"
    d6 = base / "out_6h"
    d24 = base / "out_24h"
    outdir = Path(args.outdir)

    if not d0.exists() or not d6.exists() or not d24.exists():
        raise FileNotFoundError(f'Expected subfolders "out_0h", "out_6h", "out_24h" under: {base}')

    files0, files6, files24 = list_files(d0), list_files(d6), list_files(d24)
    common = intersect_filenames(d0, d6, d24)

    per_file: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for fn in common:
        per_file[fn] = {
            "out_0h": compute_all_matrices_for_file(files0[fn], sheet=args.sheet),
            "out_6h": compute_all_matrices_for_file(files6[fn], sheet=args.sheet),
            "out_24h": compute_all_matrices_for_file(files24[fn], sheet=args.sheet),
        }

    p = per_file[common[0]]["out_0h"]["pearson"].shape[0]
    labels = _parse_labels(args.labels, p)

    # Patch Interdependence (Pearson): mean + SD for 0/6/24
    for time in ("out_0h", "out_6h", "out_24h"):
        mats = [per_file[fn][time]["pearson"] for fn in common]
        mean, sd = mean_sd_stack(mats)
        np.fill_diagonal(sd, 0.0)

        plot_heatmap(
            mean, labels,
            title=f"Patch Interdependence @ {time.replace('h','')}h (MEAN, pearson)",
            outpath=outdir / f"patch_interdependence_{time}_MEAN_pearson_5x5.png",
            vmin=-1.0, vmax=1.0,
        )
        plot_heatmap(
            sd, labels,
            title=f"Patch Interdependence @ {time.replace('h','')}h (SD across matched files)",
            outpath=outdir / f"patch_interdependence_{time}_SD_pearson_5x5.png",
            vmin=0.0, vmax=float(np.nanmax(sd)) if np.isfinite(sd).any() else 1.0,
        )

    # Deltas vs 0h (Pearson): 6h-0h and 24h-0h
    for time, tag in (("out_6h", "6h_minus_0h"), ("out_24h", "24h_minus_0h")):
        deltas = [per_file[fn][time]["pearson"] - per_file[fn]["out_0h"]["pearson"] for fn in common]
        dmean, dsd = mean_sd_stack(deltas)
        np.fill_diagonal(dsd, 0.0)

        lim = symmetric_limits(dmean, pct=99.0)
        plot_heatmap(
            dmean, labels,
            title=f"Change in Patch Overlap ({tag}, mean)",
            outpath=outdir / f"delta_{tag}_MEAN_pearson_5x5.png",
            vmin=-lim, vmax=lim, annotate=True, fmt="{:.2f}",
        )
        plot_heatmap(
            dsd, labels,
            title=f"Change in Patch Overlap ({tag}, SD across matched files)",
            outpath=outdir / f"delta_{tag}_SD_pearson_5x5.png",
            vmin=0.0, vmax=float(np.nanmax(dsd)) if np.isfinite(dsd).any() else 1.0,
        )

        ci_mask = bootstrap_ci_excludes_zero(deltas, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
        plot_heatmap(
            ci_mask, labels,
            title=f"Delta edges with CI excluding 0 ({tag})",
            outpath=outdir / f"delta_{tag}_CI_excludes_zero_mask.png",
            vmin=0.0, vmax=1.0, cmap="gray",
        )

        fdr_mask = significant_edges_fdr(deltas, alpha=args.alpha, test=args.test)
        plot_heatmap(
            fdr_mask, labels,
            title=f"Significant delta edges ({tag}) [FDR<={args.alpha:.2f}]",
            outpath=outdir / f"delta_{tag}_significant_edges_mask_FDR.png",
            vmin=0.0, vmax=1.0, cmap="gray",
        )

    # Trace Spearman: mean + SD for 0/6/24
    for time in ("out_0h", "out_6h", "out_24h"):
        mats = [per_file[fn][time]["spearman"] for fn in common]
        mean, sd = mean_sd_stack(mats)
        np.fill_diagonal(sd, 0.0)

        plot_heatmap(
            mean, labels,
            title=f"Trace Spearman @ {time.replace('h','')}h (MEAN)",
            outpath=outdir / f"trace_spearman_{time}_MEAN.png",
            vmin=-1.0, vmax=1.0,
        )
        plot_heatmap(
            sd, labels,
            title=f"Trace Spearman @ {time.replace('h','')}h (SD)",
            outpath=outdir / f"trace_spearman_{time}_SD.png",
            vmin=0.0, vmax=float(np.nanmax(sd)) if np.isfinite(sd).any() else 1.0,
        )

    # Trace Partial Corr: mean + SD for 0/6/24
    for time in ("out_0h", "out_6h", "out_24h"):
        mats = [per_file[fn][time]["partialcorr"] for fn in common]
        mean, sd = mean_sd_stack(mats)
        np.fill_diagonal(sd, 0.0)

        plot_heatmap(
            mean, labels,
            title=f"Trace Partial Corr @ {time.replace('h','')}h (MEAN)",
            outpath=outdir / f"trace_partialcorr_{time}_MEAN.png",
            vmin=-1.0, vmax=1.0,
        )
        plot_heatmap(
            sd, labels,
            title=f"Trace Partial Corr @ {time.replace('h','')}h (SD)",
            outpath=outdir / f"trace_partialcorr_{time}_SD.png",
            vmin=0.0, vmax=float(np.nanmax(sd)) if np.isfinite(sd).any() else 1.0,
        )

    print(f"Done. Matched files: {len(common)}")
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
