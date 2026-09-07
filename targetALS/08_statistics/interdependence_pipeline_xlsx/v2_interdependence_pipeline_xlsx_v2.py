#!/usr/bin/env python3
"""
interdependence_pipeline_xlsx_v2.py

Matched 0h/6h/24h Excel pipeline with optional grouping (e.g., 72 -> 12 labels).

Key features (vs v1):
- Accepts either subfolders: out_0h/out_6h/out_24h OR 0h/6h/24h (auto-detect).
- If --labels length != P, but P is a multiple of len(labels), automatically groups to len(labels):
    * If input sheet is trace-like (T,P): groups columns then computes correlations.
    * If input sheet is already square (P,P): block-averages the matrix to (L,L).

Usage (Windows CMD example):
python interdependence_pipeline_xlsx_v2.py ^
  --base_dir "C:\\Users\\riccardo-s\\Documents\\CNT\\targetALS\\exp5" ^
  --outdir   "C:\\Users\\riccardo-s\\Documents\\CNT\\targetALS\\exp5\\outputs" ^
  --labels   "6.5,7.5,7.6,8.3,8.4,8.6,8.7,9.4,9.5,10.2,10.3,10.5"

If your data are 72-wide and you pass 12 labels, the script will auto-group with group_size=6.
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


def detect_time_folders(base: Path) -> Tuple[Path, Path, Path]:
    """
    Supports either:
      - out_0h/out_6h/out_24h
      - 0h/6h/24h
    """
    candidates = [
        (base / "out_0h", base / "out_6h", base / "out_24h"),
        (base / "0h", base / "6h", base / "24h"),
    ]
    for d0, d6, d24 in candidates:
        if d0.exists() and d6.exists() and d24.exists():
            return d0, d6, d24
    raise FileNotFoundError(
        f'Expected subfolders "out_0h/out_6h/out_24h" OR "0h/6h/24h" under: {base}'
    )


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
# Grouping utilities
# -------------------------

def _infer_grouping(p: int, labels: List[str]) -> Tuple[bool, int]:
    """
    Returns (do_group, group_size). If labels length == p, do_group=False.
    If p is a multiple of len(labels), do_group=True and group_size=p//len(labels).
    """
    L = len(labels)
    if L == p:
        return False, 1
    if p % L != 0:
        raise ValueError(
            f"--labels has {L} entries but matrices are size {p} (not divisible). "
            f"Provide {p} labels or a divisor of {p}."
        )
    return True, p // L


def group_trace_columns(x: np.ndarray, n_groups: int, group_size: int, how: str = "mean") -> np.ndarray:
    """
    x is trace-like (T,P). Groups columns into n_groups with contiguous blocks of size group_size.
    """
    if x.ndim != 2:
        raise ValueError("Expected 2D array for trace grouping.")
    T, P = x.shape
    if P != n_groups * group_size:
        raise ValueError(f"Trace width P={P} does not match n_groups*group_size={n_groups*group_size}.")
    y = x.reshape(T, n_groups, group_size)
    if how == "median":
        return np.nanmedian(y, axis=2)
    return np.nanmean(y, axis=2)


def block_reduce_square(mat: np.ndarray, n_groups: int, group_size: int, how: str = "mean") -> np.ndarray:
    """
    mat is square (P,P). Block-reduces to (n_groups,n_groups) by averaging each group_size x group_size block.
    """
    P = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("block_reduce_square expects a square matrix.")
    if P != n_groups * group_size:
        raise ValueError(f"Matrix size P={P} does not match n_groups*group_size={n_groups*group_size}.")
    x = mat.reshape(n_groups, group_size, n_groups, group_size)
    if how == "median":
        out = np.nanmedian(np.nanmedian(x, axis=3), axis=1)
    else:
        out = np.nanmean(np.nanmean(x, axis=3), axis=1)
    return out


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
            _, pv = ttest_1samp(x, popmean=0.0, nan_policy="omit")
        else:
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

def parse_labels(labels_csv: Optional[str]) -> List[str]:
    if labels_csv is None or labels_csv.strip() == "":
        return []
    return [s.strip() for s in labels_csv.split(",") if s.strip() != ""]


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


def _pretty_time_label(folder: Path) -> str:
    name = folder.name
    if name.startswith("out_"):
        name = name[len("out_"):]
    return name


# -------------------------
# Main computation
# -------------------------

def compute_all_matrices_for_file(
    path: Path,
    sheet: Optional[str],
    labels: List[str],
    group_mode: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Loads the sheet, optionally groups to len(labels), then computes:
      - pearson, spearman, partialcorr
    """
    x = load_xlsx_as_array(path, sheet=sheet)
    x = _ensure_trace_shape(x)

    # Determine P before correlation (trace width or square size)
    if x.shape[0] == x.shape[1]:
        P = x.shape[0]
    else:
        P = x.shape[1]

    # If user didn't provide labels, default to P indices (no grouping).
    if not labels:
        labels_local = [str(i) for i in range(P)]
        do_group = False
        group_size = 1
        n_groups = P
    else:
        labels_local = labels
        n_groups = len(labels_local)
        do_group, group_size = _infer_grouping(P, labels_local)

    if do_group:
        if x.shape[0] == x.shape[1]:
            x = block_reduce_square(x, n_groups=n_groups, group_size=group_size, how=group_mode)
        else:
            x = group_trace_columns(x, n_groups=n_groups, group_size=group_size, how=group_mode)

    pear = pearson_corr_from_trace(x)
    spear = spearman_corr_from_trace(x)
    pcorr = partial_corr_from_trace(x)

    # After grouping, enforce diagonals for correlation-type outputs
    for m in (pear, spear, pcorr):
        np.fill_diagonal(m, 1.0)

    return {"pearson": pear, "spearman": spear, "partialcorr": pcorr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help=r"Base folder containing subfolders for 0h/6h/24h.")
    ap.add_argument("--outdir", required=True, help="Output folder for PNGs")
    ap.add_argument("--labels", default=None, help="Comma-separated axis labels (length P or a divisor of P)")
    ap.add_argument("--sheet", default=None, help="Excel sheet name to read (default: first sheet)")
    ap.add_argument("--group-mode", default="mean", choices=["mean", "median"], help="How to group within-label values")
    ap.add_argument("--n-boot", default=2000, type=int, help="Bootstrap iterations for CI masks")
    ap.add_argument("--alpha", default=0.05, type=float, help="Alpha for CI + FDR")
    ap.add_argument("--test", default="wilcoxon", choices=["wilcoxon", "ttest"], help="Delta-edge test")
    ap.add_argument("--seed", default=0, type=int, help="Seed for bootstrap")
    args = ap.parse_args()

    base = Path(args.base_dir)
    outdir = Path(args.outdir)
    labels = parse_labels(args.labels)

    d0, d6, d24 = detect_time_folders(base)

    files0, files6, files24 = list_files(d0), list_files(d6), list_files(d24)
    common = intersect_filenames(d0, d6, d24)

    per_file: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for fn in common:
        per_file[fn] = {
            d0.name: compute_all_matrices_for_file(files0[fn], sheet=args.sheet, labels=labels, group_mode=args.group_mode),
            d6.name: compute_all_matrices_for_file(files6[fn], sheet=args.sheet, labels=labels, group_mode=args.group_mode),
            d24.name: compute_all_matrices_for_file(files24[fn], sheet=args.sheet, labels=labels, group_mode=args.group_mode),
        }

    # Infer final P from processed matrices
    first_key = common[0]
    p = per_file[first_key][d0.name]["pearson"].shape[0]
    if not labels:
        labels = [str(i) for i in range(p)]
    elif len(labels) != p:
        raise RuntimeError(f"Internal error: labels length {len(labels)} does not equal final matrix size {p}.")

    # Patch Interdependence (Pearson): mean + SD for 0/6/24
    for folder in (d0, d6, d24):
        tkey = folder.name
        tlabel = _pretty_time_label(folder)

        mats = [per_file[fn][tkey]["pearson"] for fn in common]
        mean, sd = mean_sd_stack(mats)
        np.fill_diagonal(sd, 0.0)

        plot_heatmap(
            mean, labels,
            title=f"Patch Interdependence @ {tlabel} (MEAN, pearson)",
            outpath=outdir / f"patch_interdependence_{tlabel}_MEAN_pearson.png",
            vmin=-1.0, vmax=1.0,
        )
        vmax_sd = float(np.nanmax(sd)) if np.isfinite(sd).any() else 1.0
        plot_heatmap(
            sd, labels,
            title=f"Patch Interdependence @ {tlabel} (SD across matched files)",
            outpath=outdir / f"patch_interdependence_{tlabel}_SD_pearson.png",
            vmin=0.0, vmax=vmax_sd,
        )

    # Deltas vs 0h (Pearson): 6h-0h and 24h-0h
    for folder, tag in ((d6, "6h_minus_0h"), (d24, "24h_minus_0h")):
        tkey = folder.name
        deltas = [per_file[fn][tkey]["pearson"] - per_file[fn][d0.name]["pearson"] for fn in common]
        dmean, dsd = mean_sd_stack(deltas)
        np.fill_diagonal(dsd, 0.0)

        lim = symmetric_limits(dmean, pct=99.0)
        plot_heatmap(
            dmean, labels,
            title=f"Change in Patch Overlap ({tag}, mean)",
            outpath=outdir / f"delta_{tag}_MEAN_pearson.png",
            vmin=-lim, vmax=lim, annotate=True, fmt="{:.2f}",
        )

        vmax_dsd = float(np.nanmax(dsd)) if np.isfinite(dsd).any() else 1.0
        plot_heatmap(
            dsd, labels,
            title=f"Change in Patch Overlap ({tag}, SD across matched files)",
            outpath=outdir / f"delta_{tag}_SD_pearson.png",
            vmin=0.0, vmax=vmax_dsd,
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
    for folder in (d0, d6, d24):
        tkey = folder.name
        tlabel = _pretty_time_label(folder)

        mats = [per_file[fn][tkey]["spearman"] for fn in common]
        mean, sd = mean_sd_stack(mats)
        np.fill_diagonal(sd, 0.0)

        plot_heatmap(
            mean, labels,
            title=f"Trace Spearman @ {tlabel} (MEAN)",
            outpath=outdir / f"trace_spearman_{tlabel}_MEAN.png",
            vmin=-1.0, vmax=1.0,
        )
        vmax_sd = float(np.nanmax(sd)) if np.isfinite(sd).any() else 1.0
        plot_heatmap(
            sd, labels,
            title=f"Trace Spearman @ {tlabel} (SD)",
            outpath=outdir / f"trace_spearman_{tlabel}_SD.png",
            vmin=0.0, vmax=vmax_sd,
        )

    # Trace Partial Corr: mean + SD for 0/6/24
    for folder in (d0, d6, d24):
        tkey = folder.name
        tlabel = _pretty_time_label(folder)

        mats = [per_file[fn][tkey]["partialcorr"] for fn in common]
        mean, sd = mean_sd_stack(mats)
        np.fill_diagonal(sd, 0.0)

        plot_heatmap(
            mean, labels,
            title=f"Trace Partial Corr @ {tlabel} (MEAN)",
            outpath=outdir / f"trace_partialcorr_{tlabel}_MEAN.png",
            vmin=-1.0, vmax=1.0,
        )
        vmax_sd = float(np.nanmax(sd)) if np.isfinite(sd).any() else 1.0
        plot_heatmap(
            sd, labels,
            title=f"Trace Partial Corr @ {tlabel} (SD)",
            outpath=outdir / f"trace_partialcorr_{tlabel}_SD.png",
            vmin=0.0, vmax=vmax_sd,
        )

    print(f"Done. Matched files: {len(common)}")
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
