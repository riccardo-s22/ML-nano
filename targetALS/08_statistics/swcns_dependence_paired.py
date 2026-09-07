import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel, spearmanr
from sklearn.covariance import LedoitWolf


# =============================
# USER SETTINGS
# =============================
BASE_PATH = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5"

TIME_POINTS = {
    "0h":  "out_0h",
    "6h":  "out_6h",
    "24h": "out_24h",
}

OUTDIR = os.path.join(BASE_PATH, "dependence_results_paired")

# If your Excel export does NOT include an emission wavelength axis, provide a mapping.
# If it DOES include emission axis in first column, it will be used automatically.
EMISSION_MIN = 900.0
EMISSION_MAX = 1350.0

# Patch-based interdependence settings
PATCH_HALF = 2                 # 5x5
PATCH_METRIC = "pearson"       # "pearson" recommended; "cosine" optional
PAD_MODE = "edge"              # avoid zero vectors at borders

# Peak recentering settings (per file, per time point)
RECENTER = True
RECENTER_EX_HALF_STEPS = 2     # +/- 2 excitation indices (5 nm steps => +/-10 nm if step=5)
RECENTER_EM_HALF_BINS = 10     # +/- 10 emission bins

# Trace-based dependence settings
TRACE_USE = True
TRACE_EM_HALF = 2              # sum 5 emission bins around center across ALL excitation columns
TRACE_METHOD = "spearman"      # "spearman"
PARTIAL_CORR = True            # conditional dependence via shrinkage precision (recommended)

# Bootstrap settings for delta CIs (paired resampling of files)
BOOTSTRAP = True
N_BOOT = 2000
CI_ALPHA = 0.05

# Chirality centers (excitation_nm, emission_nm). Update to your validated E22/E11 coordinates.
CHIRALITY_CENTERS = {
    '6.5':  (573, 975),
    '7.5':  (647, 1024),
    '7.6':  (645, 1115),
    '8.3':  (667, 952),
    '8.4':  (726, 1100),
    '8.6':  (718, 1170),
    '8.7':  (726, 1260),
    '9.4':  (720, 1100),
    '9.5':  (800, 1240),
    '10.2': (740, 1050),
    '10.3': (800, 1100),
    '10.5': (850, 1250),
}


# =============================
# IO + utilities
# =============================
def load_ns_excel(path: str):
    """
    Loads NS Spectralyzer-style Excel exports.

    Returns:
      M: ndarray (n_excitation, n_emission) with excitation along rows, emission along columns
      ex_axis: excitation wavelengths (nm)
      em_axis: emission wavelengths (nm) if present, else pixel/bin indices
    """
    # Layout A: emission axis in first column (index_col=0)
    try:
        df0 = pd.read_excel(path, index_col=0, engine="openpyxl")
        if all(str(c).startswith("Excitation_") for c in df0.columns):
            ex_axis = (
                df0.columns.astype(str)
                .str.replace("Excitation_", "", regex=False)
                .astype(float)
                .to_numpy()
            )
            em_axis = pd.to_numeric(df0.index, errors="coerce").to_numpy()
            if np.all(np.isfinite(em_axis)):
                M = df0.to_numpy(dtype=float).T
                return M, ex_axis, em_axis
    except Exception:
        pass

    # Layout B: no emission axis
    df = pd.read_excel(path, engine="openpyxl")
    if not all(str(c).startswith("Excitation_") for c in df.columns):
        raise ValueError(f"Could not identify excitation columns in {path}")
    ex_axis = (
        df.columns.astype(str)
        .str.replace("Excitation_", "", regex=False)
        .astype(float)
        .to_numpy()
    )
    em_axis = np.arange(df.shape[0], dtype=float)
    M = df.to_numpy(dtype=float).T
    return M, ex_axis, em_axis


def sort_axes(M, ex_axis, em_axis):
    if ex_axis[0] > ex_axis[-1]:
        ex_axis = ex_axis[::-1]
        M = M[::-1, :]
    if em_axis[0] > em_axis[-1]:
        em_axis = em_axis[::-1]
        M = M[:, ::-1]
    return M, ex_axis, em_axis


def zscore(v, eps=1e-12):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + eps)


def extract_patch(M, ex_axis, em_axis, center_ex, center_em, half=2, pad_mode="edge"):
    ex_idx = int(np.argmin(np.abs(ex_axis - center_ex)))
    em_idx = int(np.argmin(np.abs(em_axis - center_em)))

    pad = half
    Mp = np.pad(M, ((pad, pad), (pad, pad)), mode=pad_mode)
    ex_idx += pad
    em_idx += pad

    return Mp[ex_idx-half:ex_idx+half+1, em_idx-half:em_idx+half+1]


def refine_center(M, ex_axis, em_axis, ex0, em0, ex_half_steps=2, em_half_bins=10):
    """
    Refine expected (ex0, em0) by searching a local window for max intensity.
    Returns (ex_nm_refined, em_nm_refined).
    """
    ex_idx0 = int(np.argmin(np.abs(ex_axis - ex0)))
    em_idx0 = int(np.argmin(np.abs(em_axis - em0)))

    ex_lo = max(0, ex_idx0 - ex_half_steps)
    ex_hi = min(M.shape[0], ex_idx0 + ex_half_steps + 1)
    em_lo = max(0, em_idx0 - em_half_bins)
    em_hi = min(M.shape[1], em_idx0 + em_half_bins + 1)

    window = M[ex_lo:ex_hi, em_lo:em_hi]
    k = int(np.argmax(window))
    dx, dy = np.unravel_index(k, window.shape)

    ex_idx = ex_lo + dx
    em_idx = em_lo + dy
    return ex_axis[ex_idx], em_axis[em_idx]


def similarity_matrix_from_patches(patches: dict, metric="pearson"):
    keys = list(patches.keys())
    V = []
    for k in keys:
        vec = patches[k].reshape(-1)
        if metric == "pearson":
            vec = zscore(vec)
        elif metric == "cosine":
            vec = vec / (np.linalg.norm(vec) + 1e-12)
        else:
            raise ValueError("metric must be 'pearson' or 'cosine'")
        V.append(vec)
    V = np.vstack(V)

    n = len(keys)
    S = np.zeros((n, n), float)
    for i in range(n):
        for j in range(n):
            if metric == "pearson":
                S[i, j] = np.corrcoef(V[i], V[j])[0, 1]
            else:
                S[i, j] = 1 - cosine(V[i], V[j])

    return pd.DataFrame(S, index=keys, columns=keys)


def chirality_traces(M, ex_axis, em_axis, centers, em_half=2, recenter=True):
    """
    For each chirality: sum emission bins around center (±em_half) for every excitation row.
    Returns X: (n_chiralities, n_excitation) z-scored per chirality.
    """
    keys = list(centers.keys())
    X = []
    for k in keys:
        ex0, em0 = centers[k]
        if recenter:
            ex0, em0 = refine_center(M, ex_axis, em_axis, ex0, em0,
                                     ex_half_steps=RECENTER_EX_HALF_STEPS,
                                     em_half_bins=RECENTER_EM_HALF_BINS)
        ex_idx = int(np.argmin(np.abs(ex_axis - ex0)))
        em_idx = int(np.argmin(np.abs(em_axis - em0)))
        lo = max(0, em_idx - em_half)
        hi = min(M.shape[1], em_idx + em_half + 1)
        trace = M[:, lo:hi].sum(axis=1)  # across emission band, for each excitation
        trace = zscore(trace)
        X.append(trace)
    X = np.vstack(X)
    return keys, X


def spearman_matrix(X):
    """
    X: (n_vars, n_samples). Spearman correlation across samples.
    """
    n = X.shape[0]
    R = np.zeros((n, n), float)
    P = np.ones((n, n), float)
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(X[i], X[j])
            R[i, j] = r
            P[i, j] = p
    return R, P


def partial_corr_from_samples(X):
    """
    X: (n_vars, n_samples). Use shrinkage covariance on samples to compute partial correlations.
    """
    Y = X.T  # (n_samples, n_vars)
    lw = LedoitWolf().fit(Y)
    Sigma = lw.covariance_
    Prec = np.linalg.inv(Sigma)
    d = np.sqrt(np.diag(Prec))
    PC = -Prec / (d[:, None] * d[None, :])
    np.fill_diagonal(PC, 1.0)
    return PC


def plot_heatmap(df, title, out_png, vmin=None, vmax=None, cmap="viridis", annot=False, fmt="{:.2f}"):
    plt.figure(figsize=(9, 7))
    im = plt.imshow(df.values, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im)
    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(range(df.shape[0]), df.index)
    if annot:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, fmt.format(df.values[i, j]), ha="center", va="center", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def bh_fdr(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR.
    pvals: 1D array of p-values.
    Returns: boolean reject array, q-values array
    """
    pvals = np.asarray(pvals, float)
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    reject_ranked = ranked <= thresh
    if not np.any(reject_ranked):
        reject = np.zeros(m, dtype=bool)
    else:
        kmax = np.max(np.where(reject_ranked)[0])
        reject = np.zeros(m, dtype=bool)
        reject[order[:kmax + 1]] = True

    # q-values (BH adjusted)
    q = np.empty(m, float)
    q[order] = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    return reject, q


def upper_triangle_pairs(labels):
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((i, j))
    return pairs


# =============================
# Core per-file computations
# =============================
def compute_patch_interdependence(path: str):
    M, ex_axis, em_axis = load_ns_excel(path)
    M, ex_axis, em_axis = sort_axes(M, ex_axis, em_axis)

    # If emission axis missing, map bins linearly to nm range
    if np.array_equal(em_axis, np.arange(em_axis.size, dtype=float)):
        em_axis = np.linspace(EMISSION_MIN, EMISSION_MAX, em_axis.size)

    patches = {}
    for name, (cex, cem) in CHIRALITY_CENTERS.items():
        if RECENTER:
            cex, cem = refine_center(M, ex_axis, em_axis, cex, cem,
                                     ex_half_steps=RECENTER_EX_HALF_STEPS,
                                     em_half_bins=RECENTER_EM_HALF_BINS)
        patches[name] = extract_patch(M, ex_axis, em_axis, cex, cem,
                                      half=PATCH_HALF, pad_mode=PAD_MODE)

    S = similarity_matrix_from_patches(patches, metric=PATCH_METRIC)
    return S


def compute_trace_dependence(path: str):
    M, ex_axis, em_axis = load_ns_excel(path)
    M, ex_axis, em_axis = sort_axes(M, ex_axis, em_axis)

    if np.array_equal(em_axis, np.arange(em_axis.size, dtype=float)):
        em_axis = np.linspace(EMISSION_MIN, EMISSION_MAX, em_axis.size)

    labels, X = chirality_traces(M, ex_axis, em_axis, CHIRALITY_CENTERS,
                                 em_half=TRACE_EM_HALF, recenter=RECENTER)

    R_spear, P_spear = spearman_matrix(X)
    R_spear_df = pd.DataFrame(R_spear, index=labels, columns=labels)
    P_spear_df = pd.DataFrame(P_spear, index=labels, columns=labels)

    PC_df = None
    if PARTIAL_CORR:
        PC = partial_corr_from_samples(X)
        PC_df = pd.DataFrame(PC, index=labels, columns=labels)

    return R_spear_df, P_spear_df, PC_df


# =============================
# Paired folder analysis
# =============================
def matched_file_table():
    """
    Build a table of matched files across time points by basename.
    Keeps only basenames present in ALL specified folders.
    """
    tp_files = {}
    for tp, folder in TIME_POINTS.items():
        folder_path = os.path.join(BASE_PATH, folder)
        files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        mapping = {os.path.basename(f): f for f in files}
        tp_files[tp] = mapping

    common = set.intersection(*(set(m.keys()) for m in tp_files.values()))
    common = sorted(common)

    rows = []
    for b in common:
        row = {"basename": b}
        for tp in TIME_POINTS.keys():
            row[tp] = tp_files[tp][b]
        rows.append(row)

    return pd.DataFrame(rows)


def stack_mats(mats):
    arr = np.stack([m.values for m in mats], axis=0)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, sd


def paired_edge_stats(delta_mats, labels, alpha=0.05):
    """
    delta_mats: list of delta matrices (DataFrames) across matched files
    Performs paired t-test of delta vs 0 for each unique edge (upper triangle).
    Returns:
      - edge table with mean_delta, sd_delta, t, p, q, reject
      - significance mask matrix (q <= alpha) on full NxN
    """
    pairs = upper_triangle_pairs(labels)
    D = np.stack([m.values for m in delta_mats], axis=0)  # (n_files, n, n)

    pvals = []
    stats = []
    for (i, j) in pairs:
        x = D[:, i, j]
        t, p = ttest_rel(x, np.zeros_like(x), nan_policy="omit")
        pvals.append(p)
        stats.append((i, j, np.nanmean(x), np.nanstd(x, ddof=1) if len(x) > 1 else 0.0, t, p))

    pvals = np.array(pvals, float)
    reject, qvals = bh_fdr(pvals, alpha=alpha)

    rows = []
    for k, (i, j, md, sd, t, p) in enumerate(stats):
        rows.append({
            "i": i, "j": j,
            "chirality_i": labels[i],
            "chirality_j": labels[j],
            "mean_delta": md,
            "sd_delta": sd,
            "t_stat": t,
            "p_value": p,
            "q_value": qvals[k],
            "reject_fdr": bool(reject[k]),
        })
    edge_df = pd.DataFrame(rows).sort_values("q_value")

    sig = np.zeros((len(labels), len(labels)), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        if reject[k]:
            sig[i, j] = True
            sig[j, i] = True
    np.fill_diagonal(sig, False)

    sig_df = pd.DataFrame(sig.astype(int), index=labels, columns=labels)
    return edge_df, sig_df


def bootstrap_ci_mean_delta(delta_mats, alpha=0.05, n_boot=2000, seed=0):
    """
    Paired bootstrap over files: resample delta matrices with replacement, compute mean.
    Returns CI_low and CI_high matrices for mean delta.
    """
    rng = np.random.default_rng(seed)
    D = np.stack([m.values for m in delta_mats], axis=0)  # (n_files, n, n)
    n = D.shape[0]
    boots = np.empty((n_boot, D.shape[1], D.shape[2]), float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = D[idx].mean(axis=0)

    lo = np.quantile(boots, alpha / 2, axis=0)
    hi = np.quantile(boots, 1 - alpha / 2, axis=0)
    return lo, hi


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) matched file table
    match_df = matched_file_table()
    match_df.to_csv(os.path.join(OUTDIR, "matched_files.csv"), index=False)
    if match_df.empty:
        raise RuntimeError("No matched .xlsx basenames found across 0h/6h/24h folders.")

    # Per-timepoint outputs
    for tp in TIME_POINTS.keys():
        os.makedirs(os.path.join(OUTDIR, tp, "per_file"), exist_ok=True)

    # 2) compute per-file interdependence (patch) and (optional) trace metrics
    patch_per_tp = {tp: [] for tp in TIME_POINTS.keys()}
    trace_spear_per_tp = {tp: [] for tp in TIME_POINTS.keys()}
    trace_partial_per_tp = {tp: [] for tp in TIME_POINTS.keys()} if PARTIAL_CORR else None

    for _, row in match_df.iterrows():
        basename = row["basename"]
        for tp in TIME_POINTS.keys():
            path = row[tp]

            # Patch interdependence
            S = compute_patch_interdependence(path)
            patch_per_tp[tp].append(S)
            S.to_csv(os.path.join(OUTDIR, tp, "per_file", f"{basename}_patch_{PATCH_METRIC}_5x5.csv"))

            # Trace dependence (optional)
            if TRACE_USE:
                R_spear, P_spear, PC = compute_trace_dependence(path)
                trace_spear_per_tp[tp].append(R_spear)
                R_spear.to_csv(os.path.join(OUTDIR, tp, "per_file", f"{basename}_trace_spearman.csv"))
                P_spear.to_csv(os.path.join(OUTDIR, tp, "per_file", f"{basename}_trace_spearman_p.csv"))
                if PARTIAL_CORR and PC is not None:
                    trace_partial_per_tp[tp].append(PC)
                    PC.to_csv(os.path.join(OUTDIR, tp, "per_file", f"{basename}_trace_partialcorr.csv"))

    labels = patch_per_tp["0h"][0].index.tolist()

    # 3) aggregate within each timepoint: mean + sd + heatmaps
    for tp in TIME_POINTS.keys():
        mean, sd = stack_mats(patch_per_tp[tp])
        mean_df = pd.DataFrame(mean, index=labels, columns=labels)
        sd_df = pd.DataFrame(sd, index=labels, columns=labels)

        mean_df.to_csv(os.path.join(OUTDIR, f"patch_interdependence_{tp}_MEAN_{PATCH_METRIC}_5x5.csv"))
        sd_df.to_csv(os.path.join(OUTDIR, f"patch_interdependence_{tp}_SD_{PATCH_METRIC}_5x5.csv"))

        vmin, vmax = (-1, 1) if PATCH_METRIC == "pearson" else (0, 1)
        plot_heatmap(mean_df,
                     f"Patch Interdependence @ {tp} (MEAN, {PATCH_METRIC})",
                     os.path.join(OUTDIR, f"patch_interdependence_{tp}_MEAN_{PATCH_METRIC}_5x5.png"),
                     vmin=vmin, vmax=vmax, cmap="viridis")

        plot_heatmap(sd_df,
                     f"Patch Interdependence @ {tp} (SD across matched files)",
                     os.path.join(OUTDIR, f"patch_interdependence_{tp}_SD_{PATCH_METRIC}_5x5.png"),
                     vmin=0, vmax=None, cmap="magma")

    # 4) paired delta analysis (per matched file)
    def delta_pack(tp_hi, tp_lo, name):
        deltas = []
        for i in range(len(patch_per_tp[tp_hi])):
            deltas.append(patch_per_tp[tp_hi][i] - patch_per_tp[tp_lo][i])

        # mean delta
        D = np.stack([d.values for d in deltas], axis=0)
        mean_delta = D.mean(axis=0)
        sd_delta = D.std(axis=0, ddof=1) if D.shape[0] > 1 else np.zeros_like(mean_delta)

        mean_delta_df = pd.DataFrame(mean_delta, index=labels, columns=labels)
        sd_delta_df = pd.DataFrame(sd_delta, index=labels, columns=labels)

        mean_delta_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_MEAN_{PATCH_METRIC}_5x5.csv"))
        sd_delta_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_SD_{PATCH_METRIC}_5x5.csv"))

        # dynamic scaling (no clipping)
        lim = float(np.nanmax(np.abs(mean_delta)))
        lim = lim if lim > 0 else 0.1

        plot_heatmap(mean_delta_df,
                     f"Change in Patch Overlap ({name}, mean)",
                     os.path.join(OUTDIR, f"delta_{name}_MEAN_{PATCH_METRIC}_5x5.png"),
                     vmin=-lim, vmax=lim, cmap="coolwarm", annot=True)

        plot_heatmap(sd_delta_df,
                     f"Change in Patch Overlap ({name}, SD across matched files)",
                     os.path.join(OUTDIR, f"delta_{name}_SD_{PATCH_METRIC}_5x5.png"),
                     vmin=0, vmax=None, cmap="magma")

        # paired edge stats + FDR
        edge_df, sig_mask_df = paired_edge_stats(deltas, labels, alpha=0.05)
        edge_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_paired_edge_stats_FDR.csv"), index=False)
        sig_mask_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_significant_edges_mask_FDR.csv"))

        plot_heatmap(sig_mask_df,
                     f"Significant delta edges ({name}) [FDR<=0.05]",
                     os.path.join(OUTDIR, f"delta_{name}_significant_edges_mask_FDR.png"),
                     vmin=0, vmax=1, cmap="Greys", annot=False)

        # bootstrap CI for mean delta (paired resampling)
        if BOOTSTRAP and len(deltas) >= 2:
            lo, hi = bootstrap_ci_mean_delta(deltas, alpha=CI_ALPHA, n_boot=N_BOOT, seed=0)
            lo_df = pd.DataFrame(lo, index=labels, columns=labels)
            hi_df = pd.DataFrame(hi, index=labels, columns=labels)
            lo_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_MEAN_CI_low.csv"))
            hi_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_MEAN_CI_high.csv"))

            # CI-excludes-zero mask
            ci_sig = (lo > 0) | (hi < 0)
            np.fill_diagonal(ci_sig, False)
            ci_sig_df = pd.DataFrame(ci_sig.astype(int), index=labels, columns=labels)
            ci_sig_df.to_csv(os.path.join(OUTDIR, f"delta_{name}_CI_excludes_zero_mask.csv"))
            plot_heatmap(ci_sig_df,
                         f"Delta edges with CI excluding 0 ({name})",
                         os.path.join(OUTDIR, f"delta_{name}_CI_excludes_zero_mask.png"),
                         vmin=0, vmax=1, cmap="Greys", annot=False)

        return deltas

    deltas_6 = delta_pack("6h", "0h", "6h_minus_0h")
    deltas_24 = delta_pack("24h", "0h", "24h_minus_0h")

    # 5) trace-based dependence aggregation (optional)
    if TRACE_USE:
        # Spearman mean matrices per timepoint
        for tp in TIME_POINTS.keys():
            arr = np.stack([m.values for m in trace_spear_per_tp[tp]], axis=0)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            mean_df = pd.DataFrame(mean, index=labels, columns=labels)
            sd_df = pd.DataFrame(sd, index=labels, columns=labels)

            mean_df.to_csv(os.path.join(OUTDIR, f"trace_spearman_{tp}_MEAN.csv"))
            sd_df.to_csv(os.path.join(OUTDIR, f"trace_spearman_{tp}_SD.csv"))

            plot_heatmap(mean_df, f"Trace Spearman @ {tp} (MEAN)",
                         os.path.join(OUTDIR, f"trace_spearman_{tp}_MEAN.png"),
                         vmin=-1, vmax=1, cmap="viridis")
            plot_heatmap(sd_df, f"Trace Spearman @ {tp} (SD)",
                         os.path.join(OUTDIR, f"trace_spearman_{tp}_SD.png"),
                         vmin=0, vmax=None, cmap="magma")

        # Partial correlation mean per timepoint (conditional dependence)
        if PARTIAL_CORR and trace_partial_per_tp is not None:
            for tp in TIME_POINTS.keys():
                arr = np.stack([m.values for m in trace_partial_per_tp[tp]], axis=0)
                mean = arr.mean(axis=0)
                sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
                mean_df = pd.DataFrame(mean, index=labels, columns=labels)
                sd_df = pd.DataFrame(sd, index=labels, columns=labels)

                mean_df.to_csv(os.path.join(OUTDIR, f"trace_partialcorr_{tp}_MEAN.csv"))
                sd_df.to_csv(os.path.join(OUTDIR, f"trace_partialcorr_{tp}_SD.csv"))

                plot_heatmap(mean_df, f"Trace Partial Corr @ {tp} (MEAN)",
                             os.path.join(OUTDIR, f"trace_partialcorr_{tp}_MEAN.png"),
                             vmin=-1, vmax=1, cmap="viridis")
                plot_heatmap(sd_df, f"Trace Partial Corr @ {tp} (SD)",
                             os.path.join(OUTDIR, f"trace_partialcorr_{tp}_SD.png"),
                             vmin=0, vmax=None, cmap="magma")

    print(f"Done. Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
