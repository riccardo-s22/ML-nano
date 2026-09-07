import os
import re
import glob
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt

# -----------------------------
# USER SETTINGS
# -----------------------------
BASE_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5"
TIME_DIRS = {"0h": "out_0h", "6h": "out_6h", "24h": "out_24h"}

# 5x5 neighborhood = +/-2 indices
WIN = 2  # total window size = (2*WIN+1) x (2*WIN+1)

# Your provided reference centers (excitation_nm, emission_nm)
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

CHIRALITIES = list(CHIRALITY_CENTERS.keys())

# Temporal delta definition:
# Log-ratio requires strictly positive features. PLE maps can contain small negatives after background subtraction.
# For robustness, default to signed fractional change:
#   delta = (It - I0) / (abs(I0) + eps)
USE_LOG_RATIO = False
EPS = 1e-12

OUT_DIR = os.path.join(BASE_DIR, "multiplexing_results")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# FILE + PARSING HELPERS
# -----------------------------
def list_files(folder: str):
    pat = os.path.join(folder, "*.*")
    files = [
        f for f in glob.glob(pat)
        if os.path.splitext(f)[1].lower() in [".xlsx", ".xls", ".csv", ".tsv", ".txt"]
    ]
    return sorted(files)

def sample_id_from_filename(fp: str) -> str:
    """
    Modify if needed.
    By default:
      - remove extension
      - strip trailing timepoint tokens like _0h, _6h, _24h if present
    """
    base = os.path.splitext(os.path.basename(fp))[0]
    base = re.sub(r"(_0h|_6h|_24h)$", "", base, flags=re.IGNORECASE)
    return base

def safe_read_table(fp: str) -> pd.DataFrame:
    ext = os.path.splitext(fp)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(fp)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(fp)
    if ext in [".tsv"]:
        return pd.read_csv(fp, sep="\t")
    raise ValueError(f"Unsupported file type: {fp}")

def parse_excitation_columns(df: pd.DataFrame):
    """
    Returns:
      exc_cols: list of excitation column names in df
      exc_vals: np.array of excitation values (float) aligned with exc_cols
    Expected column naming: Excitation_00570.000, etc.
    """
    exc_cols = [c for c in df.columns if str(c).startswith("Excitation_")]
    if len(exc_cols) == 0:
        raise ValueError("No Excitation_* columns found.")

    exc_vals = []
    for c in exc_cols:
        s = str(c)
        # Common Origin export: Excitation_00570.000
        m = re.search(r"Excitation_(\d+)\.(\d+)", s)
        if m:
            exc_vals.append(float(m.group(1)) + float("0." + m.group(2)))
        else:
            # fallback numeric parse
            nums = re.findall(r"[\d.]+", s)
            if not nums:
                raise ValueError(f"Could not parse excitation value from column: {c}")
            exc_vals.append(float(nums[0]))

    exc_vals = np.array(exc_vals, dtype=float)
    # sort by excitation
    order = np.argsort(exc_vals)
    exc_vals = exc_vals[order]
    exc_cols = [exc_cols[i] for i in order]
    return exc_cols, exc_vals

def nearest_index(arr: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(arr - target)))

def window_indices(center_idx: int, win: int, n: int):
    lo = max(0, center_idx - win)
    hi = min(n - 1, center_idx + win)
    return np.arange(lo, hi + 1)

def extract_chirality_feature_5x5(df: pd.DataFrame, exc_target: float, em_target: float, win: int = 2):
    """
    Extract 5x5 neighborhood feature at nearest (exc, em).
    Returns:
      feat_max: max value in window
      feat_sum: sum of values in window
      used_exc_vals: excitation values included (len up to 2*win+1)
      used_em_vals: emission values included (len up to 2*win+1)
      used_center: (exc_nearest, em_nearest)
    """
    if "Emission" not in df.columns:
        raise ValueError("No 'Emission' column found.")

    em_vals = df["Emission"].to_numpy(dtype=float)

    exc_cols, exc_vals = parse_excitation_columns(df)

    ie = nearest_index(exc_vals, exc_target)
    im = nearest_index(em_vals, em_target)

    exc_idx = window_indices(ie, win, len(exc_vals))
    em_idx = window_indices(im, win, len(em_vals))

    used_exc_cols = [exc_cols[i] for i in exc_idx]
    used_exc_vals = exc_vals[exc_idx]
    used_em_vals = em_vals[em_idx]

    sub = df.loc[em_idx, used_exc_cols].to_numpy(dtype=float)
    # robust in case of NaNs
    sub = np.nan_to_num(sub, nan=0.0)

    feat_max = float(np.max(sub))
    feat_sum = float(np.sum(sub))

    used_center = (float(exc_vals[ie]), float(em_vals[im]))
    return feat_max, feat_sum, used_exc_vals, used_em_vals, used_center

# -----------------------------
# PLOTS
# -----------------------------
def plot_heatmap(R, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(R, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_clustered_heatmap(R, labels, title, out_png):
    D = 1 - np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(D, 0.0)
    iu = np.triu_indices_from(D, k=1)
    condensed = D[iu]
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    R2 = R[np.ix_(order, order)]
    lab2 = [labels[i] for i in order]
    plot_heatmap(R2, lab2, title, out_png)
    return order

def plot_offdiag_hist(R, title, out_png):
    off = R[~np.eye(R.shape[0], dtype=bool)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(off, bins=20)
    ax.set_title(title)
    ax.set_xlabel("Pearson r (off-diagonal)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# -----------------------------
# MULTIPLEXING METRICS
# -----------------------------
def common_mode_subtract(delta_mat):
    cm = np.nanmean(delta_mat, axis=1, keepdims=True)
    return delta_mat - cm

def corr_from_trends(delta6, delta24, labels):
    # build per-chirality vectors by concatenation over samples
    X = np.vstack([np.concatenate([delta6[:, j], delta24[:, j]]) for j in range(len(labels))])
    R = np.corrcoef(X)
    return R

def mean_abs_offdiag(R):
    off = R[~np.eye(R.shape[0], dtype=bool)]
    return float(np.nanmean(np.abs(off)))

def bootstrap_mean_abs_offdiag(delta6, delta24, labels, common_mode=False, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = delta6.shape[0]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d6 = delta6[idx, :]
        d24 = delta24[idx, :]
        if common_mode:
            d6 = common_mode_subtract(d6)
            d24 = common_mode_subtract(d24)
        Rb = corr_from_trends(d6, d24, labels)
        stats.append(mean_abs_offdiag(Rb))
    stats = np.array(stats, dtype=float)
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

# -----------------------------
# MAIN
# -----------------------------
def main():
    # 1) collect files per timepoint
    by_time = {}
    for t, d in TIME_DIRS.items():
        folder = os.path.join(BASE_DIR, d)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")
        fps = list_files(folder)
        if len(fps) == 0:
            raise FileNotFoundError(f"No readable files found in {folder}")
        m = {sample_id_from_filename(fp): fp for fp in fps}
        by_time[t] = m

    # 2) intersect sample IDs across timepoints
    common_ids = sorted(set(by_time["0h"]).intersection(by_time["6h"]).intersection(by_time["24h"]))
    if len(common_ids) == 0:
        raise RuntimeError("No matching sample IDs across 0h/6h/24h. Check filenames.")
    print(f"Matched samples across all timepoints: {len(common_ids)}")

    # 3) extract features for each timepoint
    # We'll compute both max and sum features for transparency; you can choose which to use downstream.
    feat_max = {t: [] for t in ["0h", "6h", "24h"]}
    feat_sum = {t: [] for t in ["0h", "6h", "24h"]}

    # track which grid points were actually used (nearest coords + window coords)
    used_rows = []

    for sid in common_ids:
        for t in ["0h", "6h", "24h"]:
            fp = by_time[t][sid]
            df = safe_read_table(fp)

            row_max = []
            row_sum = []

            for ch in CHIRALITIES:
                exc0, em0 = CHIRALITY_CENTERS[ch]
                fmax, fsum, used_exc, used_em, used_center = extract_chirality_feature_5x5(
                    df, exc_target=exc0, em_target=em0, win=WIN
                )
                row_max.append(fmax)
                row_sum.append(fsum)

                used_rows.append({
                    "sample_id": sid,
                    "timepoint": t,
                    "chirality": ch,
                    "target_exc_nm": exc0,
                    "target_em_nm": em0,
                    "nearest_exc_nm": used_center[0],
                    "nearest_em_nm": used_center[1],
                    "window_exc_nm": ",".join([f"{x:.3f}" for x in used_exc]),
                    "window_em_nm": ",".join([f"{x:.4f}" for x in used_em]),
                })

            feat_max[t].append(row_max)
            feat_sum[t].append(row_sum)

    for t in feat_max:
        feat_max[t] = np.array(feat_max[t], dtype=float)
        feat_sum[t] = np.array(feat_sum[t], dtype=float)

    # 4) export the nearest-grid audit table (critical for publication defensibility)
    audit = pd.DataFrame(used_rows)
    audit.to_csv(os.path.join(OUT_DIR, "chirality_grid_mapping_audit.csv"), index=False)

    # 5) save extracted feature tables
    for t in ["0h", "6h", "24h"]:
        df_max = pd.DataFrame(feat_max[t], columns=CHIRALITIES)
        df_max.insert(0, "sample_id", common_ids)
        df_max.to_csv(os.path.join(OUT_DIR, f"features_max5x5_{t}.csv"), index=False)

        df_sum = pd.DataFrame(feat_sum[t], columns=CHIRALITIES)
        df_sum.insert(0, "sample_id", common_ids)
        df_sum.to_csv(os.path.join(OUT_DIR, f"features_sum5x5_{t}.csv"), index=False)

    # Choose which feature definition to use for multiplexing proof:
    # Max tends to reflect peak height; Sum tends to be more robust to small peak shifts and noise.
    FEATURES = feat_sum  # <-- change to feat_max if you prefer

    # 6) compute temporal deltas
    I0 = FEATURES["0h"]
    I6 = FEATURES["6h"]
    I24 = FEATURES["24h"]

    if USE_LOG_RATIO:
        # requires positive features
        d6 = np.log((I6 + EPS) / (I0 + EPS))
        d24 = np.log((I24 + EPS) / (I0 + EPS))
    else:
        # robust for signed/near-zero baseline
        d6 = (I6 - I0) / (np.abs(I0) + EPS)
        d24 = (I24 - I0) / (np.abs(I0) + EPS)

    # 7) correlation of temporal trend vectors (raw and common-mode-subtracted)
    R_raw = corr_from_trends(d6, d24, CHIRALITIES)

    d6_cm = common_mode_subtract(d6)
    d24_cm = common_mode_subtract(d24)
    R_cm = corr_from_trends(d6_cm, d24_cm, CHIRALITIES)

    pd.DataFrame(R_raw, index=CHIRALITIES, columns=CHIRALITIES).to_csv(os.path.join(OUT_DIR, "corr_trends_raw.csv"))
    pd.DataFrame(R_cm, index=CHIRALITIES, columns=CHIRALITIES).to_csv(os.path.join(OUT_DIR, "corr_trends_commonmode.csv"))

    # 8) plots
    plot_heatmap(R_raw, CHIRALITIES, "Trend correlation (raw deltas)", os.path.join(OUT_DIR, "heatmap_corr_trends_raw.png"))
    plot_clustered_heatmap(R_raw, CHIRALITIES, "Clustered trend correlation (raw)", os.path.join(OUT_DIR, "clustermap_corr_trends_raw.png"))
    plot_offdiag_hist(R_raw, "Off-diagonal r distribution (raw trends)", os.path.join(OUT_DIR, "hist_offdiag_r_raw.png"))

    plot_heatmap(R_cm, CHIRALITIES, "Trend correlation (common-mode subtracted)", os.path.join(OUT_DIR, "heatmap_corr_trends_commonmode.png"))
    plot_clustered_heatmap(R_cm, CHIRALITIES, "Clustered trend correlation (common-mode)", os.path.join(OUT_DIR, "clustermap_corr_trends_commonmode.png"))
    plot_offdiag_hist(R_cm, "Off-diagonal r distribution (common-mode trends)", os.path.join(OUT_DIR, "hist_offdiag_r_commonmode.png"))

    # 9) summary + bootstrap CI (mean abs off-diagonal is a compact redundancy index)
    summ = []
    summ.append(("mean_abs_offdiag_raw", mean_abs_offdiag(R_raw)))
    summ.append(("mean_abs_offdiag_commonmode", mean_abs_offdiag(R_cm)))

    m_raw, lo_raw, hi_raw = bootstrap_mean_abs_offdiag(d6, d24, CHIRALITIES, common_mode=False, n_boot=2000, seed=0)
    m_cm, lo_cm, hi_cm = bootstrap_mean_abs_offdiag(d6, d24, CHIRALITIES, common_mode=True, n_boot=2000, seed=0)

    summ.extend([
        ("bootstrap_mean_abs_offdiag_raw", m_raw),
        ("bootstrap_ci95_raw_low", lo_raw),
        ("bootstrap_ci95_raw_high", hi_raw),
        ("bootstrap_mean_abs_offdiag_commonmode", m_cm),
        ("bootstrap_ci95_commonmode_low", lo_cm),
        ("bootstrap_ci95_commonmode_high", hi_cm),
    ])

    pd.DataFrame(summ, columns=["metric", "value"]).to_csv(os.path.join(OUT_DIR, "multiplexing_summary.csv"), index=False)

    print("Done. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
