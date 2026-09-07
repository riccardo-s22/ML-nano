import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# -----------------------------
# USER SETTINGS
# -----------------------------
BASE_PATH = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5"

TIME_POINTS = {
    "0h":  "out_0h",
    "6h":  "out_6h",
    "24h": "out_24h",
}

OUTDIR = os.path.join(BASE_PATH, "dependence_results")

# If your Excel does not include emission wavelengths, provide a mapping here:
EMISSION_MIN = 900.0
EMISSION_MAX = 1350.0

# Similarity metric: "pearson" is recommended; "cosine" is available.
METRIC = "pearson"

# Chirality centers (excitation_nm, emission_nm) — update to your true centers
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

# -----------------------------
# IO + utilities
# -----------------------------
def load_ns_excel(path: str):
    """
    Loads NS Spectralyzer-style Excel exports.

    Returns:
      M: ndarray (n_excitation, n_emission) with excitation along rows, emission along columns
      ex_axis: excitation wavelengths (nm)
      em_axis: emission wavelengths (nm) if present, else pixel/bin indices
    """
    # Try layout with emission axis in first column
    try:
        df0 = pd.read_excel(path, index_col=0, engine="openpyxl")
        if all(str(c).startswith("Excitation_") for c in df0.columns):
            ex_axis = df0.columns.astype(str).str.replace("Excitation_", "", regex=False).astype(float).to_numpy()
            em_axis = pd.to_numeric(df0.index, errors="coerce").to_numpy()
            if np.all(np.isfinite(em_axis)):
                M = df0.to_numpy(dtype=float).T
                return M, ex_axis, em_axis
    except Exception:
        pass

    # Fallback: no emission axis present
    df = pd.read_excel(path, engine="openpyxl")
    if not all(str(c).startswith("Excitation_") for c in df.columns):
        raise ValueError(f"Could not identify excitation columns in {path}")
    ex_axis = df.columns.astype(str).str.replace("Excitation_", "", regex=False).astype(float).to_numpy()
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

def plot_heatmap(df, title, out_png, vmin=None, vmax=None, cmap="viridis", annot=False):
    plt.figure(figsize=(9, 7))
    im = plt.imshow(df.values, aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im)
    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    plt.yticks(range(df.shape[0]), df.index)
    if annot:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                plt.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

# -----------------------------
# Main analysis
# -----------------------------
def analyze_file(path: str):
    M, ex_axis, em_axis = load_ns_excel(path)
    M, ex_axis, em_axis = sort_axes(M, ex_axis, em_axis)

    # If emission axis missing, map bins linearly to nm range
    if np.array_equal(em_axis, np.arange(em_axis.size, dtype=float)):
        em_axis = np.linspace(EMISSION_MIN, EMISSION_MAX, em_axis.size)

    patches = {}
    for name, (cex, cem) in CHIRALITY_CENTERS.items():
        patches[name] = extract_patch(M, ex_axis, em_axis, cex, cem, half=2, pad_mode="edge")

    S = similarity_matrix_from_patches(patches, metric=METRIC)
    return S

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    per_time_mats = {}  # label -> list of DataFrames

    for tp, folder in TIME_POINTS.items():
        folder_path = os.path.join(BASE_PATH, folder)
        files = sorted(glob.glob(os.path.join(folder_path, "*.xlsx")))

        if not files:
            print(f"[WARN] No .xlsx files found in {folder_path}")
            continue

        mats = []
        tp_out = os.path.join(OUTDIR, tp)
        os.makedirs(tp_out, exist_ok=True)

        for f in files:
            try:
                S = analyze_file(f)
                mats.append(S)

                stem = os.path.splitext(os.path.basename(f))[0]
                S.to_csv(os.path.join(tp_out, f"{stem}_interdependence_{METRIC}_5x5.csv"))

            except Exception as e:
                print(f"[ERROR] {tp} file failed: {f}\n  {e}")

        per_time_mats[tp] = mats

        # Aggregate mean + sd
        if mats:
            arr = np.stack([m.values for m in mats], axis=0)  # (n_files, n, n)
            mean = pd.DataFrame(arr.mean(axis=0), index=mats[0].index, columns=mats[0].columns)
            sd   = pd.DataFrame(arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean.values),
                                index=mats[0].index, columns=mats[0].columns)

            mean.to_csv(os.path.join(OUTDIR, f"interdependence_{tp}_MEAN_{METRIC}_5x5.csv"))
            sd.to_csv(os.path.join(OUTDIR, f"interdependence_{tp}_SD_{METRIC}_5x5.csv"))

            vmin, vmax = (-1, 1) if METRIC == "pearson" else (0, 1)
            plot_heatmap(mean, f"Interdependence @ {tp} (MEAN, {METRIC})",
                         os.path.join(OUTDIR, f"interdependence_{tp}_MEAN_{METRIC}_5x5.png"),
                         vmin=vmin, vmax=vmax)

            plot_heatmap(sd, f"Interdependence @ {tp} (SD across files)",
                         os.path.join(OUTDIR, f"interdependence_{tp}_SD_{METRIC}_5x5.png"),
                         vmin=0, vmax=None, cmap="magma")

    # Deltas (mean matrices)
    def load_mean(tp):
        p = os.path.join(OUTDIR, f"interdependence_{tp}_MEAN_{METRIC}_5x5.csv")
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
        return None

    m0 = load_mean("0h")
    m6 = load_mean("6h")
    m24 = load_mean("24h")

    if m0 is not None and m24 is not None:
        d = m24 - m0
        d.to_csv(os.path.join(OUTDIR, f"delta_24h_minus_0h_MEAN_{METRIC}_5x5.csv"))
        plot_heatmap(d, "Change in Spectral Overlap (24h - 0h, mean)",
                     os.path.join(OUTDIR, f"delta_24h_minus_0h_MEAN_{METRIC}_5x5.png"),
                     vmin=-0.2, vmax=0.2, cmap="coolwarm", annot=True)

    if m0 is not None and m6 is not None:
        d = m6 - m0
        d.to_csv(os.path.join(OUTDIR, f"delta_6h_minus_0h_MEAN_{METRIC}_5x5.csv"))
        plot_heatmap(d, "Change in Spectral Overlap (6h - 0h, mean)",
                     os.path.join(OUTDIR, f"delta_6h_minus_0h_MEAN_{METRIC}_5x5.png"),
                     vmin=-0.2, vmax=0.2, cmap="coolwarm", annot=True)

    print(f"Done. Outputs in: {OUTDIR}")

if __name__ == "__main__":
    main()
