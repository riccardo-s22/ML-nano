"""
Normality assessment of SWCNT chirality peak-intensity features.

Rationale
---------
The global/pairwise PERMANOVA (permanova_full12_within_subject.py) and PERMDISP
tests operate on the 12-chirality peak-intensity feature vectors. PERMANOVA
(Anderson 2001) is a permutation-based, distribution-free method and does NOT
assume (multivariate) normality. This script documents that the underlying
intensity data are in fact non-normal, which is precisely why the non-parametric
permutation approach was chosen over a parametric MANOVA.

It reproduces the normality battery and renders a compact supplementary figure.

Inputs  (same files PERMANOVA consumes):
    features_max5x5_{0h,6h,24h}.csv   in ../  (multiplexing_results/)
Outputs (written next to this script):
    normality_intensity_stats.csv     per-group Shapiro/D'Agostino/skew/kurtosis
    normality_intensity_summary.csv   headline counts
    FigSX_normality_intensity.png/.pdf   supplementary figure
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))   # multiplexing_results/
FILES = {
    "0h":  "features_max5x5_0h.csv",
    "6h":  "features_max5x5_6h.csv",
    "24h": "features_max5x5_24h.csv",
}
TIMEPOINTS = ["0h", "6h", "24h"]

# Okabe-Ito colorblind-safe palette
C_DATA   = "#0072B2"  # blue   - data / reference line
C_ACCENT = "#D55E00"  # vermillion - threshold / rejected
C_OK     = "#009E73"  # green  - not rejected
C_INK    = "#222222"
C_MUTED  = "#8a8a8a"

# ----------------------------------------------------------------------
# Load + align to common subjects (identical to the PERMANOVA script)
# ----------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id in {path}")
    return df

dfs = {tp: load(os.path.join(DATA_DIR, FILES[tp])) for tp in TIMEPOINTS}
chir = [c for c in dfs["0h"].columns if c != "sample_id"]
common = sorted(set(dfs["0h"].sample_id)
                .intersection(dfs["6h"].sample_id)
                .intersection(dfs["24h"].sample_id))
for tp in TIMEPOINTS:
    dfs[tp] = dfs[tp].set_index("sample_id").loc[common].reset_index()

N = len(common)
print(f"N subjects = {N}, chiralities = {len(chir)}")

def bh(p):
    """Benjamini-Hochberg FDR-adjusted p-values."""
    return stats.false_discovery_control(np.asarray(p, float))

def normstats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    W, pW = stats.shapiro(x)
    try:
        _, pK = stats.normaltest(x)          # D'Agostino-Pearson K^2
    except ValueError:
        pK = np.nan
    return dict(n=len(x), skew=float(stats.skew(x)),
                exkurt=float(stats.kurtosis(x)),          # excess kurtosis
                shapiro_W=float(W), shapiro_p=float(pW),
                dagostino_p=float(pK))

# ----------------------------------------------------------------------
# A) RAW per-chirality x timepoint distributions (across subjects)
# ----------------------------------------------------------------------
rows = []
for tp in TIMEPOINTS:
    for c in chir:
        rows.append({"scope": "raw", "timepoint": tp, "chirality": c,
                     "group": f"{tp}:{c}", **normstats(dfs[tp][c])})
A = pd.DataFrame(rows)
A["shapiro_p_BH"] = bh(A["shapiro_p"])

# ----------------------------------------------------------------------
# B) STANDARDIZED features exactly as PERMANOVA sees them
#    (StandardScaler over the stacked (N*3, 12) matrix), pooled per chirality
# ----------------------------------------------------------------------
X = np.stack([dfs[tp][chir].astype(float).to_numpy() for tp in TIMEPOINTS], axis=1)  # (N,3,12)
flat = X.reshape(-1, len(chir))
Xz = ((flat - flat.mean(0)) / flat.std(0, ddof=0)).reshape(X.shape)
rowsB = []
for j, c in enumerate(chir):
    rowsB.append({"scope": "standardized", "timepoint": "pooled", "chirality": c,
                  "group": f"z:{c}", **normstats(Xz[:, :, j].ravel())})
B = pd.DataFrame(rowsB)
B["shapiro_p_BH"] = bh(B["shapiro_p"])

# ----------------------------------------------------------------------
# C) Omnibus on all standardized intensities pooled
# ----------------------------------------------------------------------
allz = Xz.ravel()
allz = allz[np.isfinite(allz)]
Wall, pWall = stats.shapiro(allz)
_, pKall = stats.normaltest(allz)

# ----------------------------------------------------------------------
# Save tables
# ----------------------------------------------------------------------
stats_tbl = pd.concat([A, B], ignore_index=True)
stats_tbl.to_csv(os.path.join(HERE, "normality_intensity_stats.csv"), index=False)

summary = pd.DataFrame([
    {"set": "raw per-chirality x timepoint (across subjects)", "n_groups": len(A),
     "shapiro_pass_p>0.05": int((A.shapiro_p > 0.05).sum()),
     "shapiro_reject_BH0.05": int((A.shapiro_p_BH < 0.05).sum()),
     "median_abs_skew": float(A["skew"].abs().median()),
     "median_excess_kurt": float(A["exkurt"].median())},
    {"set": "standardized per-chirality (pooled)", "n_groups": len(B),
     "shapiro_pass_p>0.05": int((B.shapiro_p > 0.05).sum()),
     "shapiro_reject_BH0.05": int((B.shapiro_p_BH < 0.05).sum()),
     "median_abs_skew": float(B["skew"].abs().median()),
     "median_excess_kurt": float(B["exkurt"].median())},
    {"set": "all standardized pooled (omnibus)", "n_groups": 1,
     "shapiro_pass_p>0.05": int(pWall > 0.05),
     "shapiro_reject_BH0.05": int(pWall < 0.05),
     "median_abs_skew": float(abs(stats.skew(allz))),
     "median_excess_kurt": float(stats.kurtosis(allz))},
])
summary.to_csv(os.path.join(HERE, "normality_intensity_summary.csv"), index=False)
print("\nSummary\n", summary.to_string(index=False))
print(f"\nOmnibus (all standardized, n={allz.size}): "
      f"Shapiro W={Wall:.3f} p={pWall:.2e} | D'Agostino p={pKall:.2e} | "
      f"skew={stats.skew(allz):.2f} excess-kurt={stats.kurtosis(allz):.2f}")

# ----------------------------------------------------------------------
# Figure  (3 panels): QQ of pooled data | Shapiro-p ECDF | skew-kurtosis map
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
})
fig = plt.figure(figsize=(10.5, 3.4))
gs = gridspec.GridSpec(1, 3, wspace=0.34, left=0.06, right=0.985, bottom=0.17, top=0.86)

# --- (a) QQ plot of ALL standardized intensities ---
axa = fig.add_subplot(gs[0, 0])
osm, osr = stats.probplot(allz, dist="norm", fit=False)
axa.scatter(osm, osr, s=6, color=C_DATA, alpha=0.35, edgecolor="none", rasterized=True)
lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
axa.plot(lim, lim, color=C_INK, lw=1.2, ls="--", zorder=3)
axa.set_xlabel("Theoretical normal quantiles")
axa.set_ylabel("Standardized intensity quantiles")
axa.set_title("(a) Normal Q–Q, all features pooled", loc="left")
axa.text(0.04, 0.96,
         f"n = {allz.size}\nShapiro W = {Wall:.3f}\np = {pWall:.1e}\n"
         f"excess kurt = {stats.kurtosis(allz):.1f}",
         transform=axa.transAxes, va="top", ha="left", fontsize=7.8, color=C_INK,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_MUTED, lw=0.6))

# --- (b) sorted Shapiro p-values for the 36 raw groups ---
axb = fig.add_subplot(gs[0, 1])
ps = np.sort(A.shapiro_p.values)
rej = ps < 0.05
idx = np.arange(1, len(ps) + 1)
axb.scatter(idx[rej], ps[rej], s=16, color=C_ACCENT, label="reject (p < 0.05)", zorder=3)
axb.scatter(idx[~rej], ps[~rej], s=16, color=C_OK, label="not rejected", zorder=3)
axb.axhline(0.05, color=C_INK, lw=1.0, ls="--")
axb.text(0.6, 0.062, "α = 0.05", fontsize=7.5, color=C_INK)
axb.set_yscale("log")
axb.set_xlabel("Raw chirality × timepoint group (sorted)")
axb.set_ylabel("Shapiro–Wilk p (log)")
axb.set_title("(b) Per-group normality tests", loc="left")
axb.legend(frameon=False, fontsize=7.5, loc="lower right")
axb.text(0.04, 0.05, f"{int(rej.sum())}/{len(ps)} reject normality",
         transform=axb.transAxes, fontsize=7.8, color=C_ACCENT)

# --- (c) skewness vs excess kurtosis map for the 36 raw groups ---
axc = fig.add_subplot(gs[0, 2])
axc.axhline(0, color=C_MUTED, lw=0.7)
axc.axvline(0, color=C_MUTED, lw=0.7)
sc = axc.scatter(A["skew"], A["exkurt"], s=22, c=[C_ACCENT if r else C_OK for r in (A.shapiro_p < 0.05)],
                 alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
axc.scatter([0], [0], marker="*", s=130, color=C_INK, zorder=4)
axc.annotate("normal\n(0, 0)", (0, 0), textcoords="offset points", xytext=(8, 6),
             fontsize=7.5, color=C_INK)
axc.set_xlabel("Skewness")
axc.set_ylabel("Excess kurtosis")
axc.set_title("(c) Shape of raw distributions", loc="left")

fig.savefig(os.path.join(HERE, "FigSX_normality_intensity.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "FigSX_normality_intensity.pdf"), bbox_inches="tight")
print("\nSaved figure -> FigSX_normality_intensity.png / .pdf")
