import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ----------------------------
# SETTINGS
# ----------------------------
DATA_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\multiplexing_results"
FILES = {
    "0h":  "features_max5x5_0h.csv",
    "6h":  "features_max5x5_6h.csv",
    "24h": "features_max5x5_24h.csv",
}
TIMEPOINTS = ["0h", "6h", "24h"]
N_PERM = 9999
SEED = 42
OUT_DIR = os.path.join(DATA_DIR, "permdisp_full12_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------
def holm_adjust(pvals):
    pvals = np.array(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * pvals[idx])
        prev = max(prev, val)
        adj[idx] = prev
    return adj

def anova_F_from_D(D):
    """
    D: N x k matrix of distances (k=3 timepoints), columns correspond to TIMEPOINTS order.
    """
    N, k = D.shape
    grand = D.mean()
    means = D.mean(axis=0)
    ss_between = N * np.sum((means - grand) ** 2)
    ss_within = np.sum((D - means[None, :]) ** 2)

    df_between = k - 1
    df_within = N * k - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F = ms_between / ms_within
    return float(F), float(ss_between), float(ss_within), int(df_between), int(df_within)

# ----------------------------
# LOAD + ALIGN + STANDARDIZE
# ----------------------------
dfs = {}
chir_cols = None
for tp in TIMEPOINTS:
    df = pd.read_csv(os.path.join(DATA_DIR, FILES[tp]))
    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id in {FILES[tp]}")
    cols = [c for c in df.columns if c != "sample_id"]
    dfs[tp] = df
    chir_cols = cols if chir_cols is None else chir_cols

common_ids = sorted(set(dfs["0h"]["sample_id"])
                    .intersection(dfs["6h"]["sample_id"])
                    .intersection(dfs["24h"]["sample_id"]))

for tp in TIMEPOINTS:
    dfs[tp] = dfs[tp].set_index("sample_id").loc[common_ids].reset_index()

X0 = dfs["0h"][chir_cols].astype(float).to_numpy()
X6 = dfs["6h"][chir_cols].astype(float).to_numpy()
X24 = dfs["24h"][chir_cols].astype(float).to_numpy()

T = np.stack([X0, X6, X24], axis=1)  # (N,3,12)

# Standardize globally across all observations
scaler = StandardScaler()
Tz = scaler.fit_transform(T.reshape(-1, T.shape[-1])).reshape(T.shape)

N, k, p = Tz.shape  # k=3

# ----------------------------
# DISTANCES TO TIMEPOINT CENTROIDS
# ----------------------------
centroids = Tz.mean(axis=0)  # (3,12)
dist = np.linalg.norm(Tz - centroids[None, :, :], axis=2)  # (N,3)

dist_df = pd.DataFrame(dist, columns=TIMEPOINTS)
dist_df.insert(0, "sample_id", common_ids)
dist_df.to_csv(os.path.join(OUT_DIR, "permdisp_distances_wide.csv"), index=False)

long = dist_df.melt(id_vars="sample_id", var_name="timepoint", value_name="dist_to_centroid")
long.to_csv(os.path.join(OUT_DIR, "permdisp_distances_long.csv"), index=False)

# ----------------------------
# GLOBAL TESTS (PAIRED)
# ----------------------------
fried = stats.friedmanchisquare(dist_df["0h"].values, dist_df["6h"].values, dist_df["24h"].values)

# Permutation test on an ANOVA-style F with within-subject timepoint permutations
rng = np.random.default_rng(SEED)
perms3 = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]], dtype=int)

F_obs, ssb, ssw, dfb, dfw = anova_F_from_D(dist)

F_perm = np.empty(N_PERM, dtype=float)
for i in range(N_PERM):
    choice = rng.integers(0, 6, size=N)
    idx = perms3[choice]
    Dp = dist[np.arange(N)[:, None], idx]
    F_perm[i], *_ = anova_F_from_D(Dp)

p_perm = (1 + np.sum(F_perm >= F_obs)) / (1 + N_PERM)

global_df = pd.DataFrame([{
    "n_samples": N,
    "friedman_stat": float(fried.statistic),
    "friedman_p": float(fried.pvalue),
    "perm_F_obs": float(F_obs),
    "perm_p": float(p_perm),
    "n_perm": int(N_PERM),
}])
global_df.to_csv(os.path.join(OUT_DIR, "permdisp_global.csv"), index=False)

# ----------------------------
# PAIRWISE TESTS (PAIRED WILCOXON)
# ----------------------------
pairs = [("0h","6h"), ("6h","24h"), ("0h","24h")]
rows = []
for a,b in pairs:
    w = stats.wilcoxon(dist_df[b].values, dist_df[a].values, zero_method="wilcox", alternative="two-sided")
    diff = dist_df[b].values - dist_df[a].values
    rows.append({
        "comparison": f"{a} vs {b}",
        "n": N,
        "mean_diff_(second-first)": float(np.mean(diff)),
        "median_diff_(second-first)": float(np.median(diff)),
        "wilcoxon_stat": float(w.statistic),
        "p": float(w.pvalue),
    })
pair_df = pd.DataFrame(rows)
pair_df["p_holm"] = holm_adjust(pair_df["p"].values)
pair_df.to_csv(os.path.join(OUT_DIR, "permdisp_pairwise.csv"), index=False)

# ----------------------------
# PLOTS
# ----------------------------
# Boxplot
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.boxplot([dist_df[tp].values for tp in TIMEPOINTS], labels=TIMEPOINTS, showfliers=True)
ax.set_title("PERMDISP-style check: distances to timepoint centroid\n(Standardized 12-chirality feature space)")
ax.set_ylabel("Euclidean distance")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "permdisp_boxplot_full12.png"), dpi=300)
plt.close(fig)

# Null histogram
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.hist(F_perm, bins=40, alpha=0.8)
ax.axvline(F_obs, linewidth=2)
ax.set_title("Permutation test (within-subject) on dispersion differences\nNull distribution of F (distances)")
ax.set_xlabel("F under permutation")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "permdisp_null_hist_full12.png"), dpi=300)
plt.close(fig)

print("Done. Results saved in:", OUT_DIR)
print(global_df)
print(pair_df)
