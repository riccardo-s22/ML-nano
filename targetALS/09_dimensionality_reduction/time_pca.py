import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5"
IN_DIR = os.path.join(BASE_DIR, "multiplexing_results")

PREFIX = "features_sum5x5_"   # or "features_max5x5_"
TIMEPOINTS = ["0h", "6h", "24h"]

OUT_DIR = os.path.join(IN_DIR, "pca_temporal")
os.makedirs(OUT_DIR, exist_ok=True)

def load_timepoint(t):
    fp = os.path.join(IN_DIR, f"{PREFIX}{t}.csv")
    df = pd.read_csv(fp)
    df["timepoint"] = t
    return df

# 1) Build long-form table: 117 rows = 39 samples x 3 timepoints
dfs = [load_timepoint(t) for t in TIMEPOINTS]
long_df = pd.concat(dfs, ignore_index=True)

chir_cols = [c for c in long_df.columns if c not in ["sample_id", "timepoint"]]
X = long_df[chir_cols].astype(float).to_numpy()

# OPTIONAL A: remove per-row common-mode (recommended if global brightness dominates)
# This makes PCA focus on the "relative chirality pattern" at each timepoint.
X_commonmode_removed = X - X.mean(axis=1, keepdims=True)

# Choose which matrix you want PCA on:
#   - X_raw: absolute intensities
#   - X_commonmode_removed: relative chirality pattern (often better for multiplexing interpretation)
X_for_pca = X_commonmode_removed

# OPTIONAL B: stabilize heavy-tailed features (only if values are strictly positive)
# X_for_pca = np.log1p(np.maximum(X_for_pca, 0))

# 2) Standardize features (critical for chirality comparability)
scaler = StandardScaler()
Xz = scaler.fit_transform(X_for_pca)

# 3) PCA
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xz)

pca_df = pd.DataFrame({
    "sample_id": long_df["sample_id"].values,
    "timepoint": long_df["timepoint"].values,
    "PC1": Z[:, 0],
    "PC2": Z[:, 1],
})

pca_df.to_csv(os.path.join(OUT_DIR, "pca_scores.csv"), index=False)

# 4) Scatter plot: points colored by timepoint
fig, ax = plt.subplots(figsize=(7.5, 6.0))
for t in TIMEPOINTS:
    sub = pca_df[pca_df["timepoint"] == t]
    ax.scatter(sub["PC1"], sub["PC2"], label=t, alpha=0.8)

ax.set_title(f"PCA of chirality features (2 PCs)\nExplained variance: "
             f"PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_scatter_by_timepoint.png"), dpi=300)
plt.close(fig)

# 5) Trajectory plot: connect each sample across timepoints
# Sort timepoints in chronological order
tp_order = {t: i for i, t in enumerate(TIMEPOINTS)}
pca_df["tp_idx"] = pca_df["timepoint"].map(tp_order)

fig, ax = plt.subplots(figsize=(7.5, 6.0))

# plot points (faint) and per-sample lines
for sid, g in pca_df.groupby("sample_id"):
    g = g.sort_values("tp_idx")
    ax.plot(g["PC1"].values, g["PC2"].values, alpha=0.25)
    ax.scatter(g["PC1"].values, g["PC2"].values, alpha=0.6, s=15)

# centroid arrows
centroids = pca_df.groupby("timepoint")[["PC1", "PC2"]].mean().loc[TIMEPOINTS]
ax.scatter(centroids["PC1"], centroids["PC2"], s=120, marker="X")

for i in range(len(TIMEPOINTS) - 1):
    x0, y0 = centroids.iloc[i]["PC1"], centroids.iloc[i]["PC2"]
    x1, y1 = centroids.iloc[i+1]["PC1"], centroids.iloc[i+1]["PC2"]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=2))

ax.set_title("PCA trajectories per sample + timepoint centroid drift")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pca_trajectories.png"), dpi=300)
plt.close(fig)

# 6) Save loadings (which chiralities drive PC1/PC2)
loadings = pd.DataFrame(pca.components_.T, index=chir_cols, columns=["PC1_loading", "PC2_loading"])
loadings.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"))

print("Done. Outputs in:", OUT_DIR)
