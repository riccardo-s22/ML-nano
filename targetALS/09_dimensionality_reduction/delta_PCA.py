import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

IN_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\multiplexing_results"
PREFIX = "features_sum5x5_"  # or features_max5x5_
TIMEPOINTS = ["0h","6h","24h"]

df0 = pd.read_csv(os.path.join(IN_DIR, f"{PREFIX}0h.csv"))
df6 = pd.read_csv(os.path.join(IN_DIR, f"{PREFIX}6h.csv"))
df24 = pd.read_csv(os.path.join(IN_DIR, f"{PREFIX}24h.csv"))

chir_cols = [c for c in df0.columns if c != "sample_id"]

# align rows by sample_id (critical)
df0 = df0.set_index("sample_id").loc[df6["sample_id"]].reset_index()
df6 = df6.set_index("sample_id").loc[df0["sample_id"]].reset_index()
df24 = df24.set_index("sample_id").loc[df0["sample_id"]].reset_index()

X0 = df0[chir_cols].astype(float).to_numpy()
X6 = df6[chir_cols].astype(float).to_numpy()
X24 = df24[chir_cols].astype(float).to_numpy()

d6  = X6  - X0
d24 = X24 - X0

# Option A: PCA on stacked deltas (captures two-phase kinetics)
X = np.hstack([d6, d24])
feat_names = [f"{c}_d6" for c in chir_cols] + [f"{c}_d24" for c in chir_cols]

# Standardize
Xz = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xz)

out = pd.DataFrame({"sample_id": df0["sample_id"], "PC1": Z[:,0], "PC2": Z[:,1]})
out.to_csv(os.path.join(IN_DIR, "pca_delta_scores.csv"), index=False)

# Plot
fig, ax = plt.subplots(figsize=(7.5,6))
ax.scatter(out["PC1"], out["PC2"], alpha=0.8)
ax.set_title(f"PCA on delta features (Δ6 and Δ24)\nExplained: PC1={pca.explained_variance_ratio_[0]:.3f}, "
             f"PC2={pca.explained_variance_ratio_[1]:.3f}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(os.path.join(IN_DIR, "pca_delta_scatter.png"), dpi=300)
plt.close(fig)

# Loadings
load = pd.DataFrame(pca.components_.T, index=feat_names, columns=["PC1_loading","PC2_loading"])
load.to_csv(os.path.join(IN_DIR, "pca_delta_loadings.csv"))
print("Saved delta PCA outputs.")
