import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list

# Folder containing your corr_residual_*.csv files
IN_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\multiplexing_results\redundancy_by_timepoint"
OUT_DIR = os.path.join(IN_DIR, "residual_heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

TIMEPOINTS = ["0h", "6h", "24h"]

def plot_heatmap(R, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(R, vmin=-1, vmax=1, aspect="equal")  # fixed scale for comparability
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def cluster_order(R):
    # Convert correlation to distance; guard against NaNs
    R = np.nan_to_num(R, nan=0.0)
    D = 1 - R
    np.fill_diagonal(D, 0.0)
    iu = np.triu_indices_from(D, k=1)
    condensed = D[iu]
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return order

for t in TIMEPOINTS:
    fp = os.path.join(IN_DIR, f"corr_residual_{t}.csv")
    df = pd.read_csv(fp, index_col=0)

    labels = list(df.columns)
    R = df.to_numpy(dtype=float)

    # 1) Standard residual heatmap (original ordering)
    plot_heatmap(
        R, labels,
        title=f"Residual chirality correlation across samples ({t})",
        out_png=os.path.join(OUT_DIR, f"heatmap_corr_residual_{t}.png")
    )

    # 2) Clustered residual heatmap (reordered for structure)
    order = cluster_order(R)
    R2 = R[np.ix_(order, order)]
    labels2 = [labels[i] for i in order]
    plot_heatmap(
        R2, labels2,
        title=f"Clustered residual chirality correlation ({t})",
        out_png=os.path.join(OUT_DIR, f"clustermap_corr_residual_{t}.png")
    )

print("Done. Saved to:", OUT_DIR)
