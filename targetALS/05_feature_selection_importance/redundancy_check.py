import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5"
IN_DIR = os.path.join(BASE_DIR, "multiplexing_results")
OUT_DIR = os.path.join(IN_DIR, "redundancy_by_timepoint")
os.makedirs(OUT_DIR, exist_ok=True)

# Choose which extracted feature you want to use:
#   "features_sum5x5_" is usually more robust than max
PREFIX = "features_sum5x5_"   # or "features_max5x5_"

TIMEPOINTS = ["0h", "6h", "24h"]

def plot_corr_heatmap(R, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
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

def offdiag_summary(R):
    off = R[~np.eye(R.shape[0], dtype=bool)]
    return {
        "mean_abs_r_offdiag": float(np.mean(np.abs(off))),
        "median_abs_r_offdiag": float(np.median(np.abs(off))),
        "frac_pairs_r_gt_0p8": float(np.mean(np.abs(off) > 0.8)),
    }

summaries = []

for t in TIMEPOINTS:
    fp = os.path.join(IN_DIR, f"{PREFIX}{t}.csv")
    df = pd.read_csv(fp)

    # keep only chirality columns (everything except sample_id)
    chir_cols = [c for c in df.columns if c != "sample_id"]
    X = df[chir_cols].astype(float)

    # 12x12 Pearson correlation across the 39 samples
    R = X.corr(method="pearson").to_numpy()
    R_df = pd.DataFrame(R, index=chir_cols, columns=chir_cols)
    R_df.to_csv(os.path.join(OUT_DIR, f"corr_{t}.csv"))

    plot_corr_heatmap(
        R, chir_cols,
        title=f"Chirality correlation across samples ({t})",
        out_png=os.path.join(OUT_DIR, f"heatmap_corr_{t}.png")
    )

    s = offdiag_summary(R)
    s["timepoint"] = t
    summaries.append(s)

pd.DataFrame(summaries).to_csv(os.path.join(OUT_DIR, "redundancy_summary.csv"), index=False)

print("Done. Outputs in:", OUT_DIR)
