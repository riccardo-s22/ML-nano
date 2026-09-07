import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
OUT = "age_sex_confound_outputs"
df = pd.read_csv(os.path.join(OUT, "merged_dim477_age_sex.csv"))
R  = pd.read_csv(os.path.join(OUT, "per_dim_group_vs_age_sex.csv"))
pal = {"ALS": "#c0392b", "CTRL": "#2980b9"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) age distribution by group
for g in ["CTRL", "ALS"]:
    ax[0].hist(df.age[df.group == g], bins=np.arange(15, 80, 6), alpha=0.6,
               color=pal[g], label=g)
ax[0].set_xlabel("Age (years)"); ax[0].set_ylabel("count")
ax[0].set_title("(a) Age imbalance\nALS 58.0±10.8 vs CTRL 49.2±14.3  (p=0.026, d=0.69)")
ax[0].legend()

# (b) dim477 vs age, within-group fits
for g in ["CTRL", "ALS"]:
    m = df.group == g
    ax[1].scatter(df.age[m], df.dim477[m], c=pal[g], s=55, edgecolor="k", lw=.5, label=g)
    if m.sum() > 2:
        b = np.polyfit(df.age[m], df.dim477[m], 1)
        xs = np.array([df.age[m].min(), df.age[m].max()])
        r = stats.spearmanr(df.age[m], df.dim477[m])[0]
        ax[1].plot(xs, np.polyval(b, xs), color=pal[g], lw=1.6, ls="--",
                   label=f"{g}: rho={r:+.2f}")
ax[1].set_xlabel("Age (years)"); ax[1].set_ylabel("dim477")
ax[1].set_title("(b) dim477 vs age WITHIN group\n(flat within-group => age not the driver)")
ax[1].legend(fontsize=8)

# (c) per-dim: group-link vs within-group age-link
ax[2].scatter(R.r_group, R.r_age_wg, s=10, alpha=.4, color="grey")
d4 = R[R.dim == "dim477"].iloc[0]
ax[2].scatter(d4.r_group, d4.r_age_wg, s=120, color="#c0392b", edgecolor="k", zorder=5, label="dim477")
ax[2].axhline(0.4, ls=":", color="k", lw=.8); ax[2].axvline(0.4, ls=":", color="k", lw=.8)
ax[2].set_xlabel("|corr with GROUP|"); ax[2].set_ylabel("|corr with AGE| (within group)")
ax[2].set_title("(c) 512 latent dims\ngroup-linked dims are NOT age-linked (rho=-0.12)")
ax[2].legend()

fig.tight_layout(); fp = os.path.join(OUT, "age_sex_confound_summary.png")
fig.savefig(fp, dpi=150); print("saved", fp)
