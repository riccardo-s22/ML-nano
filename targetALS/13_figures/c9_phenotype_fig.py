import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
OUT = os.path.join(ROOT, "all_confounders_outputs")
df = pd.read_csv(os.path.join(OUT, "c9_across_models.csv"))  # id,y,P_ALS_fold1..5,P_ALS_mean5
folds = [f"P_ALS_fold{i}" for i in range(1, 6)]
ctl = df[df.y == 0]; als = df[df.y == 1]
C10, C30 = "#c0392b", "#e67e22"
# latent ALS-composite percentiles (from c9_phenotype.py stdout)
lat10 = [37, 26, 95, 11, 47]; lat30 = [32, 32, 5, 37, 42]
x = np.arange(1, 6)

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.3))

# ---- (a) P(ALS) across the 5 fold classifiers ----
a = ax[0]
# control cloud + ALS band per fold
for i, f in enumerate(folds):
    xc = np.random.default_rng(i).normal(x[i] - 0.12, 0.03, len(ctl))
    a.scatter(xc, ctl[f], s=22, color="#95a5a6", alpha=.6, zorder=2,
              label="other controls" if i == 0 else None)
    a.scatter(np.full(len(als), x[i] + 0.12), als[f], s=14, color="#2c3e50", alpha=.25, zorder=1,
              label="ALS subjects" if i == 0 else None)
r10 = df[df.id == 10].iloc[0][folds].values.astype(float)
r30 = df[df.id == 30].iloc[0][folds].values.astype(float)
a.plot(x, r10, "-o", color=C10, lw=2.4, ms=9, zorder=5, label="control 10 (59M, C9+)")
a.plot(x, r30, "-o", color=C30, lw=2.4, ms=9, zorder=5, label="control 30 (37M, C9+)")
a.axhline(0.5, ls="--", color="k", lw=1, label="ALS decision threshold")
a.annotate("only fold3 calls\ncontrol 10 ALS", (3, 0.95), xytext=(3.1, 0.72),
           fontsize=8.5, color=C10, arrowprops=dict(arrowstyle="->", color=C10))
a.set_xticks(x); a.set_xticklabels([f"fold{i}" for i in x])
a.set_ylabel("P(ALS)"); a.set_ylim(-0.05, 1.05)
a.set_title("(a) ALS probability across the 5 fold classifiers")
a.legend(fontsize=7.5, loc="center right", framealpha=.95)

# ---- (b) latent ALS-composite percentile across folds ----
b = ax[1]
b.axhspan(50, 100, color="#c0392b", alpha=.05)
b.plot(x, lat10, "-o", color=C10, lw=2.4, ms=9, label="control 10 (59M, C9+)")
b.plot(x, lat30, "-o", color=C30, lw=2.4, ms=9, label="control 30 (37M, C9+)")
b.axhline(50, ls=":", color="k", lw=1, label="control median")
b.annotate("dim477 space:\ncontrol 10 = 95th pct\n(most ALS-like control)", (3, 95),
           xytext=(2.5, 66), fontsize=8.5, color=C10,
           arrowprops=dict(arrowstyle="->", color=C10))
b.set_xticks(x); b.set_xticklabels([f"fold{i}" + ("\n(dim477)" if i == 3 else "") for i in x])
b.set_ylabel("latent ALS-composite\n(percentile among controls)"); b.set_ylim(0, 102)
b.set_title("(b) Position on ALS-discriminative latent features")
b.legend(fontsize=8, loc="upper right", framealpha=.95)

fig.suptitle("Do the two C9orf72+ controls show an ALS phenotype?  "
             "10 is ALS-like only in the fold3/dim477 representation; 30 is control-like everywhere",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fp = os.path.join(OUT, "c9_phenotype_across_models.png")
fig.savefig(fp, dpi=160); print("saved", fp)
