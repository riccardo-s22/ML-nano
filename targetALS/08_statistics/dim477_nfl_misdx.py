"""
dim477 vs NfL — class separation & misdiagnosis check.

- dim477 (aggregated latent) from PLS/z_agg3.csv (has code, group, dim477)
- nfl_conc from PLS/early_slope.csv (code, nfl_conc)
Positive class = ALS.

Q1: how well do ALS/CTRL separate in (dim477, NfL) space?
Q2: do the dim477-misclassified samples have NfL values that "explain" the error?
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

PLS = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/PLS"
OUT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5/dim477_nfl_outputs"
os.makedirs(OUT, exist_ok=True)

# ---- load & merge ----
z = pd.read_csv(os.path.join(PLS, "z_agg3.csv"))[["code", "group", "dim477"]]
nfl = pd.read_csv(os.path.join(PLS, "early_slope.csv"))[["code", "nfl_conc"]]
df = z.merge(nfl, on="code", how="inner").dropna(subset=["dim477", "nfl_conc"]).reset_index(drop=True)
df["group"] = df["group"].astype(str).str.strip().str.upper()
df["y"] = (df["group"] == "ALS").astype(int)          # 1 = ALS
print(f"merged n={len(df)}  ALS={df.y.sum()}  CTRL={(1-df.y).sum()}")

y = df["y"].values
d477 = df["dim477"].values
lnfl = np.log10(df["nfl_conc"].values)                 # NfL is right-skewed -> log scale

# ---- univariate separation ----
def auc_oriented(score, y):
    a = roc_auc_score(y, score)
    return max(a, 1 - a), (a >= 0.5)   # AUC + whether higher score => ALS

auc_d, d_hi_als = auc_oriented(d477, y)
auc_n, n_hi_als = auc_oriented(lnfl, y)
t_d = stats.mannwhitneyu(d477[y == 1], d477[y == 0])
t_n = stats.mannwhitneyu(lnfl[y == 1], lnfl[y == 0])
rho, prho = stats.spearmanr(d477, df["nfl_conc"])
print(f"dim477 : AUC={auc_d:.3f}  MWU p={t_d.pvalue:.3g}  (ALS has {'higher' if d_hi_als else 'lower'} dim477)")
print(f"log10NfL: AUC={auc_n:.3f}  MWU p={t_n.pvalue:.3g}  (ALS has {'higher' if n_hi_als else 'lower'} NfL)")
print(f"dim477 vs NfL Spearman rho={rho:.3f}  p={prho:.3g}")

# ---- dim477-only classifier (Youden-optimal threshold) ----
# orient so larger "s" => more ALS
s = d477 if d_hi_als else -d477
fpr, tpr, thr = roc_curve(y, s)
j = np.argmax(tpr - fpr)
cut = thr[j]
pred_dim = (s >= cut).astype(int)
df["pred_dim477"] = pred_dim
df["correct_dim477"] = (pred_dim == y)
df["dim477_cut"] = cut if d_hi_als else -cut
mis = df[~df["correct_dim477"]].copy()
print(f"\ndim477 threshold (raw) = {df['dim477_cut'].iloc[0]:.3f}; "
      f"dim477-only accuracy = {df['correct_dim477'].mean():.3f}; "
      f"{len(mis)} misclassified")

# NfL median by class for "explains?" reference
als_nfl_med = np.median(df.loc[y == 1, "nfl_conc"])
ctrl_nfl_med = np.median(df.loc[y == 0, "nfl_conc"])
print(f"NfL median  ALS={als_nfl_med:.1f}  CTRL={ctrl_nfl_med:.1f}")
print("\n--- dim477-misclassified samples ---")
print(mis[["code", "group", "dim477", "nfl_conc"]].round(3).to_string(index=False))

df.to_csv(os.path.join(OUT, "dim477_nfl_merged.csv"), index=False)

# =================== PLOT ===================
fig, ax = plt.subplots(figsize=(8.2, 6.4))
pal = {"ALS": "#c0392b", "CTRL": "#2980b9"}
for g in ["CTRL", "ALS"]:
    m = df["group"] == g
    ax.scatter(df.loc[m, "dim477"], df.loc[m, "nfl_conc"], s=70, alpha=0.85,
               c=pal[g], edgecolor="black", linewidth=0.6, label=g, zorder=3)
# circle the dim477-misclassified samples
ax.scatter(mis["dim477"], mis["nfl_conc"], s=230, facecolors="none",
           edgecolors="#f39c12", linewidths=2.4, zorder=4,
           label="dim477-misclassified")
# dim477 decision boundary
ax.axvline(df["dim477_cut"].iloc[0], ls="--", color="grey", lw=1.4,
           label=f"dim477 cut = {df['dim477_cut'].iloc[0]:.2f}")
ax.set_yscale("log")
ax.set_xlabel("dim477 (aggregated latent)")
ax.set_ylabel("NfL concentration (pg/mL, log scale)")
ax.set_title(f"dim477 vs NfL  (n={len(df)})\n"
             f"dim477 AUC={auc_d:.2f} | NfL AUC={auc_n:.2f} | "
             f"dim477–NfL ρ={rho:.2f} (p={prho:.1g})")
ax.legend(loc="upper right", framealpha=0.95)
# annotate misclassified with their NfL
for _, r in mis.iterrows():
    ax.annotate(f"{r['nfl_conc']:.0f}", (r["dim477"], r["nfl_conc"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color="#7f5500")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dim477_vs_nfl_scatter.png"), dpi=150)
print("\nsaved:", os.path.join(OUT, "dim477_vs_nfl_scatter.png"))
