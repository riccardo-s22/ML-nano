"""
Cross-check: where do the exp5 CNN OOF misclassifications land in dim477-NfL space?
Uses the canonical saved models (5_fold_models_original) OOF predictions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
OUT = os.path.join(ROOT, "dim477_nfl_outputs")

m = pd.read_csv(os.path.join(OUT, "dim477_nfl_merged.csv"))
cut = m["dim477_cut"].iloc[0]
oof = pd.read_csv(os.path.join(ROOT, "model_metrics_eval/existing_models_oof_predictions.csv"))
d = m.merge(oof[["code", "P_ALS", "pred", "correct"]], on="code", how="left")

err = d[d["correct"] == False].copy()
pal = {"ALS": "#c0392b", "CTRL": "#2980b9"}

fig, ax = plt.subplots(figsize=(8.6, 6.6))
for g in ["CTRL", "ALS"]:
    s = d[d["group"] == g]
    ax.scatter(s["dim477"], s["nfl_conc"], s=72, c=pal[g], alpha=0.85,
               edgecolor="black", linewidth=0.6, label=g, zorder=3)
# ring CNN OOF errors
ax.scatter(err["dim477"], err["nfl_conc"], s=300, facecolors="none",
           edgecolors="#16a085", linewidths=2.6, zorder=4,
           label="CNN OOF misclassified")
ax.axvline(cut, ls="--", color="grey", lw=1.4, label=f"dim477 cut = {cut:.2f}")
ax.set_yscale("log")
ax.set_xlabel("dim477 (aggregated latent)")
ax.set_ylabel("NfL concentration (pg/mL, log scale)")
ax.set_title("exp5 CNN OOF errors in dim477–NfL space\n"
             "(ALS = low dim477 / high NfL)")
for _, r in err.iterrows():
    ax.annotate(f"{r['code'].split('.')[0]}\nP(ALS)={r['P_ALS']:.2f}\nNfL={r['nfl_conc']:.0f}",
                (r["dim477"], r["nfl_conc"]), textcoords="offset points",
                xytext=(10, -4), fontsize=7.5, color="#0b5345")
ax.legend(loc="upper right", framealpha=0.95)
fig.tight_layout()
p = os.path.join(OUT, "cnn_oof_errors_in_dim477_nfl.png")
fig.savefig(p, dpi=150)
print("saved", p)
print("\nCNN OOF errors and what dim477 says:")
err["dim477_pred"] = np.where(err["dim477"] < cut, "ALS", "CTRL")
print(err[["code", "group", "pred", "P_ALS", "dim477", "dim477_pred", "nfl_conc"]].round(3).to_string(index=False))
