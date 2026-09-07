"""
Part 2 plotting — graphs for the controlled re-run (model_metrics_rerun/),
plus a side-by-side comparison with the existing models (Part 1).
Positive class = ALS.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score, precision_recall_curve,
                             average_precision_score)

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
OUT = os.path.join(ROOT, "model_metrics_rerun")
EX = os.path.join(ROOT, "model_metrics_eval")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
metrics = ["accuracy", "auc", "precision", "recall", "f1"]

# ---- load rerun results ----
d = np.load(os.path.join(OUT, "rerun_cache.npz"), allow_pickle=True)
oof_p = d["oof_p"]; y = d["y"].astype(int); codes = list(d["codes"])
oof_pred = (oof_p >= 0.5).astype(int)
curves = json.load(open(os.path.join(OUT, "curves.json")))
fold_df = pd.read_csv(os.path.join(OUT, "rerun_metrics.csv"))
fold_only = fold_df[pd.to_numeric(fold_df["fold"], errors="coerce").notna()].copy()
for m in metrics:
    fold_only[m] = pd.to_numeric(fold_only[m])

# ---- 1) learning curves (loss + acc), one panel per fold ----
fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=False)
for k in range(1, 6):
    h = curves[f"fold{k}"]
    ep = np.arange(1, len(h["tr_loss"]) + 1)
    ax = axes[0, k - 1]
    ax.plot(ep, h["tr_loss"], label="train", color="#2980b9")
    ax.plot(ep, h["va_loss"], label="val", color="#c0392b")
    ax.set_title(f"Fold {k} — loss"); ax.set_xlabel("epoch")
    if k == 1:
        ax.set_ylabel("total loss"); ax.legend()
    ax2 = axes[1, k - 1]
    ax2.plot(ep, h["tr_acc"], label="train", color="#2980b9")
    ax2.plot(ep, h["va_acc"], label="val", color="#c0392b")
    ax2.set_ylim(0, 105); ax2.set_title(f"Fold {k} — accuracy"); ax2.set_xlabel("epoch")
    if k == 1:
        ax2.set_ylabel("accuracy (%)"); ax2.legend()
fig.suptitle("Re-run — learning curves per fold", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "rerun_learning_curves.png")); plt.close(fig)

# ---- 2) ROC (OOF) ----
fpr, tpr, _ = roc_curve(y, oof_p); auc = roc_auc_score(y, oof_p)
plt.figure(figsize=(5.2, 5))
plt.plot(fpr, tpr, color="#27ae60", lw=2.2, label=f"OOF ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Re-run — OOF ROC (positive = ALS)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "rerun_roc_oof.png")); plt.close()

# ---- 3) PR (OOF) ----
prec, rec, _ = precision_recall_curve(y, oof_p); ap = average_precision_score(y, oof_p)
plt.figure(figsize=(5.2, 5))
plt.plot(rec, prec, color="#2c3e50", lw=2.2, label=f"OOF PR (AP = {ap:.3f})")
plt.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"baseline = {y.mean():.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Re-run — OOF Precision–Recall")
plt.legend(loc="lower left"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "rerun_pr_oof.png")); plt.close()

# ---- 4) confusion (OOF) ----
cm = confusion_matrix(y, oof_pred, labels=[0, 1])
plt.figure(figsize=(4.8, 4.4)); plt.imshow(cm, cmap="Greens")
plt.xticks([0, 1], ["CTRL", "ALS"]); plt.yticks([0, 1], ["CTRL", "ALS"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Re-run — OOF confusion\n(acc = {accuracy_score(y, oof_pred):.3f})")
thr = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thr else "black", fontsize=15)
plt.colorbar(fraction=0.046, pad=0.04); plt.tight_layout()
plt.savefig(os.path.join(OUT, "rerun_confusion_oof.png")); plt.close()

# ---- 5) per-fold bars ----
x = np.arange(5); w = 0.16
plt.figure(figsize=(9, 5))
for mi, m in enumerate(metrics):
    plt.bar(x + (mi - 2) * w, fold_only[m].values, w, label=m.capitalize(), color=colors[mi])
plt.xticks(x, [f"Fold {k}" for k in range(1, 6)]); plt.ylim(0, 1.08); plt.ylabel("Score")
plt.title("Re-run — per-fold validation metrics")
plt.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "rerun_perfold_bars.png"), bbox_inches="tight")
plt.close()

# ---- 6) comparison: existing vs re-run (mean +/- std across folds) ----
ex = pd.read_csv(os.path.join(EX, "existing_models_metrics.csv"))
ex_fold = ex[pd.to_numeric(ex["fold"], errors="coerce").notna()].copy()
for m in metrics:
    ex_fold[m] = pd.to_numeric(ex_fold[m])
ex_mean = [ex_fold[m].mean() for m in metrics]; ex_std = [ex_fold[m].std() for m in metrics]
rr_mean = [fold_only[m].mean() for m in metrics]; rr_std = [fold_only[m].std() for m in metrics]
x = np.arange(len(metrics)); w = 0.38
plt.figure(figsize=(8.5, 5))
plt.bar(x - w/2, ex_mean, w, yerr=ex_std, capsize=5, label="Existing (5_fold_models_original)",
        color="#7f8c8d", edgecolor="black")
plt.bar(x + w/2, rr_mean, w, yerr=rr_std, capsize=5, label="Re-run (controlled)",
        color="#16a085", edgecolor="black")
for i in range(len(metrics)):
    plt.text(x[i] - w/2, ex_mean[i] + ex_std[i] + 0.02, f"{ex_mean[i]:.2f}", ha="center", fontsize=9)
    plt.text(x[i] + w/2, rr_mean[i] + rr_std[i] + 0.02, f"{rr_mean[i]:.2f}", ha="center", fontsize=9)
plt.xticks(x, [m.capitalize() for m in metrics]); plt.ylim(0, 1.18)
plt.ylabel("Score (mean ± std over folds)")
plt.title("Existing vs Re-run — CV metric comparison")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "comparison_existing_vs_rerun.png"),
                                bbox_inches="tight"); plt.close()

print("Saved re-run figures to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print("  ", f)
