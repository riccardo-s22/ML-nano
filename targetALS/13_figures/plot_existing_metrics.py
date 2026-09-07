"""
Part 1 — Metrics + graphs for the EXISTING saved models in 5_fold_models_original/.

Uses cached predictions (model_metrics_eval/pred_cache.npz) and the canonical
seed-42 StratifiedKFold(5, shuffle=True) split (independently reproduced and
matched to attention_reports/*.csv). Positive class = ALS.

NOTE: model class 1 == CTRL (sklearn LabelEncoder, alphabetical ALS<CTRL),
so P(ALS) = 1 - prob_class1.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
OUT = os.path.join(ROOT, "model_metrics_eval")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# ---- load cache ----
d = np.load(os.path.join(OUT, "pred_cache.npz"), allow_pickle=True)
prob_c1 = d["prob"]                 # [5,39] = P(CTRL)
codes = list(d["codes"])
y = d["y"].astype(int)             # 1 = ALS
P = 1.0 - prob_c1                   # P(ALS), [5,39]

# ---- canonical seed-42 split (csv order) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(len(codes)), y))   # (train_idx, val_idx) per fold

# ---- OOF predictions: each subject scored by the model that held it out ----
oof_p = np.full(len(codes), np.nan)
oof_fold = np.full(len(codes), -1, dtype=int)
for k, (_, va) in enumerate(folds):
    for i in va:
        oof_p[i] = P[k, i]
        oof_fold[i] = k + 1
oof_pred = (oof_p >= 0.5).astype(int)

# ---- per-fold metrics ----
rows = []
for k, (_, va) in enumerate(folds):
    yk, pk = y[va], P[k, va]
    prk = (pk >= 0.5).astype(int)
    try:
        auc = roc_auc_score(yk, pk)
    except ValueError:
        auc = np.nan
    rows.append(dict(fold=k + 1, n=len(va),
                     accuracy=accuracy_score(yk, prk),
                     auc=auc,
                     precision=precision_score(yk, prk, zero_division=0),
                     recall=recall_score(yk, prk, zero_division=0),
                     f1=f1_score(yk, prk, zero_division=0)))
fold_df = pd.DataFrame(rows)

# OOF aggregate row
oof_row = dict(fold="OOF", n=len(y),
               accuracy=accuracy_score(y, oof_pred),
               auc=roc_auc_score(y, oof_p),
               precision=precision_score(y, oof_pred, zero_division=0),
               recall=recall_score(y, oof_pred, zero_division=0),
               f1=f1_score(y, oof_pred, zero_division=0))
mean_row = dict(fold="mean", n=np.nan,
                **{m: fold_df[m].mean() for m in ["accuracy", "auc", "precision", "recall", "f1"]})
std_row = dict(fold="std", n=np.nan,
               **{m: fold_df[m].std() for m in ["accuracy", "auc", "precision", "recall", "f1"]})
summary = pd.concat([fold_df, pd.DataFrame([mean_row, std_row, oof_row])], ignore_index=True)
summary.to_csv(os.path.join(OUT, "existing_models_metrics.csv"), index=False)

# per-subject OOF predictions
pred_df = pd.DataFrame({
    "code": codes, "true_label": ["ALS" if v == 1 else "CTRL" for v in y],
    "held_out_fold": oof_fold, "P_ALS": np.round(oof_p, 4),
    "pred": ["ALS" if v == 1 else "CTRL" for v in oof_pred],
    "correct": (oof_pred == y)})
pred_df.to_csv(os.path.join(OUT, "existing_models_oof_predictions.csv"), index=False)

print(summary.round(3).to_string(index=False))

# ================= PLOTS =================
ALS, CTRL = 1, 0

# 1) ROC (OOF)
fpr, tpr, _ = roc_curve(y, oof_p)
auc = roc_auc_score(y, oof_p)
plt.figure(figsize=(5.2, 5))
plt.plot(fpr, tpr, color="#c0392b", lw=2.2, label=f"OOF ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Existing models — OOF ROC (positive = ALS)")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "existing_roc_oof.png")); plt.close()

# 2) PR curve (OOF)
prec, rec, _ = precision_recall_curve(y, oof_p)
ap = average_precision_score(y, oof_p)
plt.figure(figsize=(5.2, 5))
plt.plot(rec, prec, color="#2c3e50", lw=2.2, label=f"OOF PR (AP = {ap:.3f})")
plt.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"baseline = {y.mean():.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Existing models — OOF Precision–Recall")
plt.legend(loc="lower left"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "existing_pr_oof.png")); plt.close()

# 3) Confusion matrix (OOF)
cm = confusion_matrix(y, oof_pred, labels=[CTRL, ALS])
plt.figure(figsize=(4.8, 4.4))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], ["CTRL", "ALS"]); plt.yticks([0, 1], ["CTRL", "ALS"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Existing models — OOF confusion\n(acc = {accuracy_score(y, oof_pred):.3f})")
thr = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thr else "black", fontsize=15)
plt.colorbar(fraction=0.046, pad=0.04); plt.tight_layout()
plt.savefig(os.path.join(OUT, "existing_confusion_oof.png")); plt.close()

# 4) Per-fold grouped bar chart
metrics = ["accuracy", "auc", "precision", "recall", "f1"]
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
x = np.arange(5); w = 0.16
plt.figure(figsize=(9, 5))
for mi, m in enumerate(metrics):
    plt.bar(x + (mi - 2) * w, fold_df[m].values, w, label=m.capitalize(), color=colors[mi])
plt.xticks(x, [f"Fold {k}" for k in range(1, 6)])
plt.ylim(0, 1.08); plt.ylabel("Score")
plt.title("Existing models — per-fold validation metrics")
plt.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "existing_perfold_bars.png"), bbox_inches="tight"); plt.close()

# 5) Summary bar with mean +/- std across folds
plt.figure(figsize=(6.5, 5))
means = [fold_df[m].mean() for m in metrics]
stds = [fold_df[m].std() for m in metrics]
plt.bar(metrics, means, yerr=stds, capsize=6,
        color=colors, edgecolor="black", alpha=0.85)
for i, (mn, sd) in enumerate(zip(means, stds)):
    plt.text(i, mn + sd + 0.02, f"{mn:.2f}", ha="center", fontsize=10)
plt.ylim(0, 1.12); plt.ylabel("Score (mean ± std over folds)")
plt.title("Existing models — CV metric summary")
plt.xticks(range(5), [m.capitalize() for m in metrics])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "existing_summary_bars.png")); plt.close()

print("\nSaved figures + CSVs to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
