#!/usr/bin/env python3
"""
recursive_loocv_feature_selection.py
======================================

Recursive forward feature selection with LOOCV at every step,
plus final validation of the optimal feature set.

Pipeline:
  1. f-ANOVA rank all features (full data)
  2. Spearman collinearity filter (|rho| < 0.7, greedy forward)
  3. For each feature added (in rank order):
       a) Run FULL LOOCV (scaler fit on train only) -> AUC, BalAcc, Sens, Spec
       b) Compute multivariate power (LDA projection, Cohen's d)
       c) Record everything
  4. Pick optimal feature set (max AUC with power >= 90%)
  5. FINAL VALIDATION of optimal set:
       a) LOOCV with per-sample predictions
       b) Repeated Stratified 5-Fold CV (10 repeats)
       c) Bootstrap 95% CI on LOOCV AUC
       d) Permutation test (1000 shuffles)
  6. Save figures (PNG) + CSV results
"""

import os, sys, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import spearmanr, nct as nct_dist
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================================
# Power helpers
# ============================================================================

def cohens_d(g0, g1):
    n0, n1 = len(g0), len(g1)
    v0 = np.var(g0, ddof=1) if n0 > 1 else 0.0
    v1 = np.var(g1, ddof=1) if n1 > 1 else 0.0
    sp = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1))
    return float(abs(g0.mean() - g1.mean()) / sp) if sp > 1e-15 else 0.0


def power_from_d(d, n0, n1, alpha=0.05):
    if d < 1e-10:
        return alpha
    from scipy.stats import t as t_dist
    df = n0 + n1 - 2
    ncp = d * np.sqrt(n0 * n1 / (n0 + n1))
    tc = t_dist.ppf(1 - alpha / 2, df)
    return float(np.clip(1 - nct_dist.cdf(tc, df, ncp) + nct_dist.cdf(-tc, df, ncp), 0, 1))


def multivariate_power(X, y, feat_idx, alpha=0.05):
    """Power via LDA projection of the selected features."""
    if len(feat_idx) == 0:
        return 0.0
    Xs = X[:, feat_idx]
    m0, m1 = y == 0, y == 1
    g0, g1 = Xs[m0], Xs[m1]
    if len(feat_idx) == 1:
        return power_from_d(cohens_d(g0[:, 0], g1[:, 0]), m0.sum(), m1.sum(), alpha)
    mu0, mu1 = g0.mean(0), g1.mean(0)
    c0 = np.cov(g0, rowvar=False) if g0.shape[0] > 1 else np.eye(len(feat_idx)) * 1e-10
    c1 = np.cov(g1, rowvar=False) if g1.shape[0] > 1 else np.eye(len(feat_idx)) * 1e-10
    Sw = (c0 * (len(g0) - 1) + c1 * (len(g1) - 1)) / (len(g0) + len(g1) - 2)
    Sw += np.eye(Sw.shape[0]) * 1e-8
    try:
        w = np.linalg.solve(Sw, mu1 - mu0)
    except np.linalg.LinAlgError:
        w = mu1 - mu0
    return power_from_d(cohens_d(g0 @ w, g1 @ w), m0.sum(), m1.sum(), alpha)


# ============================================================================
# Classifiers
# ============================================================================

def get_clf(name):
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42)
    elif name == "LR":
        return LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    elif name == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
    raise ValueError(name)


# ============================================================================
# LOOCV
# ============================================================================

def run_loocv(X, y, feat_idx, clf_name):
    """Full LOOCV with scaler fit on train only."""
    N = len(y)
    scores = np.zeros(N)
    preds = np.zeros(N, dtype=int)
    Xs = X[:, feat_idx]

    for i in range(N):
        tr = np.ones(N, dtype=bool); tr[i] = False
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xs[tr])
        Xte = sc.transform(Xs[i:i+1])
        clf = get_clf(clf_name)
        clf.fit(Xtr, y[tr])
        scores[i] = clf.predict_proba(Xte)[0, 1]
        preds[i] = clf.predict(Xte)[0]

    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan
    ba = balanced_accuracy_score(y, preds)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(auc), "bal_acc": float(ba),
        "sens": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "spec": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "y_scores": scores, "y_preds": preds,
    }


# ============================================================================
# Recursive forward selection
# ============================================================================

def recursive_forward_selection(
    X, y, feature_names,
    clf_names=("SVM", "LR", "RF"),
    spearman_thresh=0.7,
    max_features=25,
    verbose=True,
):
    n_feat = X.shape[1]

    # f-ANOVA
    var = np.var(X, axis=0)
    F = np.zeros(n_feat); P = np.ones(n_feat)
    ok = var > 1e-15
    if ok.any():
        Fo, Po = f_classif(X[:, ok], y)
        F[ok] = np.nan_to_num(Fo, nan=0.0)
        P[ok] = np.nan_to_num(Po, nan=1.0)

    m0, m1 = y == 0, y == 1
    n0, n1 = m0.sum(), m1.sum()
    d_arr = np.array([cohens_d(X[m0, j], X[m1, j]) for j in range(n_feat)])
    pwr_arr = np.array([power_from_d(d_arr[j], n0, n1) for j in range(n_feat)])
    rank = np.argsort(-F)

    ranking_df = pd.DataFrame({
        "feature": [feature_names[i] for i in rank],
        "f_score": F[rank], "p_val": P[rank],
        "cohens_d": d_arr[rank], "univar_power": pwr_arr[rank],
    })

    if verbose:
        print(f"\n  f-ANOVA top 20 / {n_feat}:")
        for i, (_, r) in enumerate(ranking_df.head(20).iterrows()):
            print(f"  {i+1:3d}  F={r.f_score:8.2f}  p={r.p_val:.4f}  "
                  f"d={r.cohens_d:.3f}  pwr={r.univar_power:.3f}  {r.feature}")

    selected = []
    steps = []

    for ri, feat_idx in enumerate(rank):
        feat_idx = int(feat_idx)
        if len(selected) >= max_features:
            break

        xc = X[:, feat_idx]
        skip = False
        blocker = ""
        for si in selected:
            rho, _ = spearmanr(xc, X[:, si])
            if not np.isnan(rho) and abs(rho) > spearman_thresh:
                skip = True
                blocker = feature_names[si]
                break
        if skip:
            if verbose and ri < 50:
                print(f"    SKIP  {feature_names[feat_idx]:<50s} "
                      f"(|rho|>{spearman_thresh:.1f} with {blocker})")
            continue

        selected.append(feat_idx)
        k = len(selected)
        pwr = multivariate_power(X, y, selected)

        step = {
            "step": k,
            "feature_added": feature_names[feat_idx],
            "f_score": float(F[feat_idx]),
            "p_val": float(P[feat_idx]),
            "cohens_d_univar": float(d_arr[feat_idx]),
            "power": pwr,
            "n_features": k,
            "feature_set": [feature_names[s] for s in selected],
        }

        if verbose:
            print(f"\n  -- Step {k}: + {feature_names[feat_idx]}")
            print(f"     power = {pwr:.4f}")

        for cn in clf_names:
            res = run_loocv(X, y, selected, cn)
            step[f"{cn}_auc"] = res["auc"]
            step[f"{cn}_balacc"] = res["bal_acc"]
            step[f"{cn}_sens"] = res["sens"]
            step[f"{cn}_spec"] = res["spec"]
            step[f"{cn}_scores"] = res["y_scores"]
            if verbose:
                print(f"     {cn:3s}: AUC={res['auc']:.3f}  BA={res['bal_acc']:.3f}  "
                      f"Se={res['sens']:.3f}  Sp={res['spec']:.3f}")

        steps.append(step)

    return {
        "steps": steps,
        "selected_indices": selected,
        "selected_names": [feature_names[i] for i in selected],
        "ranking_df": ranking_df,
    }


def pick_optimal(steps, clf_name="LR", power_target=0.90):
    auc_key = f"{clf_name}_auc"
    powered = [s for s in steps if not np.isnan(s["power"]) and s["power"] >= power_target]
    pool = powered if powered else steps
    best = max(pool, key=lambda s: s[auc_key])
    return best["step"]


# ============================================================================
# Final validation
# ============================================================================

def final_loocv(X, y, feat_idx, clf_names, sample_ids=None):
    N = len(y)
    Xs = X[:, feat_idx]
    results = {}
    all_scores = {}

    for cn in clf_names:
        scores = np.zeros(N)
        preds = np.zeros(N, dtype=int)
        for i in range(N):
            tr = np.ones(N, dtype=bool); tr[i] = False
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xs[tr])
            Xte = sc.transform(Xs[i:i+1])
            clf = get_clf(cn)
            clf.fit(Xtr, y[tr])
            scores[i] = clf.predict_proba(Xte)[0, 1]
            preds[i] = clf.predict(Xte)[0]
        auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan
        ba = balanced_accuracy_score(y, preds)
        acc = accuracy_score(y, preds)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        results[cn] = {
            "auc": float(auc), "bal_acc": float(ba), "acc": float(acc),
            "sens": float(tp/(tp+fn)) if (tp+fn) else 0.0,
            "spec": float(tn/(tn+fp)) if (tn+fp) else 0.0,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "y_scores": scores, "y_preds": preds,
        }
        all_scores[cn] = scores

    rows = []
    for i in range(N):
        row = {"sample": sample_ids[i] if sample_ids is not None else i,
               "true_label": "ALS" if y[i] == 1 else "CTRL"}
        for cn in clf_names:
            row[f"{cn}_prob_ALS"] = float(f"{all_scores[cn][i]:.4f}")
            row[f"{cn}_pred"] = "ALS" if results[cn]["y_preds"][i] == 1 else "CTRL"
            row[f"{cn}_correct"] = bool(results[cn]["y_preds"][i] == y[i])
        rows.append(row)
    sample_df = pd.DataFrame(rows)
    return results, sample_df


def repeated_stratified_kfold(X, y, feat_idx, clf_names,
                               n_splits=5, n_repeats=10, random_state=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=random_state)
    Xs = X[:, feat_idx]
    results = {}

    for cn in clf_names:
        fold_aucs, fold_bas, fold_sens, fold_specs = [], [], [], []
        for train_idx, test_idx in rskf.split(Xs, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xs[train_idx])
            Xte = sc.transform(Xs[test_idx])
            clf = get_clf(cn)
            clf.fit(Xtr, y[train_idx])
            probs = clf.predict_proba(Xte)[:, 1]
            preds = clf.predict(Xte)
            if len(np.unique(y[test_idx])) == 2:
                fold_aucs.append(roc_auc_score(y[test_idx], probs))
            fold_bas.append(balanced_accuracy_score(y[test_idx], preds))
            tn, fp, fn, tp = confusion_matrix(y[test_idx], preds, labels=[0, 1]).ravel()
            fold_sens.append(tp/(tp+fn) if (tp+fn) else 0.0)
            fold_specs.append(tn/(tn+fp) if (tn+fp) else 0.0)

        results[cn] = {
            "auc_mean": float(np.mean(fold_aucs)), "auc_std": float(np.std(fold_aucs)),
            "balacc_mean": float(np.mean(fold_bas)), "balacc_std": float(np.std(fold_bas)),
            "sens_mean": float(np.mean(fold_sens)), "sens_std": float(np.std(fold_sens)),
            "spec_mean": float(np.mean(fold_specs)), "spec_std": float(np.std(fold_specs)),
            "n_folds": len(fold_aucs),
            "fold_aucs": np.array(fold_aucs),
        }
    return results


def bootstrap_auc_ci(y_true, y_scores, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    N = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    aucs = np.array(aucs)
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return float(lo), float(hi), aucs


def permutation_test(X, y, feat_idx, clf_name, n_perms=1000, observed_auc=None):
    rng = np.random.RandomState(42)
    nulls = []
    for _ in range(n_perms):
        yp = rng.permutation(y)
        r = run_loocv(X, yp, feat_idx, clf_name)
        nulls.append(r["auc"])
    nulls = np.array(nulls)
    pval = float(np.mean(nulls >= (observed_auc or 0.5)))
    return {"p_value": pval, "null_aucs": nulls,
            "null_mean": float(nulls.mean()), "null_std": float(nulls.std())}


# ============================================================================
# Figures
# ============================================================================

CLF_COLORS = {"SVM": "#e74c3c", "LR": "#2980b9", "RF": "#27ae60"}


def fig_auc_power_curve(steps, clf_names, optimal_step, outpath):
    ns = [s["step"] for s in steps]
    pwrs = [s["power"] for s in steps]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    for cn in clf_names:
        aucs = [s[f"{cn}_auc"] for s in steps]
        ax1.plot(ns, aucs, "-o", color=CLF_COLORS.get(cn, "gray"), markersize=4,
                 label=f"{cn} AUC", linewidth=1.5)

    ax2.plot(ns, pwrs, "k--s", markersize=4, alpha=0.5, label="Power", linewidth=1.2)
    ax2.axhline(0.90, color="black", linestyle=":", alpha=0.3, label="90% power")
    ax2.fill_between(ns, 0, pwrs, alpha=0.04, color="black")

    ax1.axvline(optimal_step, color="gold", linewidth=2.5, alpha=0.7,
                label=f"Optimal (step {optimal_step})")

    ax1.set_xlabel("Number of features", fontsize=12)
    ax1.set_ylabel("AUC (LOOCV)", fontsize=12)
    ax2.set_ylabel("Statistical Power", fontsize=12)
    ax1.set_ylim(0.3, 1.02)
    ax2.set_ylim(0.0, 1.05)
    ax1.set_xticks(ns)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    ax1.set_title("Recursive Forward Selection: LOOCV AUC & Power vs # Features", fontsize=13)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_final_roc(y, loocv_res, clf_names, boot_cis, outpath):
    fig, ax = plt.subplots(figsize=(6, 6))
    for cn in clf_names:
        fpr, tpr, _ = roc_curve(y, loocv_res[cn]["y_scores"])
        lo, hi = boot_cis[cn]
        ax.plot(fpr, tpr, color=CLF_COLORS.get(cn, "gray"), linewidth=2,
                label=f"{cn}  AUC={loocv_res[cn]['auc']:.3f}  [{lo:.3f}-{hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Final Validation: LOOCV ROC (95% Bootstrap CI)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_permutation(perm, observed_auc, clf_name, outpath):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(perm["null_aucs"], bins=50, alpha=0.7, color="#bdc3c7",
            edgecolor="black", linewidth=0.5)
    ax.axvline(observed_auc, color="red", linewidth=2.5,
               label=f"Observed AUC = {observed_auc:.3f}")
    ax.set_xlabel("AUC (permuted labels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Permutation Test ({clf_name}): p = {perm['p_value']:.4f}  "
                 f"(null = {perm['null_mean']:.3f} +/- {perm['null_std']:.3f})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_kfold_distributions(rskf_res, clf_names, outpath):
    fig, axes = plt.subplots(1, len(clf_names), figsize=(4.5 * len(clf_names), 4), sharey=True)
    if len(clf_names) == 1:
        axes = [axes]
    for ax, cn in zip(axes, clf_names):
        aucs = rskf_res[cn]["fold_aucs"]
        ax.hist(aucs, bins=20, alpha=0.7, color=CLF_COLORS.get(cn, "gray"),
                edgecolor="black", linewidth=0.5)
        ax.axvline(rskf_res[cn]["auc_mean"], color="black", linewidth=2, linestyle="--",
                   label=f"Mean = {rskf_res[cn]['auc_mean']:.3f}")
        ax.set_xlabel("AUC", fontsize=11)
        ax.set_title(f"{cn}\n{rskf_res[cn]['auc_mean']:.3f} +/- {rskf_res[cn]['auc_std']:.3f}",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Count", fontsize=11)
    fig.suptitle("Repeated Stratified 5-Fold CV (10 repeats) - AUC Distribution",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_validation_summary(loocv_res, rskf_res, boot_cis, perm_results,
                            clf_names, feat_names, power, outpath):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [1, 1.2]})

    # Top: feature list
    ax = axes[0]
    ax.axis("off")
    feat_text = "\n".join([f"  {i+1}. {f}" for i, f in enumerate(feat_names)])
    ax.text(0.02, 0.95,
            f"Optimal Feature Set ({len(feat_names)} features, "
            f"Power = {power:.3f})\n{feat_text}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))

    # Bottom: metrics table
    ax = axes[1]
    ax.axis("off")

    col_labels = ["Classifier",
                  "LOOCV\nAUC", "LOOCV\nBal Acc", "LOOCV\nSens", "LOOCV\nSpec",
                  "AUC\n95% CI",
                  "5F-CV\nAUC", "5F-CV\nBal Acc",
                  "Perm\np-val"]
    rows = []
    for cn in clf_names:
        lr = loocv_res[cn]
        kr = rskf_res[cn]
        lo, hi = boot_cis[cn]
        pp = perm_results.get(cn, {}).get("p_value", "")
        pp_str = f"{pp:.4f}" if isinstance(pp, float) else ""
        rows.append([
            cn,
            f"{lr['auc']:.3f}", f"{lr['bal_acc']:.3f}",
            f"{lr['sens']:.3f}", f"{lr['spec']:.3f}",
            f"[{lo:.3f}-{hi:.3f}]",
            f"{kr['auc_mean']:.3f}+/-{kr['auc_std']:.3f}",
            f"{kr['balacc_mean']:.3f}+/-{kr['balacc_std']:.3f}",
            pp_str,
        ])

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        elif ri % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    ax.set_title("Final Validation Summary", fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_sample_predictions(sample_df, clf_names, outpath):
    n = len(sample_df)
    fig, ax = plt.subplots(figsize=(3 + len(clf_names) * 2.2, max(6, n * 0.28)))
    ax.axis("off")

    col_labels = ["Sample", "True"]
    for cn in clf_names:
        col_labels += [f"{cn}\nP(ALS)", f"{cn}\nPred"]

    cell_text = []
    cell_colors = []
    for _, row in sample_df.iterrows():
        r = [str(row["sample"]), row["true_label"]]
        c = ["white", "#d5f5e3" if row["true_label"] == "CTRL" else "#fadbd8"]
        for cn in clf_names:
            r.append(f"{row[f'{cn}_prob_ALS']:.3f}")
            r.append(row[f"{cn}_pred"])
            correct = row[f"{cn}_correct"]
            c.append("white")
            c.append("#d5f5e3" if correct else "#f1948a")
        cell_text.append(r)
        cell_colors.append(c)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellColours=cell_colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.15)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7)

    ax.set_title("Per-Sample LOOCV Predictions (red = misclassified)", fontsize=11, pad=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors_csv", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output_dir", default="./recursive_loocv_results")
    parser.add_argument("--spearman_thresh", type=float, default=0.7)
    parser.add_argument("--power_target", type=float, default=0.90)
    parser.add_argument("--max_features", type=int, default=25)
    parser.add_argument("--n_perms", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RECURSIVE LOOCV FORWARD FEATURE SELECTION")
    print("=" * 70)

    # Load
    desc_df = pd.read_csv(args.descriptors_csv)
    labels_df = pd.read_csv(args.labels_csv)
    labels_df["code"] = labels_df["code"].str.replace(".", "_", regex=False)
    merged = desc_df.merge(labels_df, on="code", how="inner")

    feat_cols = [c for c in desc_df.columns if c != "code"]
    X = merged[feat_cols].values.astype(np.float64)
    y = (merged["group"] == "ALS").astype(int).values
    sample_ids = merged["code"].values
    X[np.isnan(X) | np.isinf(X)] = 0.0

    # Drop zero-variance
    var = np.var(X, axis=0)
    keep = var > 1e-15
    if (~keep).sum():
        print(f"  Dropped {(~keep).sum()} zero-var features")
    X = X[:, keep]
    feat_cols = [feat_cols[i] for i in range(len(feat_cols)) if keep[i]]

    print(f"  {X.shape[0]} samples ({(y==1).sum()} ALS, {(y==0).sum()} CTRL)")
    print(f"  {X.shape[1]} features")

    clf_names = ["SVM", "LR", "RF"]

    # ==================================================================
    # PHASE 1: RECURSIVE FORWARD SELECTION
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: RECURSIVE FORWARD SELECTION")
    print("=" * 70)

    fwd = recursive_forward_selection(
        X, y, feat_cols,
        clf_names=clf_names,
        spearman_thresh=args.spearman_thresh,
        max_features=args.max_features,
        verbose=True,
    )

    steps = fwd["steps"]

    # Save step CSV
    step_rows = []
    for s in steps:
        row = {k: v for k, v in s.items()
               if k not in ("feature_set",) and not k.endswith("_scores")}
        row["feature_set"] = " | ".join(s["feature_set"])
        step_rows.append(row)
    pd.DataFrame(step_rows).to_csv(outdir / "steps.csv", index=False)
    fwd["ranking_df"].to_csv(outdir / "fanova_ranking.csv", index=False)

    # Pick optimal
    optimal = pick_optimal(steps, clf_name="LR", power_target=args.power_target)
    opt_step = steps[optimal - 1]
    sel_idx = fwd["selected_indices"][:optimal]
    sel_names = opt_step["feature_set"]

    print(f"\n{'='*70}")
    print(f"OPTIMAL: step {optimal}, {optimal} features, "
          f"power={opt_step['power']:.3f}")
    for cn in clf_names:
        print(f"  {cn}: AUC={opt_step[f'{cn}_auc']:.3f}  "
              f"BA={opt_step[f'{cn}_balacc']:.3f}")
    print(f"Features: {sel_names}")
    print(f"{'='*70}")

    # Fig 1: AUC + power curve
    fig_auc_power_curve(steps, clf_names, optimal,
                        outdir / "fig1_auc_power_curve.png")

    # ==================================================================
    # PHASE 2: FINAL VALIDATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FINAL VALIDATION OF OPTIMAL FEATURE SET")
    print("=" * 70)

    # 2a. LOOCV
    print("\n  [2a] Final LOOCV with per-sample predictions...")
    loocv_res, sample_pred_df = final_loocv(X, y, sel_idx, clf_names,
                                             sample_ids=sample_ids)
    for cn in clf_names:
        r = loocv_res[cn]
        print(f"    {cn}: AUC={r['auc']:.3f}  BA={r['bal_acc']:.3f}  "
              f"Se={r['sens']:.3f}  Sp={r['spec']:.3f}  "
              f"(TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']})")
    sample_pred_df.to_csv(outdir / "final_loocv_predictions.csv", index=False)

    # 2b. Bootstrap CI
    print("\n  [2b] Bootstrap 95% CI on LOOCV AUC (2000 resamples)...")
    boot_cis = {}
    for cn in clf_names:
        lo, hi, _ = bootstrap_auc_ci(y, loocv_res[cn]["y_scores"])
        boot_cis[cn] = (lo, hi)
        print(f"    {cn}: AUC = {loocv_res[cn]['auc']:.3f}  "
              f"95% CI [{lo:.3f} - {hi:.3f}]")

    # 2c. Repeated Stratified 5-Fold CV
    print("\n  [2c] Repeated Stratified 5-Fold CV (10 repeats)...")
    rskf_res = repeated_stratified_kfold(X, y, sel_idx, clf_names)
    for cn in clf_names:
        r = rskf_res[cn]
        print(f"    {cn}: AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}  "
              f"BA={r['balacc_mean']:.3f}+/-{r['balacc_std']:.3f}  "
              f"Se={r['sens_mean']:.3f}+/-{r['sens_std']:.3f}  "
              f"Sp={r['spec_mean']:.3f}+/-{r['spec_std']:.3f}")

    # 2d. Permutation test for all classifiers
    perm_results = {}
    for cn in clf_names:
        obs = loocv_res[cn]["auc"]
        print(f"\n  [2d] Permutation test ({args.n_perms} shuffles, {cn})...")
        perm = permutation_test(X, y, sel_idx, cn,
                                n_perms=args.n_perms, observed_auc=obs)
        perm_results[cn] = perm
        print(f"    p = {perm['p_value']:.4f}  "
              f"(null = {perm['null_mean']:.3f} +/- {perm['null_std']:.3f})")

    # Save validation summary CSV
    val_rows = []
    for cn in clf_names:
        lr = loocv_res[cn]
        kr = rskf_res[cn]
        lo, hi = boot_cis[cn]
        val_rows.append({
            "classifier": cn,
            "loocv_auc": lr["auc"], "loocv_balacc": lr["bal_acc"],
            "loocv_sens": lr["sens"], "loocv_spec": lr["spec"],
            "loocv_tp": lr["tp"], "loocv_tn": lr["tn"],
            "loocv_fp": lr["fp"], "loocv_fn": lr["fn"],
            "bootstrap_auc_lo": lo, "bootstrap_auc_hi": hi,
            "rskf_auc_mean": kr["auc_mean"], "rskf_auc_std": kr["auc_std"],
            "rskf_balacc_mean": kr["balacc_mean"], "rskf_balacc_std": kr["balacc_std"],
            "rskf_sens_mean": kr["sens_mean"], "rskf_sens_std": kr["sens_std"],
            "rskf_spec_mean": kr["spec_mean"], "rskf_spec_std": kr["spec_std"],
            "perm_p": perm_results[cn]["p_value"],
            "perm_null_mean": perm_results[cn]["null_mean"],
            "perm_null_std": perm_results[cn]["null_std"],
        })
    pd.DataFrame(val_rows).to_csv(outdir / "final_validation_summary.csv", index=False)

    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    fig_final_roc(y, loocv_res, clf_names, boot_cis,
                  outdir / "fig2_final_roc.png")

    best_cn = max(clf_names, key=lambda c: loocv_res[c]["auc"])
    fig_permutation(perm_results[best_cn], loocv_res[best_cn]["auc"], best_cn,
                    outdir / "fig3_permutation.png")

    fig_kfold_distributions(rskf_res, clf_names,
                            outdir / "fig4_kfold_distributions.png")

    fig_validation_summary(loocv_res, rskf_res, boot_cis, perm_results,
                           clf_names, sel_names, opt_step["power"],
                           outdir / "fig5_validation_summary.png")

    fig_sample_predictions(sample_pred_df, clf_names,
                           outdir / "fig6_sample_predictions.png")

    print(f"\n{'='*70}")
    print(f"ALL DONE. Outputs in: {outdir}/")
    print(f"  CSVs: steps.csv, fanova_ranking.csv, "
          f"final_loocv_predictions.csv, final_validation_summary.csv")
    print(f"  Figs: fig1_auc_power_curve.png, fig2_final_roc.png, "
          f"fig3_permutation.png, fig4_kfold_distributions.png, "
          f"fig5_validation_summary.png, fig6_sample_predictions.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
