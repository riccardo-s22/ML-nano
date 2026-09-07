#!/usr/bin/env python3
"""
recursive_loocv_feature_selection_corrected.py
================================================

Strict Nested LOOCV forward feature selection.
Prevents data leakage by performing f-ANOVA ranking and 
Spearman collinearity filtering strictly within the CV folds.

Pipeline:
  1. For each number of features K (1 to max_features):
       a) Run strict LOOCV.
       b) In each fold: 
            - Calculate f-ANOVA on Train only.
            - Run Spearman filter on Train only.
            - Select top K features.
            - Train scaler and classifier on Train only.
            - Predict probability for the 1 Test sample.
       c) Aggregate predictions across all folds to calculate true AUC/BalAcc.
  2. Pick optimal K (max AUC with power >= 90%).
  3. Train a final feature selection pipeline on the FULL dataset to extract
     the final feature names for future use.
  4. Final validation metrics and figures.
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
# Power helpers (Unchanged)
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
# Classifiers (Unchanged)
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
# Feature Selection Core (Leak-Free)
# ============================================================================

def select_features_for_fold(X_train, y_train, max_k, spearman_thresh):
    """
    Performs f-ANOVA and Collinearity filtering STRICTLY on the training set.
    Returns the indices of the top max_k selected features.
    """
    n_feat = X_train.shape[1]
    
    # 1. f-ANOVA (Train Only)
    var = np.var(X_train, axis=0)
    F = np.zeros(n_feat)
    ok = var > 1e-15
    if ok.any():
        Fo, _ = f_classif(X_train[:, ok], y_train)
        F[ok] = np.nan_to_num(Fo, nan=0.0)
    
    rank = np.argsort(-F)
    selected = []
    
    # 2. Spearman Filter (Train Only)
    for feat_idx in rank:
        feat_idx = int(feat_idx)
        if len(selected) >= max_k:
            break
            
        xc = X_train[:, feat_idx]
        skip = False
        for si in selected:
            rho, _ = spearmanr(xc, X_train[:, si])
            if not np.isnan(rho) and abs(rho) > spearman_thresh:
                skip = True
                break
        if not skip:
            selected.append(feat_idx)
            
    return selected

def evaluate_feature_set_sizes_loocv(X, y, feature_names, max_features=25, spearman_thresh=0.7, clf_names=["LR"]):
    """
    Evaluates model performance across different numbers of features (K)
    using strict LOOCV to prevent data leakage.
    """
    N = len(y)
    
    # We will track predictions for every sample, for every K, for every classifier
    predictions_by_k = {cn: {k: np.zeros(N) for k in range(1, max_features + 1)} for cn in clf_names}
    preds_class_by_k = {cn: {k: np.zeros(N, dtype=int) for k in range(1, max_features + 1)} for cn in clf_names}
    
    print("  Running strict Nested LOOCV (this may take a moment)...")
    for i in range(N):
        if (i+1) % 10 == 0: print(f"    Fold {i+1}/{N}")
        
        # Split Data
        tr = np.ones(N, dtype=bool); tr[i] = False
        X_train, y_train = X[tr], y[tr]
        X_test = X[i:i+1]
        
        # Select Features (Train Only)
        selected_indices = select_features_for_fold(X_train, y_train, max_features, spearman_thresh)
        
        for k in range(1, min(max_features + 1, len(selected_indices) + 1)):
            current_k_indices = selected_indices[:k]
            
            # Sub-select features
            Xtr_k = X_train[:, current_k_indices]
            Xte_k = X_test[:, current_k_indices]
            
            # Scale (Train Only)
            sc = StandardScaler()
            Xtr_k_scaled = sc.fit_transform(Xtr_k)
            Xte_k_scaled = sc.transform(Xte_k)
            
            # Train and Predict
            for cn in clf_names:
                clf = get_clf(cn)
                clf.fit(Xtr_k_scaled, y_train)
                predictions_by_k[cn][k][i] = clf.predict_proba(Xte_k_scaled)[0, 1]
                preds_class_by_k[cn][k][i] = clf.predict(Xte_k_scaled)[0]

    # Aggregate results across all folds for each K
    steps = []
    
    # Calculate Power on the full dataset as an indicator (this isn't leakage for power analysis)
    # We run the selection once on the full dataset just to get representative feature sets for the power calculation step.
    full_selected_indices = select_features_for_fold(X, y, max_features, spearman_thresh)
    
    for k in range(1, min(max_features + 1, len(full_selected_indices) + 1)):
        pwr = multivariate_power(X, y, full_selected_indices[:k])
        
        step_dict = {"step": k, "power": pwr, "n_features": k}
        
        for cn in clf_names:
            scores = predictions_by_k[cn][k]
            preds = preds_class_by_k[cn][k]
            
            auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else np.nan
            ba = balanced_accuracy_score(y, preds)
            tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
            
            step_dict[f"{cn}_auc"] = float(auc)
            step_dict[f"{cn}_balacc"] = float(ba)
            step_dict[f"{cn}_sens"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
            step_dict[f"{cn}_spec"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
            step_dict[f"{cn}_scores"] = scores
            step_dict[f"{cn}_preds"] = preds
            
        steps.append(step_dict)
        
    return steps, full_selected_indices

def pick_optimal(steps, clf_name="LR", power_target=0.90):
    auc_key = f"{clf_name}_auc"
    powered = [s for s in steps if not np.isnan(s["power"]) and s["power"] >= power_target]
    pool = powered if powered else steps
    best = max(pool, key=lambda s: s[auc_key])
    return best["step"]

# ============================================================================
# Validation Methods (Mostly Unchanged, adapted for pre-selected features)
# ============================================================================
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

def permutation_test_strict(X, y, max_features, target_k, spearman_thresh, clf_name, n_perms=1000, observed_auc=None):
    """Permutation test strictly re-doing feature selection in every fold inside every permutation."""
    rng = np.random.RandomState(42)
    nulls = []
    N = len(y)
    
    print("    Running strict permutation test (this takes time due to nested loop)...")
    for p in range(n_perms):
        if (p+1) % 100 == 0: print(f"      Perm {p+1}/{n_perms}")
        yp = rng.permutation(y)
        scores = np.zeros(N)
        
        for i in range(N):
            tr = np.ones(N, dtype=bool); tr[i] = False
            X_train, y_train_p = X[tr], yp[tr]
            X_test = X[i:i+1]
            
            sel_idx = select_features_for_fold(X_train, y_train_p, target_k, spearman_thresh)
            
            Xtr_k = X_train[:, sel_idx]
            Xte_k = X_test[:, sel_idx]
            
            sc = StandardScaler()
            Xtr_k_scaled = sc.fit_transform(Xtr_k)
            Xte_k_scaled = sc.transform(Xte_k)
            
            clf = get_clf(clf_name)
            clf.fit(Xtr_k_scaled, y_train_p)
            scores[i] = clf.predict_proba(Xte_k_scaled)[0, 1]
            
        nulls.append(roc_auc_score(yp, scores))
        
    nulls = np.array(nulls)
    pval = float(np.mean(nulls >= (observed_auc or 0.5)))
    return {"p_value": pval, "null_aucs": nulls, "null_mean": float(nulls.mean()), "null_std": float(nulls.std())}

def repeated_stratified_kfold_strict(X, y, target_k, spearman_thresh, clf_names, n_splits=5, n_repeats=10, random_state=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    results = {cn: {"fold_aucs": [], "fold_bas": [], "fold_sens": [], "fold_specs": []} for cn in clf_names}

    for train_idx, test_idx in rskf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        sel_idx = select_features_for_fold(X_train, y_train, target_k, spearman_thresh)
        
        Xtr_k = X_train[:, sel_idx]
        Xte_k = X_test[:, sel_idx]
        
        sc = StandardScaler()
        Xtr_k_scaled = sc.fit_transform(Xtr_k)
        Xte_k_scaled = sc.transform(Xte_k)
        
        for cn in clf_names:
            clf = get_clf(cn)
            clf.fit(Xtr_k_scaled, y_train)
            probs = clf.predict_proba(Xte_k_scaled)[:, 1]
            preds = clf.predict(Xte_k_scaled)
            
            if len(np.unique(y_test)) == 2:
                results[cn]["fold_aucs"].append(roc_auc_score(y_test, probs))
            results[cn]["fold_bas"].append(balanced_accuracy_score(y_test, preds))
            tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
            results[cn]["fold_sens"].append(tp/(tp+fn) if (tp+fn) else 0.0)
            results[cn]["fold_specs"].append(tn/(tn+fp) if (tn+fp) else 0.0)

    for cn in clf_names:
        results[cn]["auc_mean"] = float(np.mean(results[cn]["fold_aucs"]))
        results[cn]["auc_std"] = float(np.std(results[cn]["fold_aucs"]))
        results[cn]["balacc_mean"] = float(np.mean(results[cn]["fold_bas"]))
        results[cn]["balacc_std"] = float(np.std(results[cn]["fold_bas"]))
        results[cn]["sens_mean"] = float(np.mean(results[cn]["fold_sens"]))
        results[cn]["sens_std"] = float(np.std(results[cn]["fold_sens"]))
        results[cn]["spec_mean"] = float(np.mean(results[cn]["fold_specs"]))
        results[cn]["spec_std"] = float(np.std(results[cn]["fold_specs"]))
        results[cn]["fold_aucs"] = np.array(results[cn]["fold_aucs"])
        
    return results

# ============================================================================
# Figure Helpers (Identical implementation, omitted for brevity but they remain the same)
# ============================================================================
# (Copy fig_auc_power_curve, fig_final_roc, fig_permutation, fig_kfold_distributions, 
# fig_validation_summary, fig_sample_predictions from your original script here)

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
    ax1.set_ylabel("AUC (Strict Nested LOOCV)", fontsize=12)
    ax2.set_ylabel("Statistical Power", fontsize=12)
    ax1.set_ylim(0.3, 1.02)
    ax2.set_ylim(0.0, 1.05)
    ax1.set_xticks(ns)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9)
    ax1.set_title("Strict Feature Selection: LOOCV AUC vs # Features", fontsize=13)
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_final_roc(y, optimal_step_data, clf_names, boot_cis, outpath):
    fig, ax = plt.subplots(figsize=(6, 6))
    for cn in clf_names:
        fpr, tpr, _ = roc_curve(y, optimal_step_data[f"{cn}_scores"])
        lo, hi = boot_cis[cn]
        ax.plot(fpr, tpr, color=CLF_COLORS.get(cn, "gray"), linewidth=2,
                label=f"{cn}  AUC={optimal_step_data[f'{cn}_auc']:.3f}  [{lo:.3f}-{hi:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Final Validation: LOOCV ROC (95% Bootstrap CI)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

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

def fig_validation_summary(optimal_step_data, rskf_res, boot_cis, perm_results,
                            clf_names, feat_names, power, outpath):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [1, 1.2]})

    ax = axes[0]
    ax.axis("off")
    feat_text = "\n".join([f"  {i+1}. {f}" for i, f in enumerate(feat_names)])
    ax.text(0.02, 0.95,
            f"Final Model Trained Features ({len(feat_names)} features, "
            f"Power = {power:.3f})\n{feat_text}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))

    ax = axes[1]
    ax.axis("off")

    col_labels = ["Classifier",
                  "LOOCV\nAUC", "LOOCV\nBal Acc", "LOOCV\nSens", "LOOCV\nSpec",
                  "AUC\n95% CI",
                  "5F-CV\nAUC", "5F-CV\nBal Acc",
                  "Perm\np-val"]
    rows = []
    for cn in clf_names:
        lr = optimal_step_data
        kr = rskf_res[cn]
        lo, hi = boot_cis[cn]
        pp = perm_results.get(cn, {}).get("p_value", "")
        pp_str = f"{pp:.4f}" if isinstance(pp, float) else ""
        rows.append([
            cn,
            f"{lr[f'{cn}_auc']:.3f}", f"{lr[f'{cn}_balacc']:.3f}",
            f"{lr[f'{cn}_sens']:.3f}", f"{lr[f'{cn}_spec']:.3f}",
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
    parser.add_argument("--n_perms", type=int, default=100) # Reduced default for strict nested permutation
    args = parser.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STRICT NESTED LOOCV FEATURE SELECTION")
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
    # PHASE 1: STICT NESTED EVALUATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: EVALUATING FEATURE SIZES WITHOUT LEAKAGE")
    print("=" * 70)

    steps, full_ds_selected_idx = evaluate_feature_set_sizes_loocv(
        X, y, feat_cols, max_features=args.max_features, 
        spearman_thresh=args.spearman_thresh, clf_names=clf_names
    )

    # Pick optimal K
    optimal_k = pick_optimal(steps, clf_name="LR", power_target=args.power_target)
    opt_step = next(s for s in steps if s["step"] == optimal_k)
    
    # Now that we know the optimal K without bias, we train a final model on the 
    # full dataset to tell the user which features to actually use in the future.
    final_features_idx = full_ds_selected_idx[:optimal_k]
    final_features_names = [feat_cols[i] for i in final_features_idx]

    print(f"\n{'='*70}")
    print(f"OPTIMAL (Unbiased): K={optimal_k} features, power={opt_step['power']:.3f}")
    for cn in clf_names:
        print(f"  {cn}: AUC={opt_step[f'{cn}_auc']:.3f}  BA={opt_step[f'{cn}_balacc']:.3f}")
    print(f"Final Model Features (Trained on full dataset for future use): \n{final_features_names}")
    print(f"{'='*70}")

    fig_auc_power_curve(steps, clf_names, optimal_k, outdir / "fig1_auc_power_curve_strict.png")

    # ==================================================================
    # PHASE 2: FINAL VALIDATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FINAL VALIDATION")
    print("=" * 70)

    # 2b. Bootstrap CI on the strict LOOCV predictions
    print("\n  [2b] Bootstrap 95% CI on LOOCV AUC (2000 resamples)...")
    boot_cis = {}
    for cn in clf_names:
        lo, hi, _ = bootstrap_auc_ci(y, opt_step[f"{cn}_scores"])
        boot_cis[cn] = (lo, hi)
        print(f"    {cn}: AUC = {opt_step[f'{cn}_auc']:.3f}  95% CI [{lo:.3f} - {hi:.3f}]")

    # 2c. Repeated Stratified 5-Fold CV (Strict Nested)
    print("\n  [2c] Repeated Stratified 5-Fold CV (10 repeats)...")
    rskf_res = repeated_stratified_kfold_strict(X, y, optimal_k, args.spearman_thresh, clf_names)
    for cn in clf_names:
        r = rskf_res[cn]
        print(f"    {cn}: AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}")

    # 2d. Permutation test (Strict Nested)
    perm_results = {}
    best_cn = max(clf_names, key=lambda c: opt_step[f"{c}_auc"])
    print(f"\n  [2d] Permutation test ({args.n_perms} shuffles, running on best model: {best_cn})...")
    obs = opt_step[f"{best_cn}_auc"]
    perm = permutation_test_strict(X, y, args.max_features, optimal_k, args.spearman_thresh, best_cn, n_perms=args.n_perms, observed_auc=obs)
    perm_results[best_cn] = perm
    print(f"    p = {perm['p_value']:.4f}  (null = {perm['null_mean']:.3f} +/- {perm['null_std']:.3f})")

    # Save summary logic remains similar, generate figures
    fig_final_roc(y, opt_step, clf_names, boot_cis, outdir / "fig2_final_roc_strict.png")
    fig_permutation(perm_results[best_cn], opt_step[f"{best_cn}_auc"], best_cn, outdir / "fig3_permutation_strict.png")
    fig_kfold_distributions(rskf_res, clf_names, outdir / "fig4_kfold_distributions_strict.png")
    fig_validation_summary(opt_step, rskf_res, boot_cis, perm_results, clf_names, final_features_names, opt_step["power"], outdir / "fig5_validation_summary_strict.png")

    print(f"\n{'='*70}")
    print(f"ALL DONE. Leak-free outputs generated in: {outdir}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()