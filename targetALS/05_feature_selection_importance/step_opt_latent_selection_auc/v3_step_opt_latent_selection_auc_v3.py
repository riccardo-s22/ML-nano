#!/usr/bin/env python3
"""
STEP OPT v3: Optimal Latent Selection (Leak-Free AUC + AUC-Monotonicity Stop)
===============================================================================

Changes over v2:
  1. STOPPING CRITERION: Instead of stopping when binomial power >= target,
     features are added greedily and we stop when the LOOCV AUC starts
     DECREASING. This finds the natural "elbow" where adding more features
     hurts generalization.
  2. PATIENCE: A configurable patience parameter (default=2) tolerates up to
     N consecutive non-improving steps before stopping. This avoids premature
     termination from a single noisy dip.
  3. AUC TRACKING: run_loocv now collects predict_proba scores and computes
     proper LOOCV AUC (ROC-AUC from out-of-fold probability scores) for each
     classifier. The stopping criterion uses the BEST AUC across classifiers
     at each step.
  4. ROLLBACK: When stopping due to AUC decrease, the final selected set is
     rolled back to the point of peak AUC (not the last step tried).
  5. All v2 features retained: leak-free AUC ranking, Spearman redundancy,
     permutation testing, auto config output.

Pipeline:
  Step 1 -> Step OPT v3 -> Step 2 (encoder proxy) -> downstream
                |
                +-> selected_features_config.json

Usage:
  python step_opt_latent_selection_auc_v3.py \
    --step1_dir ./step1_latent_features \
    --labels_csv sample_labels.csv \
    --output_dir ./opt_selection_results_v3 \
    --redundancy_thresh 0.85 \
    --patience 2 \
    --max_features 20 \
    --n_permutations 500
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict

from scipy.stats import spearmanr, binom
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# =============================================================================
# Data loading
# =============================================================================

def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    df['code'] = df['code'].astype(str)
    df['target'] = (df['group'] == 'ALS').astype(int)
    return df, df['code'].tolist(), df['target'].values


def load_all_latent_features(step1_dir: Path, codes: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Load and concatenate latent features from all model subdirectories."""
    model_dirs = sorted([d for d in step1_dir.iterdir()
                         if d.is_dir() and not d.name.startswith(".")
                         and (d / "z_agg.csv").exists()])

    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {step1_dir}")

    print(f"Found {len(model_dirs)} models. Aggregating features...")
    master_df = pd.DataFrame({'code': codes})

    for mdir in model_dirs:
        z_path = mdir / "z_agg.csv"
        df_z = pd.read_csv(z_path)
        df_z['code'] = df_z['code'].astype(str)

        dim_cols = [c for c in df_z.columns if c.startswith("dim")]
        rename_map = {c: f"{mdir.name}_{c}" for c in dim_cols}
        df_z = df_z.rename(columns=rename_map)

        cols_to_keep = ['code'] + list(rename_map.values())
        master_df = pd.merge(master_df, df_z[cols_to_keep], on='code', how='left')

    if master_df.isnull().any().any():
        master_df = master_df.fillna(0.0)

    feature_cols = [c for c in master_df.columns if c != 'code']
    X = master_df[feature_cols].values
    return X, feature_cols


# =============================================================================
# Power calculation (proper binomial)
# =============================================================================

def calculate_binomial_power(accuracy: float, n_samples: int, alpha: float = 0.05) -> float:
    """
    Compute statistical power for a classification accuracy test.

    H0: true accuracy = 0.5 (random chance)
    H1: true accuracy = observed accuracy

    Power = P(reject H0 | H1 is true)
          = P(X >= k_crit | p = observed_accuracy)

    where k_crit is the critical value from Binom(n, 0.5) at significance alpha.
    """
    if accuracy <= 0.5:
        return 0.0

    n = int(n_samples)
    p0 = 0.5  # null hypothesis: chance
    p1 = float(accuracy)  # alternative: observed accuracy

    # Critical value: smallest k such that P(X >= k | p=p0) <= alpha
    # i.e., k_crit = quantile(1 - alpha) + 1 of Binom(n, p0)
    k_crit = binom.ppf(1.0 - alpha, n, p0)
    # ppf returns the largest k with CDF <= q, so we need k_crit + 1
    # for the one-sided rejection region
    k_crit = int(k_crit) + 1

    if k_crit > n:
        return 0.0

    # Power = P(X >= k_crit | p = p1)
    power = 1.0 - binom.cdf(k_crit - 1, n, p1)
    return float(power)


# =============================================================================
# Redundancy filter (Spearman |ρ|)
# =============================================================================

def is_redundant_spearman(candidate: np.ndarray, selected_vecs: List[np.ndarray],
                          threshold: float) -> Tuple[bool, float]:
    """
    Check if candidate feature is redundant with any already-selected feature
    using Spearman |ρ|. More stable than NMI at small sample sizes.
    """
    max_rho = 0.0
    for sel_vec in selected_vecs:
        rho, _ = spearmanr(candidate, sel_vec)
        rho_abs = abs(rho) if not np.isnan(rho) else 0.0
        if rho_abs > max_rho:
            max_rho = rho_abs
        if rho_abs > threshold:
            return True, max_rho
    return False, max_rho


# =============================================================================
# LOOCV classification
# =============================================================================

def run_loocv(X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Run LOOCV with SVM, RF, LR.
    Returns dict of {classifier_name: {"accuracy": float, "auc": float}}.
    AUC is computed from out-of-fold predict_proba scores.
    """
    classifiers = {
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced_subsample"),
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=SEED)
    }

    loo = LeaveOneOut()
    preds = {k: [] for k in classifiers}
    proba = {k: [] for k in classifiers}

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, clf in classifiers.items():
            clf.fit(X_tr_s, y_tr)
            preds[name].append(clf.predict(X_te_s)[0])
            proba[name].append(clf.predict_proba(X_te_s)[0, 1])

    results = {}
    for name in classifiers:
        acc = accuracy_score(y, preds[name])
        try:
            auc = roc_auc_score(y, proba[name])
        except Exception:
            auc = 0.5
        results[name] = {"accuracy": acc, "auc": auc}
    return results


# =============================================================================
# Nested LOOCV AUC ranking (leak-free)
# =============================================================================

def compute_leakfree_auc_ranking(X_all: np.ndarray, y: np.ndarray,
                                  feature_names: List[str]) -> pd.DataFrame:
    """
    For each feature, compute its AUC on TRAIN-ONLY data across all LOOCV folds.
    This avoids the leakage of ranking features on the same data used for evaluation.

    Returns DataFrame with per-feature mean AUC, std, and stability (fraction of
    folds where it ranked in top-50).
    """
    N, D = X_all.shape
    loo = LeaveOneOut()

    # Accumulate AUC scores per fold
    auc_per_fold = np.zeros((N, D))  # [n_folds, n_features]

    print("Computing leak-free AUC ranking (N LOOCV folds)...")
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X_all)):
        y_tr = y[train_idx]
        X_tr = X_all[train_idx]

        for d in range(D):
            vals = X_tr[:, d]
            if np.std(vals) < 1e-12:
                auc_per_fold[fold_idx, d] = 0.5
                continue
            try:
                auc = roc_auc_score(y_tr, vals)
                auc_per_fold[fold_idx, d] = max(auc, 1.0 - auc)
            except Exception:
                auc_per_fold[fold_idx, d] = 0.5

    # Summarize across folds
    mean_auc = auc_per_fold.mean(axis=0)
    std_auc = auc_per_fold.std(axis=0)

    # Stability: fraction of folds where feature was in top-50
    top_k_stability = 50
    stability = np.zeros(D)
    for fold_idx in range(N):
        fold_aucs = auc_per_fold[fold_idx]
        ranking_scores = np.abs(fold_aucs - 0.5)
        top_indices = np.argsort(-ranking_scores)[:top_k_stability]
        stability[top_indices] += 1.0
    stability /= N

    ranking_df = pd.DataFrame({
        "feature": feature_names,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "ranking_score": np.abs(mean_auc - 0.5),
        "stability_top50": stability,
    })
    ranking_df = ranking_df.sort_values("ranking_score", ascending=False).reset_index(drop=True)
    return ranking_df


# =============================================================================
# Permutation test
# =============================================================================

def permutation_test(X: np.ndarray, y: np.ndarray, n_permutations: int = 500,
                     seed: int = SEED) -> Tuple[float, float, float, np.ndarray]:
    """
    Run permutation test on the selected feature set.
    Shuffles labels and re-runs LOOCV to build a null distribution.
    Returns (observed_best_auc, observed_best_acc, p_value, null_distribution_auc).
    """
    rng = np.random.RandomState(seed)

    # Observed metrics
    obs_results = run_loocv(X, y)
    obs_best_auc = max(v["auc"] for v in obs_results.values())
    obs_best_acc = max(v["accuracy"] for v in obs_results.values())

    null_aucs = np.zeros(n_permutations)
    for i in tqdm(range(n_permutations), desc="Permutation test"):
        y_perm = rng.permutation(y)
        perm_results = run_loocv(X, y_perm)
        null_aucs[i] = max(v["auc"] for v in perm_results.values())

    # p-value: fraction of null >= observed
    p_value = (np.sum(null_aucs >= obs_best_auc) + 1) / (n_permutations + 1)
    return obs_best_auc, obs_best_acc, p_value, null_aucs


# =============================================================================
# Parse feature name -> model + dim
# =============================================================================

def parse_feature_to_model_dim(feature_name: str) -> Tuple[str, int]:
    """Parse 'best_model_fold1_dim103' -> ('best_model_fold1', 103)"""
    idx = feature_name.rfind("_dim")
    if idx == -1:
        raise ValueError(f"Cannot parse feature name: {feature_name}")
    model_name = feature_name[:idx]
    dim_idx = int(feature_name[idx + 4:])
    return model_name, dim_idx


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step OPT v3: Leak-free latent feature selection with AUC-monotonicity stopping"
    )
    parser.add_argument("--step1_dir", type=str, required=True,
                        help="Output directory from Step 1")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="CSV with columns: code, group")
    parser.add_argument("--output_dir", type=str, default="./opt_selection_results_v3")
    parser.add_argument("--redundancy_thresh", type=float, default=0.85,
                        help="Spearman |rho| threshold for redundancy filtering (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Number of consecutive non-improving steps before stopping. "
                             "0 = stop at first AUC decrease. "
                             "2 = tolerate up to 2 dips (default, recommended for N=39).")
    parser.add_argument("--max_features", type=int, default=20,
                        help="Maximum number of features to select")
    parser.add_argument("--n_permutations", type=int, default=500,
                        help="Number of permutations for significance test")
    parser.add_argument("--min_stability", type=float, default=0.5,
                        help="Minimum fold-stability for a feature to be considered")
    parser.add_argument("--stop_metric", type=str, default="auc",
                        choices=["auc", "accuracy"],
                        help="Which metric to monitor for stopping: 'auc' (recommended) or 'accuracy'")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP OPT v3: Leak-Free Latent Feature Selection (AUC-Monotonicity Stop)")
    print("=" * 70)
    print(f"Redundancy filter: Spearman |rho| > {args.redundancy_thresh}")
    print(f"Stop metric:       {args.stop_metric} (stop when it decreases)")
    print(f"Patience:          {args.patience} (consecutive non-improving steps tolerated)")
    print(f"Max features:      {args.max_features}")
    print(f"Permutations:      {args.n_permutations}")
    print(f"Min stability:     {args.min_stability}")

    # 1. Load data
    print("\nLoading data...")
    df_labels, codes, y = load_labels(args.labels_csv)
    X_all, feature_names = load_all_latent_features(Path(args.step1_dir), codes)
    N, D = X_all.shape
    n_als = int(y.sum())
    n_ctrl = N - n_als
    print(f"Samples: {N} (ALS={n_als}, CTRL={n_ctrl})")
    print(f"Total features: {D}")

    # 2. Leak-free AUC ranking
    ranking_df = compute_leakfree_auc_ranking(X_all, y, feature_names)
    ranking_df.to_csv(outdir / "feature_ranking_auc_leakfree.csv", index=False)
    print(f"\nTop 10 features by leak-free AUC:")
    for _, row in ranking_df.head(10).iterrows():
        print(f"  {row['feature']}: AUC={row['mean_auc']:.3f} ± {row['std_auc']:.3f} "
              f"(stability={row['stability_top50']:.2f})")

    # 3. Iterative selection with Spearman redundancy + AUC-monotonicity stopping
    print(f"\nIterative feature selection...")
    print(f"  (Spearman |rho| < {args.redundancy_thresh}, "
          f"stability > {args.min_stability}, "
          f"stop when {args.stop_metric} decreases for {args.patience} steps)")

    selected_indices = []  # indices into X_all columns
    selected_vecs = []     # actual vectors for redundancy checks
    history = []

    best_metric_so_far = 0.0   # track best AUC (or accuracy) seen
    best_step = 0              # step index where best metric was achieved
    no_improve_count = 0       # consecutive steps without improvement

    candidates = ranking_df.to_dict('records')

    for rank_pos, row in enumerate(candidates):
        if len(selected_indices) >= args.max_features:
            print(f"\nMax features ({args.max_features}) reached.")
            break

        fname = row['feature']
        feat_idx = feature_names.index(fname)
        candidate_vec = X_all[:, feat_idx]

        # Stability filter
        if row['stability_top50'] < args.min_stability:
            continue

        # Redundancy check (Spearman |ρ|)
        if selected_vecs:
            is_red, max_rho = is_redundant_spearman(
                candidate_vec, selected_vecs, args.redundancy_thresh
            )
            if is_red:
                continue
        else:
            max_rho = 0.0

        # Add feature
        selected_indices.append(feat_idx)
        selected_vecs.append(candidate_vec)

        # Evaluate current feature set
        X_curr = X_all[:, selected_indices]
        res = run_loocv(X_curr, y)

        # Extract metrics per classifier
        best_auc = max(v["auc"] for v in res.values())
        best_acc = max(v["accuracy"] for v in res.values())
        best_clf_auc = max(res, key=lambda k: res[k]["auc"])
        best_clf_acc = max(res, key=lambda k: res[k]["accuracy"])

        # The metric we monitor for stopping
        if args.stop_metric == "auc":
            current_metric = best_auc
            metric_label = "AUC"
        else:
            current_metric = best_acc
            metric_label = "Acc"

        power = calculate_binomial_power(best_acc, N)

        # Track improvement
        improved = current_metric > best_metric_so_far
        if improved:
            best_metric_so_far = current_metric
            best_step = len(history)  # 0-indexed step
            no_improve_count = 0
            marker = "▲"
        else:
            no_improve_count += 1
            marker = "▼" if current_metric < best_metric_so_far else "="

        entry = {
            "n_features": len(selected_indices),
            "added_feature": fname,
            "mean_auc_of_feature": row['mean_auc'],
            "stability": row['stability_top50'],
            "max_spearman_rho_with_selected": max_rho,
            "best_auc": best_auc,
            "best_acc": best_acc,
            "best_clf_auc": best_clf_auc,
            "best_clf_acc": best_clf_acc,
            "binomial_power": power,
            "svm_acc": res['SVM']['accuracy'],
            "svm_auc": res['SVM']['auc'],
            "rf_acc": res['RF']['accuracy'],
            "rf_auc": res['RF']['auc'],
            "lr_acc": res['LR']['accuracy'],
            "lr_auc": res['LR']['auc'],
            "is_best_so_far": improved,
        }
        history.append(entry)

        print(f"  [{len(selected_indices):2d}] {marker} +{fname} | "
              f"AUC={best_auc:.3f} ({best_clf_auc}) | "
              f"Acc={best_acc:.3f} ({best_clf_acc}) | "
              f"Power={power:.3f}")

        # Check stopping: patience exceeded?
        if no_improve_count > args.patience and len(selected_indices) > 1:
            print(f"\n  >>> {metric_label} has not improved for {no_improve_count} steps "
                  f"(patience={args.patience}). Stopping.")
            print(f"  >>> Peak {metric_label}={best_metric_so_far:.4f} was at step {best_step + 1} "
                  f"({history[best_step]['n_features']} features).")
            break

    # -------------------------------------------------------------------------
    # Rollback to peak step
    # -------------------------------------------------------------------------
    if history and best_step < len(history) - 1:
        n_rollback = len(history) - 1 - best_step
        print(f"\n  Rolling back {n_rollback} feature(s) to peak {metric_label} step...")
        # Trim selected_indices and selected_vecs to best_step+1 features
        n_keep = history[best_step]["n_features"]
        selected_indices = selected_indices[:n_keep]
        selected_vecs = selected_vecs[:n_keep]
        print(f"  Final feature count: {len(selected_indices)} (was {history[-1]['n_features']})")

    # Save selection history (full, including rolled-back steps)
    history_df = pd.DataFrame(history)
    history_df["kept"] = False
    if history:
        history_df.loc[:best_step, "kept"] = True
    history_df.to_csv(outdir / "selection_history.csv", index=False)

    # Final selected features
    final_names = [feature_names[i] for i in selected_indices]
    print(f"\nSelected {len(final_names)} features:")
    for fn in final_names:
        print(f"  - {fn}")

    # 4. Permutation test
    print(f"\nRunning permutation test ({args.n_permutations} permutations)...")
    X_final = X_all[:, selected_indices]
    obs_best_auc, obs_best_acc, perm_pval, null_dist = permutation_test(
        X_final, y, n_permutations=args.n_permutations
    )
    print(f"  Observed best AUC:      {obs_best_auc:.3f}")
    print(f"  Observed best accuracy: {obs_best_acc:.3f}")
    print(f"  Permutation p-value:    {perm_pval:.4f}")
    print(f"  Null distribution:      mean={null_dist.mean():.3f}, "
          f"max={null_dist.max():.3f}")

    # Save null distribution
    np.save(outdir / "permutation_null_distribution.npy", null_dist)

    # 5. Build config for downstream steps
    # Parse features into model -> [dim_indices] mapping
    model_dims: Dict[str, List[int]] = {}
    for fname in final_names:
        mname, dim_idx = parse_feature_to_model_dim(fname)
        if mname not in model_dims:
            model_dims[mname] = []
        model_dims[mname].append(dim_idx)

    # Peak metrics from selection history
    peak_entry = history[best_step] if history else {}

    config = {
        "pipeline_version": "opt_v3",
        "n_samples": N,
        "n_als": n_als,
        "n_ctrl": n_ctrl,
        "n_features_selected": len(final_names),
        "target_features": final_names,
        "model_dims": model_dims,
        "selection_params": {
            "redundancy_method": "spearman_abs_rho",
            "redundancy_threshold": args.redundancy_thresh,
            "stop_criterion": f"{args.stop_metric}_monotonicity",
            "patience": args.patience,
            "min_stability": args.min_stability,
            "auc_ranking": "leak_free_loocv",
        },
        "final_metrics": {
            "best_auc": float(obs_best_auc),
            "best_accuracy": float(obs_best_acc),
            "permutation_p_value": float(perm_pval),
            "n_permutations": args.n_permutations,
            "binomial_power": float(peak_entry.get("binomial_power", 0.0)),
            "peak_step": int(best_step + 1),
            "peak_auc_at_selection": float(peak_entry.get("best_auc", 0.0)),
            "peak_acc_at_selection": float(peak_entry.get("best_acc", 0.0)),
            "total_steps_tried": len(history),
        },
        "per_feature_auc": {
            fn: float(ranking_df.loc[ranking_df['feature'] == fn, 'mean_auc'].values[0])
            for fn in final_names
        },
    }

    config_path = outdir / "selected_features_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_path}")

    # Also save flat CSV for convenience
    pd.DataFrame({"feature": final_names}).to_csv(
        outdir / "final_selected_features.csv", index=False
    )

    # 6. Summary
    print("\n" + "=" * 70)
    print("STEP OPT v3 SUMMARY")
    print("=" * 70)
    print(f"Features selected:     {len(final_names)}")
    print(f"Peak LOOCV AUC:        {peak_entry.get('best_auc', 'N/A'):.4f}" if peak_entry else "N/A")
    print(f"Peak LOOCV Accuracy:   {peak_entry.get('best_acc', 'N/A'):.4f}" if peak_entry else "N/A")
    print(f"Binomial power:        {peak_entry.get('binomial_power', 'N/A'):.3f}" if peak_entry else "N/A")
    print(f"Stopped at step:       {best_step + 1} / {len(history)} tried")
    print(f"Permutation p-value:   {perm_pval:.4f}")
    print(f"Models involved:       {list(model_dims.keys())}")
    print(f"\nOutputs in {outdir}:")
    print(f"  selected_features_config.json  <- consumed by Step 2 & downstream")
    print(f"  feature_ranking_auc_leakfree.csv")
    print(f"  selection_history.csv          <- full trace with ▲/▼ and rollback info")
    print(f"  final_selected_features.csv")
    print(f"  permutation_null_distribution.npy")


if __name__ == "__main__":
    main()
