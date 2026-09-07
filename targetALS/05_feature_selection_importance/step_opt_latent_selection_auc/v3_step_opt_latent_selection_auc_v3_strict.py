#!/usr/bin/env python3
"""
STEP OPT v3 (CORRECTED): Optimal Latent Selection (Strict Nested Leak-Free AUC)
===============================================================================

Corrections applied:
  1. STRICT NESTED CV: The univariate AUC ranking, Spearman redundancy filter, 
     and the early-stopping (patience) mechanism are now wrapped inside an 
     inner cross-validation loop. They are calculated ONLY on the N-1 training 
     samples of each fold.
  2. HONEST OUTER METRICS: The final reported AUC and Accuracy are generated 
     by the Outer LOOCV loop, which tests the model on the strictly held-out sample.
  3. STRICT PERMUTATION TEST: The permutation test now shuffles labels and 
     re-runs the ENTIRE nested selection pipeline to properly account for 
     multiple-testing bias.
"""

import os, sys, warnings, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from scipy.stats import spearmanr, binom
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")
SEED = 42

# ============================================================================
# Original Custom Utility Functions
# ============================================================================

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

def parse_feature_to_model_dim(feature_name: str) -> Tuple[str, int]:
    """Parse 'best_model_fold1_dim103' -> ('best_model_fold1', 103)"""
    idx = feature_name.rfind("_dim")
    if idx == -1:
        raise ValueError(f"Cannot parse feature name: {feature_name}")
    model_name = feature_name[:idx]
    dim_idx = int(feature_name[idx + 4:])
    return model_name, dim_idx

# ============================================================================
# Power and Classifiers
# ============================================================================

def calculate_binomial_power(accuracy: float, n_samples: int, alpha: float = 0.05) -> float:
    if accuracy <= 0.5: return 0.0
    n = int(n_samples)
    p0 = 0.5
    p1 = float(accuracy)
    k_crit = int(binom.ppf(1.0 - alpha, n, p0)) + 1
    if k_crit > n: return 0.0
    power = 1.0 - binom.cdf(k_crit - 1, n, p1)
    return float(power)

def get_classifiers():
    return {
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced_subsample"),
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=SEED)
    }

# ============================================================================
# Core Inner Logic (Runs ONLY on Training Data)
# ============================================================================

def run_inner_loocv(X_train: np.ndarray, y_train: np.ndarray):
    """Evaluates a feature subset purely within the training data to guide stopping."""
    classifiers = get_classifiers()
    loo = LeaveOneOut()
    preds = {k: [] for k in classifiers}
    proba = {k: [] for k in classifiers}

    for tr_idx, te_idx in loo.split(X_train):
        X_tr, X_te = X_train[tr_idx], X_train[te_idx]
        y_tr = y_train[tr_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, clf in classifiers.items():
            clf.fit(X_tr_s, y_tr)
            preds[name].append(clf.predict(X_te_s)[0])
            proba[name].append(clf.predict_proba(X_te_s)[0, 1])

    results = {}
    for name in classifiers:
        acc = accuracy_score(y_train, preds[name])
        try: auc = roc_auc_score(y_train, proba[name])
        except: auc = 0.5
        results[name] = {"accuracy": acc, "auc": auc}
    return results

def select_features_inner(X_train, y_train, feature_names, max_features, redundancy_thresh, patience, stop_metric):
    """
    The core selection engine. 
    It ranks, filters, and applies patience strictly on the provided X_train slice.
    """
    # 1. Univariate AUC Ranking (Train only)
    aucs = []
    for d in range(X_train.shape[1]):
        v = X_train[:, d]
        if np.std(v) < 1e-12: aucs.append(0.5)
        else:
            try: aucs.append(max(roc_auc_score(y_train, v), 1.0 - roc_auc_score(y_train, v)))
            except: aucs.append(0.5)
            
    rank = np.argsort(-np.array(aucs))
    
    selected_idx = []
    best_metric = 0.0
    best_step = 0
    no_improve_count = 0
    history = []
    
    for feat_idx in rank:
        if len(selected_idx) >= max_features: break
        
        # 2. Redundancy Filter (Train only)
        if selected_idx:
            cand_vec = X_train[:, feat_idx]
            skip = False
            for s_idx in selected_idx:
                rho, _ = spearmanr(cand_vec, X_train[:, s_idx])
                if abs(rho) > redundancy_thresh:
                    skip = True
                    break
            if skip: continue
            
        selected_idx.append(feat_idx)
        
        # 3. Evaluate new set (Train only)
        inner_res = run_inner_loocv(X_train[:, selected_idx], y_train)
        current_metric = max([v[stop_metric] for v in inner_res.values()])
        
        step_data = {
            "n_features": len(selected_idx),
            "added_feature": feature_names[feat_idx] if feature_names else str(feat_idx),
            "best_metric": current_metric,
            "inner_results": inner_res
        }
        history.append(step_data)
        
        # 4. Patience Stopping
        if current_metric > best_metric:
            best_metric = current_metric
            best_step = len(selected_idx)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > patience and len(selected_idx) > 1:
            break
            
    # Rollback to optimal step
    return selected_idx[:best_step], history

# ============================================================================
# Strict Outer Evaluation & Permutations
# ============================================================================

def evaluate_pipeline_strictly(X_all, y, feature_names, args):
    """
    Outer LOOCV loop. Hides 1 sample, forces the pipeline to select features 
    from scratch on N-1, and tests on the hidden sample.
    """
    N = len(y)
    preds = {cn: np.zeros(N) for cn in ["SVM", "RF", "LR"]}
    proba = {cn: np.zeros(N) for cn in ["SVM", "RF", "LR"]}
    
    for i in tqdm(range(N), desc="Strict Outer LOOCV"):
        tr = np.ones(N, dtype=bool); tr[i] = False
        X_tr, y_tr = X_all[tr], y[tr]
        X_te = X_all[i:i+1]
        
        # Select optimal features WITHOUT the test sample
        opt_idx, _ = select_features_inner(
            X_tr, y_tr, None, args.max_features, 
            args.redundancy_thresh, args.patience, args.stop_metric
        )
        
        if len(opt_idx) == 0:
            # Fallback if patience kills selection on fold 1
            opt_idx = [0]
            
        # Scale and Train using the selected indices
        X_tr_s = X_tr[:, opt_idx]
        X_te_s = X_te[:, opt_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_s)
        X_te_s = sc.transform(X_te_s)
        
        classifiers = get_classifiers()
        for cn, clf in classifiers.items():
            clf.fit(X_tr_s, y_tr)
            preds[cn][i] = clf.predict(X_te_s)[0]
            proba[cn][i] = clf.predict_proba(X_te_s)[0, 1]
            
    results = {}
    for cn in ["SVM", "RF", "LR"]:
        acc = accuracy_score(y, preds[cn])
        try: auc = roc_auc_score(y, proba[cn])
        except: auc = 0.5
        results[cn] = {"accuracy": acc, "auc": auc}
    return results

def strict_permutation_test(X_all, y, obs_auc, args, n_perms=50):
    """Runs the full nested pipeline on shuffled labels."""
    null_aucs = []
    rng = np.random.RandomState(SEED)
    
    print(f"\nRunning strict permutation test (this will take time. Perms={n_perms})...")
    for p in tqdm(range(n_perms), desc="Permutations"):
        y_perm = rng.permutation(y)
        perm_res = evaluate_pipeline_strictly(X_all, y_perm, None, args)
        null_aucs.append(max(v["auc"] for v in perm_res.values()))
        
    null_aucs = np.array(null_aucs)
    pval = (np.sum(null_aucs >= obs_auc) + 1) / (n_perms + 1)
    return pval, null_aucs

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./opt_selection_results_v3_strict")
    parser.add_argument("--redundancy_thresh", type=float, default=0.85)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--n_permutations", type=int, default=50) # Reduced due to strict nesting cost
    parser.add_argument("--stop_metric", type=str, default="auc", choices=["auc", "accuracy"])
    args = parser.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP OPT v3: STRICT NESTED Leak-Free Latent Feature Selection")
    print("=" * 70)

    # 1. Load data
    df_labels, codes, y = load_labels(args.labels_csv)
    X_all, feature_names = load_all_latent_features(Path(args.step1_dir), codes)
    N, D = X_all.shape
    print(f"Samples: {N} (ALS={int(y.sum())}, CTRL={N - int(y.sum())}) | Features: {D}")

    # 2. Extract Final Features for Future Use (Runs on Full Dataset)
    # We run the inner loop logic on ALL data solely to output the config.json features for downstream apps.
    print("\n[1/3] Extracting final model features (Trained on full dataset)...")
    final_idx, full_history = select_features_inner(
        X_all, y, feature_names, args.max_features, 
        args.redundancy_thresh, args.patience, args.stop_metric
    )
    final_names = [feature_names[i] for i in final_idx]
    
    # Save full dataset history
    # Add proper handling to parse 'inner_results' into columns before saving
    history_rows = []
    for step in full_history:
        row = {"n_features": step["n_features"], "added_feature": step["added_feature"], "best_metric": step["best_metric"]}
        for cn, mets in step["inner_results"].items():
            row[f"{cn}_acc"] = mets["accuracy"]
            row[f"{cn}_auc"] = mets["auc"]
        history_rows.append(row)
    pd.DataFrame(history_rows).to_csv(outdir / "full_dataset_selection_history.csv", index=False)

    # 3. Strict Nested Evaluation (The Honest Performance)
    print("\n[2/3] Evaluating methodology with Strict Nested LOOCV...")
    strict_results = evaluate_pipeline_strictly(X_all, y, feature_names, args)
    
    obs_best_auc = max(v["auc"] for v in strict_results.values())
    obs_best_acc = max(v["accuracy"] for v in strict_results.values())
    best_clf = max(strict_results, key=lambda k: strict_results[k]["auc"])

    # 4. Strict Permutation
    print("\n[3/3] Permutation Testing...")
    perm_pval, null_dist = strict_permutation_test(X_all, y, obs_best_auc, args, n_perms=args.n_permutations)
    np.save(outdir / "permutation_null_distribution.npy", null_dist)

    # 5. Save Config
    model_dims = {}
    for fname in final_names:
        mname, dim_idx = parse_feature_to_model_dim(fname)
        if mname not in model_dims: model_dims[mname] = []
        model_dims[mname].append(dim_idx)

    config = {
        "pipeline_version": "opt_v3_strict",
        "n_samples": N,
        "n_features_selected": len(final_names),
        "target_features": final_names,
        "model_dims": model_dims,
        "final_metrics": {
            "best_strict_auc": float(obs_best_auc),
            "best_strict_accuracy": float(obs_best_acc),
            "best_classifier": best_clf,
            "permutation_p_value": float(perm_pval),
            "binomial_power": float(calculate_binomial_power(obs_best_acc, N))
        }
    }
    with open(outdir / "selected_features_config.json", "w") as f:
        json.dump(config, f, indent=2)
    pd.DataFrame({"feature": final_names}).to_csv(outdir / "final_selected_features.csv", index=False)

    # 6. Summary
    print("\n" + "=" * 70)
    print("STRICT STEP OPT v3 SUMMARY")
    print("=" * 70)
    print(f"Features selected for downstream: {len(final_names)}")
    print(f"Models involved: {list(model_dims.keys())}")
    print("\n--- HONEST PIPELINE PERFORMANCE ---")
    print(f"Peak Strict LOOCV AUC:      {obs_best_auc:.4f} ({best_clf})")
    print(f"Peak Strict LOOCV Accuracy: {obs_best_acc:.4f} ({best_clf})")
    print(f"Strict Permutation p-value: {perm_pval:.4f}")
    print(f"Binomial power:             {config['final_metrics']['binomial_power']:.3f}")
    print(f"\nOutputs saved in {outdir}")

if __name__ == "__main__":
    main()