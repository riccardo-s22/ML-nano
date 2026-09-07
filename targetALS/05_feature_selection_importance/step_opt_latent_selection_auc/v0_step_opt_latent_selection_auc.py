#!/usr/bin/env python3
"""
STEP OPT: Optimal Latent Selection (AUC Ranking + MI Redundancy)
================================================================
1. Aggregates all latent dimensions (512 x 5 models).
2. Ranks features by Univariate AUC (ALS vs Control).
3. Prunes redundancy using Normalized Mutual Information (NMI).
4. Iteratively adds features and runs LOOCV (SVM, RF, LR).
5. Stops when Statistical Power >= 90%.

Usage:
  python step_opt_latent_selection_auc.py \
    --step1_dir ./step1_latent_features \
    --labels_csv sample_labels.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict  # <--- FIXED: Added missing imports

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from statsmodels.stats.power import TTestIndPower

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    df['code'] = df['code'].astype(str)
    df['target'] = (df['group'] == 'ALS').astype(int)
    return df, df['code'].tolist(), df['target'].values

def load_all_latent_features(step1_dir: Path, codes: List[str]) -> Tuple[np.ndarray, List[str]]:
    model_dirs = sorted([d for d in step1_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith(".") and (d / "z_agg.csv").exists()])
    
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {step1_dir}")

    print(f"Found {len(model_dirs)} models. Aggregating features...")
    master_df = pd.DataFrame({'code': codes})
    
    for mdir in model_dirs:
        z_path = mdir / "z_agg.csv"
        df_z = pd.read_csv(z_path)
        df_z['code'] = df_z['code'].astype(str)
        
        # Rename dim0 -> best_model_fold1_dim0
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

def calc_normalized_mi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimates Normalized Mutual Information between two continuous vectors.
    NMI = MI(X,Y) / sqrt(H(X)*H(Y))
    """
    # Reshape for sklearn
    x = x.reshape(-1, 1)
    
    # MI between x and y
    mi_xy = mutual_info_regression(x, y, random_state=SEED)[0]
    
    # Estimate "Self-MI" (Entropy) to normalize
    mi_xx = mutual_info_regression(x, x.ravel(), random_state=SEED)[0]
    mi_yy = mutual_info_regression(y.reshape(-1, 1), y, random_state=SEED)[0]
    
    denom = np.sqrt(mi_xx * mi_yy)
    
    if denom < 1e-9:
        return 0.0
    return mi_xy / denom

def calculate_power(accuracy: float, n_samples: int) -> float:
    """
    Calculates statistical power of the observed accuracy vs random chance (0.5).
    """
    if accuracy <= 0.5: return 0.0
    
    # Effect size (Cohen's h)
    p1 = accuracy
    p2 = 0.5
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    analysis = TTestIndPower()
    # ratio=0 implies one-sample test against a constant
    # alternative='larger' means we are testing if acc > 0.5
    try:
        power = analysis.solve_power(effect_size=h, nobs1=n_samples, alpha=0.05, ratio=0, alternative='larger')
    except:
        power = 0.0
    return float(power)

def run_loocv(X: np.ndarray, y: np.ndarray):
    classifiers = {
        "SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced_subsample"),
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=SEED)
    }
    
    loo = LeaveOneOut()
    preds = {k: [] for k in classifiers}
    
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        for name, clf in classifiers.items():
            clf.fit(X_tr_s, y_tr)
            preds[name].append(clf.predict(X_te_s)[0])
            
    results = {}
    for name in classifiers:
        acc = accuracy_score(y, preds[name])
        results[name] = acc
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./opt_selection_results")
    parser.add_argument("--mi_threshold", type=float, default=0.7, 
                        help="Normalized MI threshold (0.0-1.0) for redundancy.")
    parser.add_argument("--target_power", type=float, default=0.90)
    parser.add_argument("--max_features", type=int, default=30)
    args = parser.parse_args()
    
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load
    print("Loading data...")
    df_labels, codes, y = load_labels(args.labels_csv)
    X_all, feature_names = load_all_latent_features(Path(args.step1_dir), codes)
    print(f"Total features: {X_all.shape[1]}")
    
    # 2. Rank by AUC
    print("Ranking features by AUC...")
    auc_scores = []
    for i in tqdm(range(X_all.shape[1])):
        score = roc_auc_score(y, X_all[:, i])
        # Rank by distance from 0.5 (random)
        ranking_score = abs(score - 0.5)
        auc_scores.append((i, ranking_score, score))
        
    auc_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [x[0] for x in auc_scores]
    
    # Save Ranking
    ranking_df = pd.DataFrame([{
        "feature": feature_names[x[0]], 
        "auc": x[2], 
        "ranking_score": x[1]
    } for x in auc_scores])
    ranking_df.to_csv(outdir / "feature_ranking_auc.csv", index=False)
    
    # 3. Iterative Selection
    print(f"\nStarting Selection (Stop at Power >= {args.target_power} or MI > {args.mi_threshold})...")
    
    selected_indices = []
    history = []
    
    pbar = tqdm(total=args.max_features)
    
    for rank_idx, feat_idx in enumerate(ranked_indices):
        if len(selected_indices) >= args.max_features:
            print("Max features reached.")
            break
            
        candidate_vec = X_all[:, feat_idx]
        candidate_name = feature_names[feat_idx]
        
        # --- Redundancy Check (NMI) ---
        is_redundant = False
        max_mi = 0.0
        
        for sel_idx in selected_indices:
            sel_vec = X_all[:, sel_idx]
            nmi = calc_normalized_mi(candidate_vec, sel_vec)
            if nmi > max_mi: max_mi = nmi
            
            if nmi > args.mi_threshold:
                is_redundant = True
                break
        
        if is_redundant:
            continue
            
        # --- Add & Eval ---
        selected_indices.append(feat_idx)
        X_curr = X_all[:, selected_indices]
        
        res = run_loocv(X_curr, y)
        best_acc = max(res.values())
        power = calculate_power(best_acc, len(y))
        
        history.append({
            "n_features": len(selected_indices),
            "added_feature": candidate_name,
            "auc_of_feature": auc_scores[rank_idx][2],
            "max_nmi_with_selected": max_mi,
            "best_acc": best_acc,
            "power": power,
            "svm_acc": res['SVM'],
            "rf_acc": res['RF'],
            "lr_acc": res['LR']
        })
        
        pbar.update(1)
        pbar.set_description(f"Feat {len(selected_indices)} | Acc {best_acc:.2f} | Pwr {power:.2f}")
        
        if power >= args.target_power:
            print(f"\nTarget Power Reached! ({power:.3f})")
            break
            
    pbar.close()
    
    # Save Results
    pd.DataFrame(history).to_csv(outdir / "selection_history.csv", index=False)
    
    # Save final list
    final_names = [feature_names[i] for i in selected_indices]
    pd.DataFrame({"feature": final_names}).to_csv(outdir / "final_selected_features.csv", index=False)
    
    print(f"\nDone. Results in {outdir}")

if __name__ == "__main__":
    main()