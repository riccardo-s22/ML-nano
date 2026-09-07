#!/usr/bin/env python3
"""
chirality_descriptor_analysis.py
=================================

Analyze chirality descriptors for ALS vs CTRL classification.

Inputs:
  - chirality_full_descriptors.csv  (from chirality_descriptor_extraction.py)
  - sample_labels.csv               (code, group)

Outputs:
  - univariate_feature_analysis.csv          (Cohen's d, t-test per feature)
  - descriptor_classification_comparison.csv (LOOCV AUC per descriptor × timepoint × classifier)
  - descriptor_combo_results.csv             (multi-descriptor combo LOOCV results)

Usage:
  python chirality_descriptor_analysis.py \
      --descriptors chirality_full_descriptors.csv \
      --labels_csv sample_labels.csv \
      --output_dir ./results
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# The 20 descriptor names
DESCRIPTOR_NAMES = [
    "mean", "max", "center", "median", "integral",
    "std", "range", "cv", "iqr",
    "prominence", "sharpness", "snr", "peak2ring", "contrast",
    "kurtosis", "skewness",
    "grad_center", "grad_mean",
    "fwhm_frac", "bg_integral",
]

CHIRALITY_NAMES = [
    "ch8_3", "ch6_5", "ch7_5", "ch10_2", "ch9_4", "ch8_4",
    "ch7_6", "ch8_6", "ch8_7", "ch9_5", "ch10_3", "ch10_5",
]

TP_LABELS = ["0h", "6h", "24h"]


# ============================================================================
# Helpers
# ============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled standard deviation)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float(abs(group1.mean() - group2.mean()) / pooled_std)


def select_features_fanova_collinear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fanova_k: int = 10,
    collinear_thresh: float = 0.90,
    min_keep: int = 3,
) -> list:
    """In-fold feature selection: fANOVA top-k then greedy collinearity pruning."""
    n_feat = X_train.shape[1]
    if n_feat == 0:
        return []

    var = np.var(X_train, axis=0)
    F = np.zeros(n_feat)
    ok = var > 1e-12
    if ok.any():
        F_ok, _ = f_classif(X_train[:, ok], y_train)
        F[ok] = np.nan_to_num(F_ok, nan=0.0)

    order = np.argsort(-F)
    order = order[: min(fanova_k, len(order))]

    keep = []
    for idx in order:
        idx = int(idx)
        if len(keep) == 0:
            keep.append(idx)
            continue
        x = X_train[:, idx]
        correlated = False
        for j in keep:
            r = np.corrcoef(x, X_train[:, j])[0, 1]
            if np.isnan(r):
                continue
            if abs(r) > collinear_thresh:
                correlated = True
                break
        if not correlated:
            keep.append(idx)

    if len(keep) < min_keep:
        keep = [int(i) for i in order[: min(min_keep, len(order))]]

    return sorted(set(keep))


def loocv_classify(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    feature_selection: bool = False,
    fanova_k: int = 10,
    collinear_thresh: float = 0.90,
) -> dict:
    """
    LOOCV classification. Returns dict with auc, bal_acc, acc.
    """
    N = len(y)
    loo = LeaveOneOut()

    y_scores = np.full(N, np.nan)
    y_preds = np.full(N, -1, dtype=int)

    for train_idx, test_idx in loo.split(X):
        test_i = test_idx[0]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        # In-fold feature selection
        if feature_selection and X_tr.shape[1] > 3:
            keep = select_features_fanova_collinear(
                X_tr, y_tr, fanova_k=fanova_k, collinear_thresh=collinear_thresh
            )
            if len(keep) == 0:
                keep = list(range(X_tr.shape[1]))
            X_tr = X_tr[:, keep]
            X_te = X_te[:, keep]

        # Build classifier
        if classifier_name == "SVM":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=1.0, probability=True,
                            class_weight="balanced", random_state=SEED)),
            ])
        elif classifier_name == "LR":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                           class_weight="balanced",
                                           solver="liblinear", random_state=SEED)),
            ])
        elif classifier_name == "RF":
            clf = RandomForestClassifier(
                n_estimators=500, random_state=SEED,
                class_weight="balanced_subsample", n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[0, 1]
        y_scores[test_i] = prob
        y_preds[test_i] = 1 if prob >= 0.5 else 0

    try:
        auc = roc_auc_score(y, y_scores)
    except ValueError:
        auc = np.nan

    bal_acc = balanced_accuracy_score(y, y_preds)
    acc = accuracy_score(y, y_preds)

    return {"auc": auc, "bal_acc": bal_acc, "acc": acc}


# ============================================================================
# 1. Univariate analysis
# ============================================================================

def run_univariate_analysis(desc_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every feature column, compute t-test and Cohen's d between ALS and CTRL.
    """
    merged = desc_df.merge(labels_df[["code", "group"]], on="code", how="inner")
    als_mask = merged["group"] == "ALS"
    ctrl_mask = merged["group"] == "CTRL"

    feature_cols = [c for c in desc_df.columns if c != "code"]
    rows = []

    for feat in feature_cols:
        vals_als = merged.loc[als_mask, feat].astype(float).values
        vals_ctrl = merged.loc[ctrl_mask, feat].astype(float).values

        # Skip if constant
        if np.std(vals_als) < 1e-15 and np.std(vals_ctrl) < 1e-15:
            continue

        t_stat, p_val = stats.ttest_ind(vals_als, vals_ctrl, equal_var=False)
        d = cohens_d(vals_als, vals_ctrl)

        # Parse feature name: ch8_3_mean_0h -> chirality=ch8_3, descriptor=mean, timepoint=0h
        parts = feat.rsplit("_", 1)  # split on last underscore -> [ch8_3_mean, 0h]
        tp = parts[-1]
        rest = parts[0]
        # Find chirality prefix
        chir = None
        desc = None
        for cn in CHIRALITY_NAMES:
            if rest.startswith(cn + "_"):
                chir = cn
                desc = rest[len(cn) + 1:]
                break

        rows.append({
            "feature": feat,
            "chirality": chir,
            "descriptor": desc,
            "timepoint": tp,
            "t_stat": t_stat,
            "p_val": p_val,
            "cohens_d": d,
            "als_mean": float(vals_als.mean()),
            "ctrl_mean": float(vals_ctrl.mean()),
        })

    df = pd.DataFrame(rows).sort_values("cohens_d", ascending=False).reset_index(drop=True)
    return df


# ============================================================================
# 2. Single-descriptor classification comparison
# ============================================================================

def run_single_descriptor_classification(
    desc_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each descriptor × timepoint × classifier, run LOOCV using all 12
    chirality positions as features (12 features per run).
    """
    merged = desc_df.merge(labels_df[["code", "group"]], on="code", how="inner")
    y = (merged["group"] == "ALS").astype(int).values

    rows = []

    for desc_name in DESCRIPTOR_NAMES:
        for tp in TP_LABELS:
            # Collect features: all chirality positions for this descriptor+timepoint
            feat_cols = [f"{ch}_{desc_name}_{tp}" for ch in CHIRALITY_NAMES]
            feat_cols = [c for c in feat_cols if c in merged.columns]

            if len(feat_cols) == 0:
                continue

            X = merged[feat_cols].values.astype(float)
            # Replace NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            for clf_name in ["SVM", "LR", "RF"]:
                result = loocv_classify(X, y, clf_name, feature_selection=False)
                rows.append({
                    "descriptor": desc_name,
                    "timepoint": tp,
                    "classifier": clf_name,
                    "auc": result["auc"],
                    "bal_acc": result["bal_acc"],
                    "n_features": len(feat_cols),
                })

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


# ============================================================================
# 3. Multi-descriptor combination analysis
# ============================================================================

def run_combo_classification(
    desc_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Test predefined descriptor combinations with and without feature selection.
    """
    merged = desc_df.merge(labels_df[["code", "group"]], on="code", how="inner")
    y = (merged["group"] == "ALS").astype(int).values

    # Define combos
    combos = {
        "all_20": DESCRIPTOR_NAMES,
        "shape_4": ["kurtosis", "skewness", "grad_center", "fwhm_frac"],
        "top5_by_auc": ["skewness", "contrast", "integral", "mean", "grad_mean"],
        "original_3": ["mean", "max", "std"],
        "intensity_5": ["mean", "max", "center", "median", "integral"],
        "morph_5": ["prominence", "sharpness", "snr", "peak2ring", "contrast"],
        "intensity+shape": ["mean", "max", "center", "median", "integral",
                            "kurtosis", "skewness", "grad_center", "fwhm_frac"],
        "morph+shape": ["prominence", "sharpness", "snr", "peak2ring", "contrast",
                        "kurtosis", "skewness", "grad_center", "grad_mean", "fwhm_frac"],
        "comprehensive": DESCRIPTOR_NAMES,  # same as all_20 but tested at different TP combos
    }

    rows = []

    for combo_name, desc_list in combos.items():
        for tp_mode in TP_LABELS + ["all_tp"]:
            if tp_mode == "all_tp":
                tps = TP_LABELS
            else:
                tps = [tp_mode]

            feat_cols = []
            for desc_name in desc_list:
                for tp in tps:
                    for ch in CHIRALITY_NAMES:
                        col = f"{ch}_{desc_name}_{tp}"
                        if col in merged.columns:
                            feat_cols.append(col)

            if len(feat_cols) == 0:
                continue

            X = merged[feat_cols].values.astype(float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            for clf_name in ["SVM", "LR"]:
                # With feature selection
                result_fs = loocv_classify(
                    X, y, clf_name,
                    feature_selection=True,
                    fanova_k=10,
                    collinear_thresh=0.90,
                )
                rows.append({
                    "combo": combo_name,
                    "timepoint": tp_mode,
                    "classifier": clf_name,
                    "fselect": "fanova10+collin90",
                    "auc": result_fs["auc"],
                    "bal_acc": result_fs["bal_acc"],
                    "acc": result_fs["acc"],
                    "n_features": "",  # dynamic per fold
                })

                # Without feature selection (only for smaller combos)
                if len(feat_cols) <= 72:  # max ~6 descriptors × 12 chiralities
                    result_no = loocv_classify(
                        X, y, clf_name,
                        feature_selection=False,
                    )
                    rows.append({
                        "combo": combo_name,
                        "timepoint": tp_mode,
                        "classifier": clf_name,
                        "fselect": "none",
                        "auc": result_no["auc"],
                        "bal_acc": result_no["bal_acc"],
                        "acc": result_no["acc"],
                        "n_features": float(len(feat_cols)),
                    })

    df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chirality descriptor classification analysis"
    )
    parser.add_argument(
        "--descriptors", type=str, required=True,
        help="Path to chirality_full_descriptors.csv"
    )
    parser.add_argument(
        "--labels_csv", type=str, required=True,
        help="CSV with columns: code, group"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save results"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    desc_df = pd.read_csv(args.descriptors)
    labels_df = pd.read_csv(args.labels_csv)

    # Normalize code column (strip whitespace, match formats)
    desc_df["code"] = desc_df["code"].astype(str).str.strip()
    labels_df["code"] = labels_df["code"].astype(str).str.strip()

    n_samples = len(desc_df)
    n_features = len([c for c in desc_df.columns if c != "code"])
    print(f"Loaded {n_samples} samples, {n_features} features")
    print(f"Labels: {labels_df['group'].value_counts().to_dict()}")

    # ---- 1. Univariate analysis ----
    print("\n" + "=" * 60)
    print("1. UNIVARIATE ANALYSIS (t-test + Cohen's d)")
    print("=" * 60)
    univ_df = run_univariate_analysis(desc_df, labels_df)
    univ_path = os.path.join(args.output_dir, "univariate_feature_analysis.csv")
    univ_df.to_csv(univ_path, index=False)
    print(f"Saved: {univ_path} ({len(univ_df)} features)")
    print(f"\nTop 10 by Cohen's d:")
    print(univ_df.head(10)[["feature", "cohens_d", "p_val"]].to_string(index=False))

    # ---- 2. Single descriptor classification ----
    print("\n" + "=" * 60)
    print("2. SINGLE DESCRIPTOR CLASSIFICATION (LOOCV)")
    print("=" * 60)
    single_df = run_single_descriptor_classification(desc_df, labels_df)
    single_path = os.path.join(args.output_dir, "descriptor_classification_comparison.csv")
    single_df.to_csv(single_path, index=False)
    print(f"Saved: {single_path} ({len(single_df)} rows)")
    print(f"\nTop 10 by AUC:")
    print(single_df.head(10)[["descriptor", "timepoint", "classifier", "auc"]].to_string(index=False))

    # ---- 3. Multi-descriptor combos ----
    print("\n" + "=" * 60)
    print("3. MULTI-DESCRIPTOR COMBO CLASSIFICATION (LOOCV)")
    print("=" * 60)
    combo_df = run_combo_classification(desc_df, labels_df)
    combo_path = os.path.join(args.output_dir, "descriptor_combo_results.csv")
    combo_df.to_csv(combo_path, index=False)
    print(f"Saved: {combo_path} ({len(combo_df)} rows)")
    print(f"\nTop 10 by AUC:")
    print(combo_df.head(10)[["combo", "timepoint", "classifier", "fselect", "auc"]].to_string(index=False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_single = single_df.iloc[0]
    best_combo = combo_df.iloc[0]
    print(f"Best single descriptor: {best_single['descriptor']} @ {best_single['timepoint']} "
          f"({best_single['classifier']}) -> AUC={best_single['auc']:.3f}")
    print(f"Best combo: {best_combo['combo']} @ {best_combo['timepoint']} "
          f"({best_combo['classifier']}, {best_combo['fselect']}) -> AUC={best_combo['auc']:.3f}")


if __name__ == "__main__":
    main()
