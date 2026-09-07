#!/usr/bin/env python3
"""
ROI-BASED CLASSIFICATION PIPELINE
=================================

Trains SVM, Random Forest, and Logistic Regression classifiers using
ROI-derived features from spectral data. Separate models per timepoint.

Features per ROI:
- mean_intensity: Average intensity within ROI
- max_intensity: Maximum intensity within ROI (optional)

Evaluation:
- Leave-One-Out Cross-Validation (LOOCV) for robust performance on small samples
- Metrics: Accuracy, Balanced Accuracy, AUC, Sensitivity, Specificity

USAGE:
------
python roi_classification_pipeline.py \
    --rois_csv rois_filtered_by_mse.csv \
    --tp_dirs out_0h,out_6h,out_24h \
    --labels_csv sample_labels.csv \
    --output_dir ./roi_classification_results
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_excel_as_array(filepath: str) -> np.ndarray:
    """Load Excel file and return as numpy array (skip first row/col)."""
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    
    data = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0:  # Skip header row
            continue
        
        row_data = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0:  # Skip first column (emission labels)
                continue
            if isinstance(cell, (int, float)) and cell is not None:
                row_data.append(float(cell))
            else:
                row_data.append(0.0)
        
        if len(row_data) > 0:
            data.append(row_data)
    
    return np.array(data, dtype=np.float32)


def load_spectral_axes(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load emission and excitation wavelengths from Excel."""
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    
    excitation_axis = []
    emission_axis = []
    
    def extract_number(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
            matches = re.findall(r'[\d.]+', value)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    pass
        return None
    
    first_row = list(next(ws.iter_rows(min_row=1, max_row=1, values_only=True)))
    
    for col_idx, cell in enumerate(first_row):
        if col_idx == 0:
            continue
        num = extract_number(cell)
        if num is not None:
            excitation_axis.append(num)
    
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0:
            continue
        num = extract_number(row[0])
        if num is not None:
            emission_axis.append(num)
    
    return np.array(emission_axis), np.array(excitation_axis)


def find_file(directory: str, code: str) -> str:
    """Find Excel file for given sample code."""
    for suffix in ['.xlsx', '.xls']:
        path = os.path.join(directory, f"{code}{suffix}")
        if os.path.exists(path):
            return path
    
    for f in os.listdir(directory):
        if f.startswith(code) and (f.endswith('.xlsx') or f.endswith('.xls')):
            return os.path.join(directory, f)
    
    raise FileNotFoundError(f"No file found for code {code} in {directory}")


def nm_to_pixel(value_nm: float, axis_nm: np.ndarray) -> int:
    """Convert wavelength in nm to pixel index."""
    idx = np.argmin(np.abs(axis_nm - value_nm))
    return int(idx)


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_roi_features(
    spectral_data: np.ndarray,
    rois: pd.DataFrame,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    include_max: bool = True,
    use_pixel_coords: bool = False
) -> np.ndarray:
    """
    Extract features from ROIs in spectral data.
    
    Args:
        spectral_data: 2D array [emission, excitation]
        rois: DataFrame with ROI definitions 
        emission_axis: Array of emission wavelengths
        excitation_axis: Array of excitation wavelengths
        include_max: Whether to include max intensity alongside mean
        use_pixel_coords: If True, use y0/y1/x0/x1 directly as pixel indices
    
    Returns:
        1D feature vector [roi1_mean, (roi1_max), roi2_mean, (roi2_max), ...]
    """
    features = []
    
    for _, roi in rois.iterrows():
        if use_pixel_coords or len(excitation_axis) == 0:
            # Use pixel coordinates directly from ROI definition
            em_min_px = int(roi['y0'])
            em_max_px = int(roi['y1'])
            exc_min_px = int(roi['x0'])
            exc_max_px = int(roi['x1'])
        else:
            # Convert nm to pixel indices
            em_min_px = nm_to_pixel(roi['em_min_nm'], emission_axis)
            em_max_px = nm_to_pixel(roi['em_max_nm'], emission_axis)
            exc_min_px = nm_to_pixel(roi['exc_min_nm'], excitation_axis)
            exc_max_px = nm_to_pixel(roi['exc_max_nm'], excitation_axis)
        
        # Ensure valid bounds
        em_min_px = max(0, min(em_min_px, em_max_px))
        em_max_px = min(spectral_data.shape[0] - 1, max(em_min_px, em_max_px))
        exc_min_px = max(0, min(exc_min_px, exc_max_px))
        exc_max_px = min(spectral_data.shape[1] - 1, max(exc_min_px, exc_max_px))
        
        # Extract ROI region
        roi_region = spectral_data[em_min_px:em_max_px+1, exc_min_px:exc_max_px+1]
        
        if roi_region.size > 0:
            features.append(roi_region.mean())
            if include_max:
                features.append(roi_region.max())
        else:
            features.append(0.0)
            if include_max:
                features.append(0.0)
    
    return np.array(features)


def build_feature_matrix(
    codes: List[str],
    tp_dir: str,
    rois: pd.DataFrame,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    include_max: bool = True,
    use_pixel_coords: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix for all samples at one timepoint.
    
    Returns:
        X: Feature matrix [n_samples, n_features]
        feature_names: List of feature names
    """
    X_list = []
    
    for code in codes:
        filepath = find_file(tp_dir, code)
        spectral_data = load_excel_as_array(filepath)
        
        features = extract_roi_features(
            spectral_data, rois, emission_axis, excitation_axis, 
            include_max, use_pixel_coords
        )
        X_list.append(features)
    
    X = np.vstack(X_list)
    
    # Generate feature names
    feature_names = []
    for i, (_, roi) in enumerate(rois.iterrows()):
        em_nm = roi.get('em_centroid_nm', roi.get('cy', i))
        exc_nm = roi.get('exc_centroid_nm', roi.get('cx', i))
        feature_names.append(f"ROI{i+1}_em{em_nm:.0f}_exc{exc_nm:.0f}_mean")
        if include_max:
            feature_names.append(f"ROI{i+1}_em{em_nm:.0f}_exc{exc_nm:.0f}_max")
    
    return X, feature_names


# ==============================================================================
# CLASSIFICATION
# ==============================================================================

def train_and_evaluate_loocv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model,
    output_dir: Path,
    timepoint: str
) -> Dict:
    """
    Train model using LOOCV and return evaluation metrics.
    """
    loo = LeaveOneOut()
    
    y_pred = []
    y_scores = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Clone model for fresh training
        if model_name == 'SVM':
            clf = SVC(kernel='linear', C=1.0, probability=True, random_state=SEED)
        elif model_name == 'RF':
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
        elif model_name == 'LR':
            clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=SEED)
        else:
            clf = model
        
        clf.fit(X_train_scaled, y_train)
        
        y_pred.append(clf.predict(X_test_scaled)[0])
        
        if hasattr(clf, 'predict_proba'):
            y_scores.append(clf.predict_proba(X_test_scaled)[0, 1])
        elif hasattr(clf, 'decision_function'):
            y_scores.append(clf.decision_function(X_test_scaled)[0])
        else:
            y_scores.append(y_pred[-1])
    
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    
    try:
        auc = roc_auc_score(y, y_scores)
    except:
        auc = 0.5
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    results = {
        'model': model_name,
        'timepoint': timepoint,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'n_samples': len(y),
        'n_features': X.shape[1]
    }
    
    return results, y_pred, y_scores


def get_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Train final model on all data and extract feature importance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_name == 'SVM':
        clf = SVC(kernel='linear', C=1.0, random_state=SEED)
        clf.fit(X_scaled, y)
        importance = np.abs(clf.coef_[0])
    elif model_name == 'RF':
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
        clf.fit(X_scaled, y)
        importance = clf.feature_importances_
    elif model_name == 'LR':
        clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=SEED)
        clf.fit(X_scaled, y)
        importance = np.abs(clf.coef_[0])
    else:
        importance = np.zeros(len(feature_names))
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_confusion_matrix(cm, output_path, title="Confusion Matrix"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['CTRL', 'ALS'], yticklabels=['CTRL', 'ALS'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results_dict, y_true, output_path, title="ROC Curves"):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'SVM': 'blue', 'RF': 'green', 'LR': 'red'}
    
    for model_name, (y_scores, auc_val) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, color=colors.get(model_name, 'gray'), 
                label=f'{model_name} (AUC={auc_val:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df, output_path, title="Feature Importance", top_n=15):
    """Plot feature importance bar chart."""
    df = importance_df.head(top_n).copy()
    df = df.iloc[::-1]  # Reverse for horizontal bar
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    
    bars = ax.barh(df['feature'], df['importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_barplot(results_df, output_path, metric='auc'):
    """Plot comparison of models across timepoints."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot for grouped bar
    pivot_df = results_df.pivot(index='timepoint', columns='model', values=metric)
    pivot_df = pivot_df.reindex(['0h', '6h', '24h'])
    
    pivot_df.plot(kind='bar', ax=ax, rot=0, width=0.8)
    
    ax.set_ylabel(metric.upper())
    ax.set_xlabel('Timepoint')
    ax.set_title(f'Model Comparison: {metric.upper()} by Timepoint')
    ax.legend(title='Model')
    ax.set_ylim([0, 1])
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ROI-based classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--rois_csv', type=str, required=True,
                       help='CSV file with ROI definitions (from importance pipeline)')
    parser.add_argument('--tp_dirs', type=str, required=True,
                       help='Comma-separated paths to 0h,6h,24h directories')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV with columns: code/sample_id, group (ALS/CTRL)')
    parser.add_argument('--output_dir', type=str, default='./roi_classification_results')
    parser.add_argument('--include_max', action='store_true',
                       help='Include max intensity alongside mean')
    parser.add_argument('--use_pixel_coords', action='store_true',
                       help='Use pixel coordinates (y0,y1,x0,x1) instead of nm coordinates')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tp_dirs = args.tp_dirs.split(',')
    tp_labels = ['0h', '6h', '24h']
    assert len(tp_dirs) == 3, "Must provide exactly 3 timepoint directories"
    
    print(f"{'='*70}")
    print("ROI-BASED CLASSIFICATION PIPELINE")
    print(f"{'='*70}")
    print(f"ROIs file: {args.rois_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Include max intensity: {args.include_max}")
    print(f"Use pixel coordinates: {args.use_pixel_coords}")
    
    # Load ROIs
    rois_df = pd.read_csv(args.rois_csv)
    print(f"\nLoaded {len(rois_df)} ROIs total")
    
    for tp in tp_labels:
        n_rois = len(rois_df[rois_df['timepoint'] == tp])
        print(f"  {tp}: {n_rois} ROIs")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    if 'sample_id' in labels_df.columns and 'code' not in labels_df.columns:
        labels_df['code'] = labels_df['sample_id']
    
    codes = labels_df['code'].astype(str).tolist()
    groups = labels_df['group'].tolist()
    y = np.array([1 if g == 'ALS' else 0 for g in groups])
    
    print(f"\nLoaded {len(codes)} samples")
    print(f"  ALS: {sum(y)}, CTRL: {len(y) - sum(y)}")
    
    # Load spectral axes from first file
    first_file = find_file(tp_dirs[0], codes[0])
    emission_axis, excitation_axis = load_spectral_axes(first_file)
    print(f"\nSpectral grid: {len(emission_axis)} emission × {len(excitation_axis)} excitation")
    
    # Determine if we should use pixel coordinates
    use_pixel_coords = args.use_pixel_coords
    if len(excitation_axis) == 0:
        print("WARNING: Could not parse excitation axis from Excel. Using pixel coordinates.")
        use_pixel_coords = True
    
    if use_pixel_coords:
        print("Using ROI pixel coordinates (y0, y1, x0, x1) directly.")
        # Create dummy axes based on data shape
        sample_data = load_excel_as_array(first_file)
        emission_axis = np.arange(sample_data.shape[0])
        excitation_axis = np.arange(sample_data.shape[1])
    
    # Models to evaluate
    models = {
        'SVM': SVC(kernel='linear', C=1.0, probability=True, random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED),
        'LR': LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=SEED)
    }
    
    all_results = []
    
    # Process each timepoint separately
    for tp_idx, (tp, tp_dir) in enumerate(zip(tp_labels, tp_dirs)):
        print(f"\n{'='*70}")
        print(f"TIMEPOINT: {tp}")
        print(f"{'='*70}")
        
        # Get ROIs for this timepoint
        tp_rois = rois_df[rois_df['timepoint'] == tp].copy()
        
        if len(tp_rois) == 0:
            print(f"  No ROIs for {tp}, skipping...")
            continue
        
        print(f"  ROIs: {len(tp_rois)}")
        
        # Build feature matrix
        print(f"  Extracting features...")
        X, feature_names = build_feature_matrix(
            codes, tp_dir, tp_rois, emission_axis, excitation_axis,
            include_max=args.include_max, use_pixel_coords=use_pixel_coords
        )
        
        print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Create timepoint output directory
        tp_output_dir = output_dir / tp
        tp_output_dir.mkdir(exist_ok=True)
        
        # Save feature matrix
        feature_df = pd.DataFrame(X, columns=feature_names)
        feature_df.insert(0, 'code', codes)
        feature_df.insert(1, 'group', groups)
        feature_df.insert(2, 'label', y)
        feature_df.to_csv(tp_output_dir / 'feature_matrix.csv', index=False)
        
        # Train and evaluate each model
        roc_data = {}
        
        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            
            results, y_pred, y_scores = train_and_evaluate_loocv(
                X, y, model_name, model, tp_output_dir, tp
            )
            
            all_results.append(results)
            roc_data[model_name] = (y_scores, results['auc'])
            
            print(f"    Accuracy: {results['accuracy']:.3f}")
            print(f"    Balanced Accuracy: {results['balanced_accuracy']:.3f}")
            print(f"    AUC: {results['auc']:.3f}")
            print(f"    Sensitivity: {results['sensitivity']:.3f}")
            print(f"    Specificity: {results['specificity']:.3f}")
            
            # Confusion matrix
            cm = np.array([[results['tn'], results['fp']], 
                          [results['fn'], results['tp']]])
            plot_confusion_matrix(
                cm, tp_output_dir / f'confusion_matrix_{model_name}.png',
                title=f'{tp} - {model_name} Confusion Matrix'
            )
            
            # Feature importance
            importance_df = get_feature_importance(X, y, feature_names, model_name)
            importance_df.to_csv(tp_output_dir / f'feature_importance_{model_name}.csv', index=False)
            
            plot_feature_importance(
                importance_df, tp_output_dir / f'feature_importance_{model_name}.png',
                title=f'{tp} - {model_name} Feature Importance'
            )
            
            # Save predictions
            pred_df = pd.DataFrame({
                'code': codes,
                'group': groups,
                'y_true': y,
                'y_pred': y_pred,
                'y_score': y_scores,
                'correct': y == y_pred
            })
            pred_df.to_csv(tp_output_dir / f'predictions_{model_name}.csv', index=False)
        
        # ROC curves for this timepoint
        plot_roc_curves(
            roc_data, y, tp_output_dir / 'roc_curves.png',
            title=f'{tp} - ROC Curves (LOOCV)'
        )
    
    # ==============================================================================
    # TEMPORAL DYNAMICS MODEL (COMBINED)
    # ==============================================================================
    
    print(f"\n{'='*70}")
    print("TEMPORAL DYNAMICS MODEL")
    print(f"{'='*70}")
    
    # Build combined feature matrix with deltas
    # Use intersection of ROIs that appear at similar locations across timepoints
    # Or use all ROIs from each timepoint + compute deltas
    
    # Strategy: Extract features for ALL samples at ALL timepoints, then compute deltas
    all_tp_features = {}
    all_tp_feature_names = {}
    
    for tp_idx, (tp, tp_dir) in enumerate(zip(tp_labels, tp_dirs)):
        tp_rois = rois_df[rois_df['timepoint'] == tp].copy()
        if len(tp_rois) > 0:
            X_tp, names_tp = build_feature_matrix(
                codes, tp_dir, tp_rois, emission_axis, excitation_axis,
                include_max=args.include_max, use_pixel_coords=use_pixel_coords
            )
            all_tp_features[tp] = X_tp
            all_tp_feature_names[tp] = [f"{tp}_{n}" for n in names_tp]
    
    # Build combined feature matrix: [0h_features, 6h_features, 24h_features, delta_6h_0h, delta_24h_0h]
    combined_features = []
    combined_names = []
    
    for tp in tp_labels:
        if tp in all_tp_features:
            combined_features.append(all_tp_features[tp])
            combined_names.extend(all_tp_feature_names[tp])
    
    # Add delta features (if we have multiple timepoints)
    if '0h' in all_tp_features and '6h' in all_tp_features:
        # Match ROIs by position (use smaller dimension)
        min_cols = min(all_tp_features['0h'].shape[1], all_tp_features['6h'].shape[1])
        delta_6h_0h = all_tp_features['6h'][:, :min_cols] - all_tp_features['0h'][:, :min_cols]
        combined_features.append(delta_6h_0h)
        for i in range(min_cols):
            combined_names.append(f"delta_6h_0h_feat{i}")
    
    if '0h' in all_tp_features and '24h' in all_tp_features:
        min_cols = min(all_tp_features['0h'].shape[1], all_tp_features['24h'].shape[1])
        delta_24h_0h = all_tp_features['24h'][:, :min_cols] - all_tp_features['0h'][:, :min_cols]
        combined_features.append(delta_24h_0h)
        for i in range(min_cols):
            combined_names.append(f"delta_24h_0h_feat{i}")
    
    if len(combined_features) > 0:
        X_combined = np.hstack(combined_features)
        print(f"\nCombined feature matrix: {X_combined.shape[0]} samples × {X_combined.shape[1]} features")
        print(f"  (includes static features + temporal deltas)")
        
        # Create combined output directory
        combined_dir = output_dir / 'combined_temporal'
        combined_dir.mkdir(exist_ok=True)
        
        # Save combined feature matrix
        combined_feature_df = pd.DataFrame(X_combined, columns=combined_names)
        combined_feature_df.insert(0, 'code', codes)
        combined_feature_df.insert(1, 'group', groups)
        combined_feature_df.insert(2, 'label', y)
        combined_feature_df.to_csv(combined_dir / 'feature_matrix_combined.csv', index=False)
        
        # Train models
        roc_data_combined = {}
        
        for model_name, model in models.items():
            print(f"\n  Training {model_name} (combined)...")
            
            results, y_pred, y_scores = train_and_evaluate_loocv(
                X_combined, y, model_name, model, combined_dir, 'combined'
            )
            
            results['timepoint'] = 'combined'
            all_results.append(results)
            roc_data_combined[model_name] = (y_scores, results['auc'])
            
            print(f"    Accuracy: {results['accuracy']:.3f}")
            print(f"    Balanced Accuracy: {results['balanced_accuracy']:.3f}")
            print(f"    AUC: {results['auc']:.3f}")
            print(f"    Sensitivity: {results['sensitivity']:.3f}")
            print(f"    Specificity: {results['specificity']:.3f}")
            
            # Confusion matrix
            cm = np.array([[results['tn'], results['fp']], 
                          [results['fn'], results['tp']]])
            plot_confusion_matrix(
                cm, combined_dir / f'confusion_matrix_{model_name}.png',
                title=f'Combined Temporal - {model_name} Confusion Matrix'
            )
            
            # Feature importance
            importance_df = get_feature_importance(X_combined, y, combined_names, model_name)
            importance_df.to_csv(combined_dir / f'feature_importance_{model_name}.csv', index=False)
            
            plot_feature_importance(
                importance_df, combined_dir / f'feature_importance_{model_name}.png',
                title=f'Combined Temporal - {model_name} Feature Importance', top_n=20
            )
            
            # Save predictions
            pred_df = pd.DataFrame({
                'code': codes,
                'group': groups,
                'y_true': y,
                'y_pred': y_pred,
                'y_score': y_scores,
                'correct': y == y_pred
            })
            pred_df.to_csv(combined_dir / f'predictions_{model_name}.csv', index=False)
        
        # ROC curves
        plot_roc_curves(
            roc_data_combined, y, combined_dir / 'roc_curves.png',
            title='Combined Temporal - ROC Curves (LOOCV)'
        )
    
    # ==============================================================================
    # SUMMARY ACROSS TIMEPOINTS
    # ==============================================================================
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY (LOOCV)")
    print("="*70)
    
    for tp in tp_labels + ['combined']:
        tp_results = results_df[results_df['timepoint'] == tp]
        if len(tp_results) == 0:
            continue
        
        print(f"\n{tp}:")
        print("-" * 50)
        for _, row in tp_results.iterrows():
            print(f"  {row['model']:3s}: Acc={row['accuracy']:.3f}, "
                  f"BalAcc={row['balanced_accuracy']:.3f}, "
                  f"AUC={row['auc']:.3f}, "
                  f"Sens={row['sensitivity']:.3f}, Spec={row['specificity']:.3f}")
    
    # Best model per timepoint
    print("\n" + "="*70)
    print("BEST MODEL PER TIMEPOINT (by AUC)")
    print("="*70)
    
    for tp in tp_labels + ['combined']:
        tp_results = results_df[results_df['timepoint'] == tp]
        if len(tp_results) == 0:
            continue
        
        best = tp_results.loc[tp_results['auc'].idxmax()]
        print(f"  {tp}: {best['model']} (AUC={best['auc']:.3f})")
    
    # Comparison plots
    if len(results_df) > 0:
        plot_comparison_barplot(results_df, output_dir / 'comparison_auc.png', metric='auc')
        plot_comparison_barplot(results_df, output_dir / 'comparison_balanced_accuracy.png', 
                               metric='balanced_accuracy')
    
    # Save config
    config = {
        'rois_csv': args.rois_csv,
        'n_samples': len(codes),
        'n_als': int(sum(y)),
        'n_ctrl': int(len(y) - sum(y)),
        'include_max': args.include_max,
        'rois_per_timepoint': {tp: len(rois_df[rois_df['timepoint'] == tp]) for tp in tp_labels},
        'features_per_timepoint': {tp: len(rois_df[rois_df['timepoint'] == tp]) * (2 if args.include_max else 1) 
                                   for tp in tp_labels}
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
