#!/usr/bin/env python3
"""
MULTI-SCALE CHIRALITY-ANCHORED FEATURE EXTRACTION PIPELINE (v3 - No Heights)
==============================================================

Updates vs. v2:
  - IGNORES 'Height' column in coordinates file. 
    All chiralities are treated as having equal weight (1.0) for any
    weighted calculations, focusing purely on their spectral position.

Original concept:
Extracts physics-informed features from EEM spectroscopy data using
the 12 known carbon nanotube chirality positions as anchors.

Three scales of information:
1. LOCAL:    Spectral behavior at/near each chirality position
2. PAIRWISE: Interactions between chirality pairs (ratios, corridors)
3. TEMPORAL: Dynamic changes across 0h -> 6h -> 24h

Classification uses properly nested LOOCV (no direct data leakage).
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------

# Sensor grid definitions (Fixed for your instrument)
EM_AXIS = np.linspace(852.6, 1675.9, 512)  # 512 pixels
EX_AXIS = np.linspace(500, 850, 71)        # 71 pixels

# Feature Extraction Parameters
# Region size for "Local" features (in pixels)
# +/- 10nm emission is roughly +/- 6 pixels
# +/- 10nm excitation is roughly +/- 2 pixels
ROI_H_PIXELS = 6 
ROI_W_PIXELS = 2

# -----------------------------------------------------------------------------
# 2. Data Loading Helpers
# -----------------------------------------------------------------------------

def load_chirality_coords(filepath):
    """
    Parses the Coordinates_DNA.txt file.
    
    MODIFIED: Ignores 'Height' value and sets it to 1.0 for all peaks.
    Returns a dict: { (n,m): {'Em': float, 'Ex': float, 'Height': 1.0} }
    """
    coords = {}
    current_chirality = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Match header [n,m]
            match_header = re.match(r'\[(\d+),(\d+)\]', line)
            if match_header:
                current_chirality = (int(match_header.group(1)), int(match_header.group(2)))
                coords[current_chirality] = {}
                continue
            
            if current_chirality:
                # Parse Key = Value
                if '=' in line:
                    key, val = [x.strip() for x in line.split('=')]
                    if key == 'Height':
                        # FORCE IGNORE HEIGHT
                        coords[current_chirality][key] = 1.0
                    else:
                        coords[current_chirality][key] = float(val)

    # Convert to DataFrame for easier handling
    data = []
    for (n,m), props in coords.items():
        data.append({
            'n': n, 'm': m,
            'Em': props['Emission'],
            'Ex': props['Excitation'],
            'Height': 1.0 # Enforce 1.0
        })
    
    return pd.DataFrame(data)

def load_eem_matrix(filepath):
    """
    Loads a single Excel file into a (512, 71) numpy array.
    Expects standard format with headers/indices to be skipped.
    """
    try:
        wb = load_workbook(filepath, data_only=True)
        ws = wb.active
        
        data = []
        # Skip header row (row 1) and index column (col A)
        for row in ws.iter_rows(min_row=2, min_col=2, values_only=True):
            # Replace None with 0, ensure float
            cleaned_row = [float(x) if x is not None else 0.0 for x in row]
            data.append(cleaned_row)
            
        arr = np.array(data)
        
        # Validation
        if arr.shape != (512, 71):
            # Try to crop or pad if slightly off (simple robustness)
            if arr.shape[0] > 512: arr = arr[:512, :]
            if arr.shape[1] > 71: arr = arr[:, :71]
            # If smaller, pad with 0
            if arr.shape[0] < 512 or arr.shape[1] < 71:
                padded = np.zeros((512, 71))
                h, w = arr.shape
                padded[:h, :w] = arr
                arr = padded
                
        return arr
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.zeros((512, 71))

def get_roi_indices(em_center, ex_center):
    """
    Returns slice objects for extracting a region around a peak.
    """
    # Find closest index
    r_idx = (np.abs(EM_AXIS - em_center)).argmin()
    c_idx = (np.abs(EX_AXIS - ex_center)).argmin()
    
    # Define bounds
    r_start = max(0, r_idx - ROI_H_PIXELS)
    r_end = min(512, r_idx + ROI_H_PIXELS + 1)
    
    c_start = max(0, c_idx - ROI_W_PIXELS)
    c_end = min(71, c_idx + ROI_W_PIXELS + 1)
    
    return slice(r_start, r_end), slice(c_start, c_end)

# -----------------------------------------------------------------------------
# 3. Feature Extraction Core
# -----------------------------------------------------------------------------

def extract_features_for_sample(eem_dict, chirality_df, config):
    """
    Extracts all features for one subject (containing 0h, 6h, 24h EEMs).
    
    eem_dict: {'0h': array, '6h': array, '24h': array}
    """
    features = {}
    
    # Pre-calculate total intensity for normalization (if enabled)
    totals = {}
    if config['normalization'] == 'global_sum':
        for tp, mat in eem_dict.items():
            totals[tp] = np.sum(mat) + 1e-9
    elif config['normalization'] == 'peak_sum':
        for tp, mat in eem_dict.items():
            peak_sum = 0
            for _, row in chirality_df.iterrows():
                r_sl, c_sl = get_roi_indices(row['Em'], row['Ex'])
                peak_sum += np.max(mat[r_sl, c_sl])
            totals[tp] = peak_sum + 1e-9
    else:
        for tp in eem_dict.items(): totals[tp[0]] = 1.0

    # --- SCALE 1: LOCAL PEAK FEATURES ---
    # For each chirality at each timepoint
    peak_vals = {} # Store for pairwise ratios later
    
    for _, row in chirality_df.iterrows():
        name = f"({int(row['n'])},{int(row['m'])})"
        
        for tp in ['0h', '6h', '24h']:
            mat = eem_dict[tp]
            norm_mat = mat / totals[tp] # Normalized EEM
            
            r_sl, c_sl = get_roi_indices(row['Em'], row['Ex'])
            roi = norm_mat[r_sl, c_sl]
            
            # 1. Peak Intensity (Max)
            val = np.max(roi)
            feat_name = f"{name}_{tp}_Peak"
            if not config['drop_absolute']: features[feat_name] = val
            
            # Store raw value for ratios (un-normalized often better for ratios, but normalized is safer)
            # Actually, for ratios, normalization cancels out if same TP.
            peak_vals[(name, tp)] = val
            
            # 2. Local Shift (Center of Mass vs Theoretical)
            # Simple weighted average of indices
            if np.sum(roi) > 0:
                y_idxs, x_idxs = np.indices(roi.shape)
                y_cm = np.average(y_idxs, weights=roi)
                x_cm = np.average(x_idxs, weights=roi)
                
                # Convert back to global indices then nm
                # (Skipping complex nm conversion for stability, using pixel shift)
                # Theoretical center is exactly in middle of ROI (ROI_H_PIXELS, ROI_W_PIXELS)
                shift_em = y_cm - ROI_H_PIXELS
                shift_ex = x_cm - ROI_W_PIXELS
                
                features[f"{name}_{tp}_ShiftEm"] = shift_em
                features[f"{name}_{tp}_ShiftEx"] = shift_ex
                
                # 3. Local FWHM / Spread (Standard Deviation)
                # Proxy for broadening
                spread = np.sqrt(np.average((y_idxs - y_cm)**2 + (x_idxs - x_cm)**2, weights=roi))
                features[f"{name}_{tp}_Width"] = spread
            else:
                features[f"{name}_{tp}_ShiftEm"] = 0
                features[f"{name}_{tp}_ShiftEx"] = 0
                features[f"{name}_{tp}_Width"] = 0

    # --- SCALE 2: PAIRWISE RATIOS ---
    # Ratios between all pairs of chiralities at same timepoint
    chirality_names = [f"({int(row['n'])},{int(row['m'])})" for _, row in chirality_df.iterrows()]
    
    for tp in ['0h', '6h', '24h']:
        for c1, c2 in combinations(chirality_names, 2):
            v1 = peak_vals.get((c1, tp), 1e-9)
            v2 = peak_vals.get((c2, tp), 1e-9)
            
            # Ratio (always put larger on top or consistent order? Consistent order is better for ML)
            if v2 > 1e-12:
                ratio = v1 / v2
            else:
                ratio = 0
            
            features[f"Ratio_{c1}/{c2}_{tp}"] = ratio

    # --- SCALE 3: TEMPORAL DYNAMICS ---
    # Changes of Scale 1 & 2 features over time
    # We calculate Delta (Diff) and Ratio (Fold Change)
    
    # Get list of static features we just created
    static_keys = list(features.keys())
    
    for key in static_keys:
        if '_0h_' in key:
            base_name = key.replace('_0h_', '_')
            v0 = features[key]
            
            # Find corresponding 6h and 24h
            k6 = key.replace('_0h_', '_6h_')
            k24 = key.replace('_0h_', '_24h_')
            
            if k6 in features and k24 in features:
                v6 = features[k6]
                v24 = features[k24]
                
                # 0h -> 6h
                features[f"Delta_0-6_{base_name}"] = v6 - v0
                features[f"Fold_0-6_{base_name}"] = v6 / (v0 + 1e-9)
                
                # 6h -> 24h
                features[f"Delta_6-24_{base_name}"] = v24 - v6
                features[f"Fold_6-24_{base_name}"] = v24 / (v6 + 1e-9)
                
                # 0h -> 24h (Total change)
                features[f"Delta_0-24_{base_name}"] = v24 - v0
                
                # Slope (Linear fit of 0, 6, 24)
                # x = [0, 6, 24], y = [v0, v6, v24]
                # Simple slope approx
                slope = (v24 - v0) / 24.0
                features[f"Slope_{base_name}"] = slope

    return features

# -----------------------------------------------------------------------------
# 4. Main Pipeline
# -----------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multiscale Chirality Feature Extraction (No Heights)")
    parser.add_argument('--tp_dirs', required=True, help="Comma-separated: path_to_0h,path_to_6h,path_to_24h")
    parser.add_argument('--coords_file', required=True, help="Path to Coordinates_DNA.txt")
    parser.add_argument('--labels_csv', required=True, help="Path to sample_labels.csv")
    parser.add_argument('--output_dir', required=True, help="Output folder")
    
    # Options
    parser.add_argument('--normalization', default='global_sum', choices=['global_sum', 'peak_sum', 'none'], help="How to normalize EEM intensity")
    parser.add_argument('--drop_absolute', action='store_true', help="If true, drops raw intensity features, keeping only ratios/shifts")
    parser.add_argument('--stability_top_n', type=int, default=20, help="Number of stable features to report")
    
    args = parser.parse_args()
    
    # Setup
    tp_paths = [x.strip() for x in args.tp_dirs.split(',')]
    if len(tp_paths) != 3:
        print("Error: Must provide exactly 3 timepoint directories (0h, 6h, 24h)")
        sys.exit(1)
        
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Metadata
    print("Loading coordinates (IGNORING HEIGHTS)...")
    chirality_df = load_chirality_coords(args.coords_file)
    print(f"Loaded {len(chirality_df)} chiralities.")
    
    print("Loading labels...")
    labels_df = pd.read_csv(args.labels_csv)
    # Assume cols: 'code', 'group'
    # Map code -> group
    code_map = dict(zip(labels_df['code'].astype(str).str.strip(), labels_df['group']))
    
    # 2. Process Samples
    X_rows = []
    y = []
    codes = []
    
    # Find common codes across all TPs
    # (Assuming filenames are code.xlsx)
    files_0h = [f for f in os.listdir(tp_paths[0]) if f.endswith('.xlsx') and not f.startswith('~')]
    
    print(f"Processing {len(files_0h)} samples...")
    
    for fname in files_0h:
        code = fname.replace('.xlsx', '')
        if code not in code_map:
            continue # Skip unlabeled
            
        # Check existence in all TPs
        paths = [os.path.join(tp, fname) for tp in tp_paths]
        if not all(os.path.exists(p) for p in paths):
            print(f"Skipping {code}: missing timepoints")
            continue
            
        # Load EEMs
        eems = {
            '0h': load_eem_matrix(paths[0]),
            '6h': load_eem_matrix(paths[1]),
            '24h': load_eem_matrix(paths[2])
        }
        
        # Extract Features
        config = {
            'normalization': args.normalization,
            'drop_absolute': args.drop_absolute
        }
        feats = extract_features_for_sample(eems, chirality_df, config)
        
        X_rows.append(feats)
        y.append(1 if code_map[code] == 'ALS' else 0)
        codes.append(code)
        
    # 3. Create DataFrame
    X_df = pd.DataFrame(X_rows)
    print(f"Extracted {X_df.shape[1]} features for {len(X_df)} samples.")
    
    # Clean NaNs (e.g. div by zero)
    X_df.fillna(0, inplace=True)
    # Remove constant columns
    X_df = X_df.loc[:, (X_df != X_df.iloc[0]).any()] 
    
    # Save Raw Features
    X_df['Subject'] = codes
    X_df['Label'] = y
    X_df.to_csv(out_path / 'extracted_features_no_heights.csv', index=False)
    
    # 4. LOOCV Evaluation with Feature Selection
    print("\nStarting Nested LOOCV Evaluation...")
    
    X = X_df.drop(columns=['Subject', 'Label']).values
    y = np.array(y)
    feature_names = X_df.drop(columns=['Subject', 'Label']).columns
    
    loo = LeaveOneOut()
    y_true, y_probs = [], []
    feature_counts = np.zeros(len(feature_names))
    feature_importances = np.zeros(len(feature_names))
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 4a. Feature Selection (Inside Loop!)
        # Filter constant features first
        selector = SelectKBest(f_classif, k=min(20, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        
        # Track selected features
        selected_indices = selector.get_support(indices=True)
        feature_counts[selected_indices] += 1
        
        # 4b. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # 4c. Classification
        clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
        clf.fit(X_train_scaled, y_train)
        
        # Track Importance
        coefs = np.abs(clf.coef_[0])
        feature_importances[selected_indices] += coefs
        
        # Predict
        prob = clf.predict_proba(X_test_scaled)[:, 1]
        y_true.append(y_test[0])
        y_probs.append(prob[0])
        
    # 5. Results
    auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_probs])
    
    print(f"\nFinal Results (No Heights):")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    # Stability Report
    stability = pd.DataFrame({
        'Feature': feature_names,
        'Selection_Freq': feature_counts / len(y),
        'Total_Importance': feature_importances
    })
    stability.sort_values('Total_Importance', ascending=False, inplace=True)
    
    print(f"\nTop {args.stability_top_n} Most Important Features:")
    print(stability.head(args.stability_top_n)[['Feature', 'Selection_Freq', 'Total_Importance']])
    
    stability.to_csv(out_path / 'feature_stability_no_heights.csv', index=False)

if __name__ == '__main__':
    main()