#!/usr/bin/env python3
"""
STEP 5 (CORRECTED): Visualize Top 5 ROI Masks on Mean EEM Spectrum
==================================================================

This script:
1. Computes the Grand Average EEM (Mean of all samples) to use as a background.
2. Loads the ROI masks for the Top 5 validated features.
3. Generates plots overlaying the masks on the spectrum.

USAGE:
------
python step5_visualize_top5_rois.py \
    --tp_dirs ./out_0h,./out_6h,./out_24h \
    --step3_dir ./step3_rois \
    --output_dir ./step5_visualization
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from openpyxl import load_workbook
# --- FIXED: Added Dict and Tuple to imports ---
from typing import List, Dict, Tuple, Optional 

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG: The Top 5 Features (Must match Step 4)
# =============================================================================
TARGET_FEATURES = [
    "best_model_fold3_dim477",
    "best_model_fold1_dim103",
    "best_model_fold3_dim376",
    "best_model_fold1_dim179",
    "best_model_fold3_dim370"
]

TP_NAMES = ["0h", "6h", "24h"]

# =============================================================================
# Helpers
# =============================================================================

def list_excel_files(tp_dir: Path) -> List[Path]:
    return [f for f in tp_dir.glob("*.xlsx") if not f.name.startswith("~$")]

def load_excel_as_array(filepath: Path) -> np.ndarray:
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    data = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0: continue
        row_data = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0: continue
            if isinstance(cell, (int, float)) and cell is not None:
                row_data.append(float(cell))
            else:
                row_data.append(0.0)
        if row_data: data.append(row_data)
    
    # Normalize to 0-1 for visualization consistency
    arr = np.array(data, dtype=np.float32)
    if arr.size == 0: return np.zeros((10,10), dtype=np.float32)
    
    mn, mx = arr.min(), arr.max()
    if mx > mn: arr = (arr - mn) / (mx - mn)
    return arr

def compute_grand_mean(tp_dirs: List[str]) -> Dict[str, np.ndarray]:
    """Computes the mean EEM image for each timepoint across all samples."""
    print("Computing Grand Mean EEM (Background)...")
    means = {}
    
    for tp_idx, tp_name in enumerate(TP_NAMES):
        tp_dir = Path(tp_dirs[tp_idx])
        files = list_excel_files(tp_dir)
        if not files:
            raise FileNotFoundError(f"No files found in {tp_dir}")
            
        accum = None
        count = 0
        
        for f in files:
            arr = load_excel_as_array(f)
            if accum is None:
                accum = np.zeros_like(arr, dtype=np.float64)
            
            # Handle slight shape mismatches by cropping to min
            h_min = min(accum.shape[0], arr.shape[0])
            w_min = min(accum.shape[1], arr.shape[1])
            accum = accum[:h_min, :w_min]
            # If arr is larger, crop it
            arr_crop = arr[:h_min, :w_min]
            
            accum += arr_crop
            count += 1
            
        if count > 0:
            means[tp_name] = (accum / count).astype(np.float32)
            print(f"  {tp_name}: Averaged {count} samples. Shape: {means[tp_name].shape}")
        else:
            raise RuntimeError(f"No valid EEMs found for {tp_name}")
        
    return means

def parse_feature_name(fname: str) -> Tuple[str, int]:
    """Extracts 'best_model_fold3' and 477 from 'best_model_fold3_dim477'"""
    idx = fname.rfind("_dim")
    mname = fname[:idx]
    dim = int(fname[idx+4:])
    return mname, dim

def load_roi_mask(step3_dir: Path, model_name: str, dim: int, tp: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """Loads ROI mask from .npy or returns empty if missing."""
    # Try direct file
    fpath = step3_dir / model_name / f"roi_mask_dim{dim}_{tp}.npy"
    if fpath.exists():
        mask = np.load(fpath).astype(bool)
        
        # Crop/Pad if shape mismatch with mean image
        if mask.shape != target_shape:
            h, w = target_shape
            mh, mw = mask.shape
            new_mask = np.zeros((h, w), dtype=bool)
            h_min, w_min = min(h, mh), min(w, mw)
            new_mask[:h_min, :w_min] = mask[:h_min, :w_min]
            return new_mask
        return mask
    
    # Return empty mask if missing
    return np.zeros(target_shape, dtype=bool)

# =============================================================================
# Main Visualization Logic
# =============================================================================

def plot_feature_overlay(feature_name: str, 
                         means: Dict[str, np.ndarray], 
                         masks: Dict[str, np.ndarray], 
                         out_path: Path):
    """Generates a 3-panel plot (0h, 6h, 24h) for one feature."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Feature: {feature_name}", fontsize=16)
    
    for i, tp in enumerate(TP_NAMES):
        ax = axes[i]
        bg = means[tp]
        mask = masks[tp]
        
        # Plot Background (Gray EEM)
        ax.imshow(bg, cmap='gray', aspect='auto', origin='upper')
        
        # Overlay ROI (Red/Yellow alpha blend)
        if mask.any():
            # Create an RGBA overlay
            overlay = np.zeros((*bg.shape, 4))
            # Red overlay (1, 0, 0, alpha=0.5)
            overlay[mask] = [1, 0, 0, 0.4] 
            ax.imshow(overlay, origin='upper', aspect='auto')
            
            # Contour
            # Contours require float/int input, not bool
            ax.contour(mask.astype(float), colors='yellow', linewidths=1.0, alpha=0.8, origin='upper')

        n_pixels = mask.sum()
        pct = 100 * n_pixels / mask.size
        ax.set_title(f"{tp}\n(ROI: {n_pixels} px, {pct:.1f}%)")
        ax.set_xlabel("Emission Wavelength (bins)")
        if i == 0: ax.set_ylabel("Excitation Wavelength (bins)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_dirs", required=True, help="Comma-separated paths to out_0h, out_6h, out_24h")
    parser.add_argument("--step3_dir", required=True)
    parser.add_argument("--output_dir", default="./step5_visualization")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    tp_dirs_list = [d.strip() for d in args.tp_dirs.split(",")]
    step3_dir = Path(args.step3_dir)
    
    # 1. Compute Mean EEMs
    means = compute_grand_mean(tp_dirs_list)
    
    # Save mean EEMs for reference
    for tp, arr in means.items():
        plt.figure(figsize=(6,5))
        plt.imshow(arr, cmap='viridis', aspect='auto', origin='upper')
        plt.title(f"Grand Mean EEM - {tp}")
        plt.colorbar()
        plt.savefig(outdir / f"grand_mean_{tp}.png")
        plt.close()
    
    print(f"\nVisualizing Top {len(TARGET_FEATURES)} Features...")
    
    # 2. Iterate Features and Plot
    for feat in TARGET_FEATURES:
        mname, dim = parse_feature_name(feat)
        print(f"  Processing {feat} (Model: {mname}, Dim: {dim})...")
        
        # Load masks for all 3 TPs
        masks = {}
        for tp in TP_NAMES:
            shape = means[tp].shape
            masks[tp] = load_roi_mask(step3_dir, mname, dim, tp, shape)
        
        # Plot
        plot_feature_overlay(feat, means, masks, outdir / f"viz_{feat}.png")
        
    print(f"\nDone. Visualizations saved to {outdir}")

if __name__ == "__main__":
    main()