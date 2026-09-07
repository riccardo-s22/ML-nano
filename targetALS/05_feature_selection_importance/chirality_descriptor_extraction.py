#!/usr/bin/env python3
"""
chirality_descriptor_extraction.py
====================================

Extract 20 patch-level descriptors from 7×7 regions centered on 12 DNA
chirality positions across 3 timepoints (0h, 6h, 24h).

Produces: chirality_full_descriptors.csv
          (39 samples × 721 columns: 1 code + 12 positions × 20 descriptors × 3 timepoints)

Descriptors extracted per patch:
  Intensity:   mean, max, center, median, integral
  Variability: std, range, cv, iqr
  Morphology:  prominence, sharpness, snr, peak2ring, contrast
  Shape:       kurtosis, skewness
  Gradient:    grad_center, grad_mean
  Width/BG:    fwhm_frac, bg_integral

Usage:
  python chirality_descriptor_extraction.py \
      --tp_dirs out_0h,out_6h,out_24h \
      --labels_csv sample_labels.csv \
      --output chirality_full_descriptors.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from typing import Dict, List, Tuple

# ============================================================================
# DNA chirality positions from Coordinates_DNA.txt
# Format: [helix_turn, base_pair] -> (Emission_nm, Excitation_nm)
# ============================================================================

CHIRALITY_POSITIONS = {
    "ch8_3":  {"emission_nm": 973.98,  "excitation_nm": 673.94},
    "ch6_5":  {"emission_nm": 987.82,  "excitation_nm": 577.12},
    "ch7_5":  {"emission_nm": 1047.81, "excitation_nm": 653.32},
    "ch10_2": {"emission_nm": 1080.60, "excitation_nm": 745.92},
    "ch9_4":  {"emission_nm": 1131.96, "excitation_nm": 731.39},
    "ch8_4":  {"emission_nm": 1130.34, "excitation_nm": 599.78},
    "ch7_6":  {"emission_nm": 1138.19, "excitation_nm": 659.79},
    "ch8_6":  {"emission_nm": 1200.03, "excitation_nm": 727.40},
    "ch8_7":  {"emission_nm": 1288.27, "excitation_nm": 740.87},
    "ch9_5":  {"emission_nm": 1262.98, "excitation_nm": 685.15},
    "ch10_3": {"emission_nm": 1267.70, "excitation_nm": 648.97},
    "ch10_5": {"emission_nm": 1282.97, "excitation_nm": 801.23},
}

PATCH_RADIUS = 3  # 7×7 patch = radius 3 around center

DESCRIPTOR_NAMES = [
    "mean", "max", "center", "median", "integral",
    "std", "range", "cv", "iqr",
    "prominence", "sharpness", "snr", "peak2ring", "contrast",
    "kurtosis", "skewness",
    "grad_center", "grad_mean",
    "fwhm_frac", "bg_integral",
]


# ============================================================================
# EEM file loading
# ============================================================================

def load_eem_matrix(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an EEM Excel file and return (data_matrix, emission_axis, excitation_axis).

    - data_matrix: shape (n_emission, n_excitation), min-max normalized to [0,1]
    - emission_axis: 1-D array of emission wavelengths (nm)
    - excitation_axis: 1-D array of excitation wavelengths (nm)
    """
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    # Parse excitation axis from header row
    header = rows[0]
    excitation = []
    for h in header[1:]:
        if h is None:
            continue
        m = re.search(r"(\d+\.?\d*)", str(h))
        if m:
            excitation.append(float(m.group(1)))
    excitation = np.array(excitation)

    # Parse emission axis and data matrix
    emission = []
    data = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        em_val = float(row[0])
        emission.append(em_val)
        row_data = []
        for cell in row[1:]:
            if isinstance(cell, (int, float)) and cell is not None:
                row_data.append(float(cell))
            else:
                row_data.append(0.0)
        # Ensure same length as excitation
        row_data = row_data[: len(excitation)]
        data.append(row_data)

    emission = np.array(emission)
    mat = np.array(data, dtype=np.float64)

    # Min-max normalize per file
    mn, mx = mat.min(), mat.max()
    if mx > mn:
        mat = (mat - mn) / (mx - mn)
    else:
        mat = np.zeros_like(mat)

    return mat, emission, excitation


def nm_to_index(axis: np.ndarray, target_nm: float) -> int:
    """Find the nearest index in axis for a given wavelength in nm."""
    return int(np.argmin(np.abs(axis - target_nm)))


# ============================================================================
# Descriptor computation on a single 7×7 patch
# ============================================================================

def extract_patch(mat: np.ndarray, row_c: int, col_c: int,
                  radius: int = PATCH_RADIUS) -> Tuple[np.ndarray, int, int]:
    """
    Extract a (2*radius+1) × (2*radius+1) patch centered at (row_c, col_c).
    Clips to matrix boundaries. Returns (patch, local_row_center, local_col_center).
    """
    H, W = mat.shape
    r0 = max(0, row_c - radius)
    r1 = min(H, row_c + radius + 1)
    c0 = max(0, col_c - radius)
    c1 = min(W, col_c + radius + 1)
    patch = mat[r0:r1, c0:c1].copy()
    local_r = row_c - r0
    local_c = col_c - c0
    return patch, local_r, local_c


def ring_pixels(patch: np.ndarray, local_r: int, local_c: int) -> np.ndarray:
    """Return pixels on the outermost ring of the patch (border pixels)."""
    h, w = patch.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    # Exclude the center if it happens to be on the border (shouldn't for 7×7)
    mask[local_r, local_c] = False
    return patch[mask]


def compute_descriptors(patch: np.ndarray, local_r: int, local_c: int,
                        full_mat: np.ndarray, row_c: int, col_c: int) -> Dict[str, float]:
    """
    Compute all 20 descriptors for a single patch.

    Parameters
    ----------
    patch : 2D array, the extracted 7×7 patch
    local_r, local_c : center coordinates within the patch
    full_mat : full EEM matrix (for gradient computation at boundaries)
    row_c, col_c : center coordinates in the full matrix
    """
    flat = patch.ravel()
    center_val = patch[local_r, local_c]
    ring = ring_pixels(patch, local_r, local_c)
    ring_mean = float(ring.mean()) if ring.size > 0 else 0.0
    ring_std = float(ring.std()) if ring.size > 0 else 1e-10

    # --- Intensity ---
    d_mean = float(flat.mean())
    d_max = float(flat.max())
    d_center = float(center_val)
    d_median = float(np.median(flat))
    d_integral = float(flat.sum())

    # --- Variability ---
    d_std = float(flat.std())
    d_range = float(flat.max() - flat.min())
    d_cv = float(d_std / d_mean) if d_mean > 1e-12 else 0.0
    q25, q75 = np.percentile(flat, [25, 75])
    d_iqr = float(q75 - q25)

    # --- Peak Morphology ---
    d_prominence = float(center_val - ring_mean)
    d_sharpness = float(center_val / d_mean) if d_mean > 1e-12 else 1.0
    d_snr = float((center_val - ring_mean) / ring_std) if ring_std > 1e-12 else 0.0
    d_peak2ring = float(center_val / ring_mean) if ring_mean > 1e-12 else 1.0
    denom_contrast = float(center_val + ring_mean)
    d_contrast = float((center_val - ring_mean) / denom_contrast) if denom_contrast > 1e-12 else 0.0

    # --- Distribution Shape ---
    d_kurtosis = float(sp_kurtosis(flat, fisher=True, bias=True))
    d_skewness = float(sp_skew(flat, bias=True))

    # --- Gradient ---
    H, W = full_mat.shape
    # Gradient at center pixel (central differences, clipped at boundary)
    dy = 0.0
    if row_c > 0 and row_c < H - 1:
        dy = (full_mat[row_c + 1, col_c] - full_mat[row_c - 1, col_c]) / 2.0
    elif row_c == 0:
        dy = full_mat[row_c + 1, col_c] - full_mat[row_c, col_c]
    else:
        dy = full_mat[row_c, col_c] - full_mat[row_c - 1, col_c]

    dx = 0.0
    if col_c > 0 and col_c < W - 1:
        dx = (full_mat[row_c, col_c + 1] - full_mat[row_c, col_c - 1]) / 2.0
    elif col_c == 0:
        dx = full_mat[row_c, col_c + 1] - full_mat[row_c, col_c]
    else:
        dx = full_mat[row_c, col_c] - full_mat[row_c, col_c - 1]

    d_grad_center = float(np.sqrt(dy ** 2 + dx ** 2))

    # Mean gradient over patch using np.gradient
    gy, gx = np.gradient(patch)
    grad_mag = np.sqrt(gy ** 2 + gx ** 2)
    d_grad_mean = float(grad_mag.mean())

    # --- FWHM fraction ---
    # Threshold at half-prominence: midpoint between center value and ring baseline
    # This measures how wide the peak is at half its height above background
    half_prom_level = (center_val + ring_mean) / 2.0
    d_fwhm_frac = float(np.sum(flat > half_prom_level) / flat.size) if flat.size > 0 else 0.0

    # --- Background-subtracted integral ---
    above = flat - ring_mean
    d_bg_integral = float(above[above > 0].sum())

    return {
        "mean": d_mean, "max": d_max, "center": d_center, "median": d_median,
        "integral": d_integral,
        "std": d_std, "range": d_range, "cv": d_cv, "iqr": d_iqr,
        "prominence": d_prominence, "sharpness": d_sharpness, "snr": d_snr,
        "peak2ring": d_peak2ring, "contrast": d_contrast,
        "kurtosis": d_kurtosis, "skewness": d_skewness,
        "grad_center": d_grad_center, "grad_mean": d_grad_mean,
        "fwhm_frac": d_fwhm_frac, "bg_integral": d_bg_integral,
    }


# ============================================================================
# Main extraction pipeline
# ============================================================================

def normalize_code(code: str) -> str:
    """Normalize sample code: replace dots with underscores to match filenames."""
    return code.replace(".", "_")


def find_file(tp_dir: str, code: str) -> str:
    """Find the Excel file for a sample code in a timepoint directory."""
    # Normalize dots to underscores (labels CSV uses dots, filenames use underscores)
    code_norm = normalize_code(code)

    # tp_dir name ends with _0h, _6h, _24h
    tp_suffix = os.path.basename(tp_dir)  # e.g. "out_0h"
    tp_tag = tp_suffix.split("_")[-1]      # e.g. "0h"

    direct = os.path.join(tp_dir, f"{code_norm}_{tp_tag}.xlsx")
    if os.path.exists(direct):
        return direct

    # Also try original code (in case it already uses underscores)
    direct2 = os.path.join(tp_dir, f"{code}_{tp_tag}.xlsx")
    if os.path.exists(direct2):
        return direct2

    # Search for files starting with code
    for f in os.listdir(tp_dir):
        if f.lower().endswith(".xlsx") and not f.startswith("~$"):
            basename = os.path.splitext(f)[0]
            # Strip the timepoint suffix to get the code
            for suffix in ["_0h", "_6h", "_24h"]:
                if basename.endswith(suffix):
                    basename = basename[: -len(suffix)]
                    break
            if basename == code_norm or basename == code:
                return os.path.join(tp_dir, f)

    raise FileNotFoundError(f"No file found for code={code} (normalized: {code_norm}) in {tp_dir}")


def extract_all_descriptors(
    codes: List[str],
    tp_dirs: List[str],
    tp_labels: List[str],
) -> pd.DataFrame:
    """
    Extract descriptors for all samples × chirality positions × timepoints.
    Returns a DataFrame with columns: code, {chir}_{desc}_{tp}, ...
    """
    # Pre-compute pixel indices for chirality positions using first file
    first_code = codes[0]
    first_file = find_file(tp_dirs[0], first_code)
    _, emission_axis, excitation_axis = load_eem_matrix(first_file)

    chir_pixel_coords = {}
    for chir_name, coords in CHIRALITY_POSITIONS.items():
        row_idx = nm_to_index(emission_axis, coords["emission_nm"])
        col_idx = nm_to_index(excitation_axis, coords["excitation_nm"])
        chir_pixel_coords[chir_name] = (row_idx, col_idx)
        print(f"  {chir_name}: em={coords['emission_nm']:.1f}nm -> row={row_idx}, "
              f"ex={coords['excitation_nm']:.1f}nm -> col={col_idx}")

    # Build column order: for each chirality, for each descriptor, for each timepoint
    # Matches original output: ch8_3_mean_0h, ch8_3_max_0h, ..., ch6_5_mean_0h, ...
    chir_order = list(CHIRALITY_POSITIONS.keys())
    columns = ["code"]
    for chir in chir_order:
        for tp in tp_labels:
            for desc in DESCRIPTOR_NAMES:
                columns.append(f"{chir}_{desc}_{tp}")

    all_rows = []

    for i, code in enumerate(codes):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing sample {i+1}/{len(codes)}: {code}")

        row_dict = {"code": normalize_code(code)}

        for tp_dir, tp_label in zip(tp_dirs, tp_labels):
            filepath = find_file(tp_dir, code)
            mat, _, _ = load_eem_matrix(filepath)

            for chir_name in chir_order:
                row_c, col_c = chir_pixel_coords[chir_name]
                patch, lr, lc = extract_patch(mat, row_c, col_c, PATCH_RADIUS)
                descs = compute_descriptors(patch, lr, lc, mat, row_c, col_c)

                for desc_name, val in descs.items():
                    col_name = f"{chir_name}_{desc_name}_{tp_label}"
                    row_dict[col_name] = val

        all_rows.append(row_dict)

    df = pd.DataFrame(all_rows, columns=columns)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract chirality patch descriptors from EEM spectral data"
    )
    parser.add_argument(
        "--tp_dirs", type=str, required=True,
        help="Comma-separated paths to 0h,6h,24h directories"
    )
    parser.add_argument(
        "--labels_csv", type=str, required=True,
        help="CSV with columns: code, group"
    )
    parser.add_argument(
        "--output", type=str, default="chirality_full_descriptors.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    tp_dirs = [s.strip() for s in args.tp_dirs.split(",")]
    assert len(tp_dirs) == 3, "Provide exactly 3 timepoint directories (0h, 6h, 24h)"
    tp_labels = ["0h", "6h", "24h"]

    # Load sample codes from labels CSV
    labels_df = pd.read_csv(args.labels_csv)
    codes = labels_df["code"].astype(str).tolist()
    print(f"Loaded {len(codes)} samples from {args.labels_csv}")
    print(f"Groups: {labels_df['group'].value_counts().to_dict()}")

    print(f"\nMapping chirality positions to pixel coordinates:")
    df = extract_all_descriptors(codes, tp_dirs, tp_labels)

    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: code + {len(CHIRALITY_POSITIONS)} chiralities × "
          f"{len(DESCRIPTOR_NAMES)} descriptors × {len(tp_labels)} timepoints "
          f"= {len(CHIRALITY_POSITIONS) * len(DESCRIPTOR_NAMES) * len(tp_labels)} features")


if __name__ == "__main__":
    main()
