#!/usr/bin/env python3
"""
chirality_interpolated_extraction.py
=====================================

Precision extraction of physical descriptors from EEM spectral data using:
  1) Bicubic interpolation of the full EEM matrix → continuous intensity surface
  2) Local-maximum search within 2 nm radius of each chirality coordinate
  3) Gaussian fitting along the emission axis through the local max,
     using only points with a consistently degrading gradient from peak
  4) Six Gaussian-curve descriptors per chirality per timepoint
  5) Pairwise intensity ratios and EEM-plane distances between all peaks

Output columns per chirality per timepoint (72 chiralities × 6 descriptors × 3 tp = 1296):
  {chir}_gauss_max_{tp}       – fitted Gaussian amplitude (peak intensity)
  {chir}_gauss_em_center_{tp} – emission center of fitted Gaussian (nm)
  {chir}_gauss_ex_center_{tp} – excitation at local max (nm)  
  {chir}_gauss_fwhm_{tp}      – full width at half maximum (nm)
  {chir}_gauss_auc_{tp}       – area under Gaussian curve
  {chir}_gauss_skew_{tp}      – skewness of the data points used for fitting
  {chir}_gauss_kurt_{tp}      – kurtosis of the data points used for fitting

Pairwise features (66 pairs × 2 × 3 tp = 396):
  {chirA}_vs_{chirB}_ratio_{tp}    – intensity ratio max_A / max_B
  {chirA}_vs_{chirB}_eemdist_{tp}  – Euclidean distance in (emission, excitation) nm

Total features: code + per-chirality descriptors + pairwise features

Usage:
  python chirality_interpolated_extraction.py \
      --data_dir /path/to/project \
      --labels_csv sample_labels.csv \
      --output chirality_interp_descriptors.csv
"""

import os
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from itertools import combinations
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# DNA chirality positions (from Coordinates_DNA.txt)
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

SEARCH_RADIUS_NM = 2.0       # nm radius to search for local maximum
INTERP_FACTOR = 10            # upsampling factor for interpolation
GRADIENT_REVERSAL_LIMIT = 2   # max consecutive gradient reversals before stop

TIMEPOINT_LABELS = ["0h", "6h", "24h"]


# ============================================================================
# Gaussian model
# ============================================================================

def gaussian(x, amplitude, center, sigma):
    """1D Gaussian function."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def gaussian_with_baseline(x, amplitude, center, sigma, baseline):
    """1D Gaussian with a constant baseline offset."""
    return baseline + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


# ============================================================================
# EEM loading (raw, no normalization – physical intensities matter)
# ============================================================================

def load_eem_raw(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EEM Excel file → (data_matrix, emission_axis, excitation_axis).
    No normalization applied – we work with raw intensities.
    """
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    header = rows[0]
    excitation = []
    for h in header[1:]:
        if h is None:
            continue
        m = re.search(r"(\d+\.?\d*)", str(h))
        if m:
            excitation.append(float(m.group(1)))
    excitation = np.array(excitation, dtype=np.float64)

    emission = []
    data = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        emission.append(float(row[0]))
        row_data = []
        for cell in row[1:]:
            if isinstance(cell, (int, float)) and cell is not None:
                row_data.append(float(cell))
            else:
                row_data.append(0.0)
        row_data = row_data[:len(excitation)]
        data.append(row_data)

    emission = np.array(emission, dtype=np.float64)
    mat = np.array(data, dtype=np.float64)

    return mat, emission, excitation


# ============================================================================
# Interpolation: build a continuous EEM surface
# ============================================================================

def build_interpolated_surface(
    mat: np.ndarray,
    emission: np.ndarray,
    excitation: np.ndarray,
    factor: int = INTERP_FACTOR,
) -> Tuple[RectBivariateSpline, np.ndarray, np.ndarray]:
    """
    Build a bicubic spline interpolator over the EEM matrix.
    Also returns the fine-grained emission and excitation axes.

    Parameters
    ----------
    mat : (n_em, n_ex) raw intensity matrix
    emission : (n_em,) emission wavelengths
    excitation : (n_ex,) excitation wavelengths
    factor : upsampling factor

    Returns
    -------
    spline : RectBivariateSpline callable(em, ex)
    em_fine : upsampled emission axis
    ex_fine : upsampled excitation axis
    """
    # RectBivariateSpline requires strictly increasing axes
    # emission is the row axis (y), excitation is column axis (x)
    spline = RectBivariateSpline(emission, excitation, mat, kx=3, ky=3)

    em_fine = np.linspace(emission[0], emission[-1],
                          len(emission) * factor)
    ex_fine = np.linspace(excitation[0], excitation[-1],
                          len(excitation) * factor)

    return spline, em_fine, ex_fine


# ============================================================================
# Local max search in 2 nm radius on the interpolated surface
# ============================================================================

def find_local_max_in_radius(
    spline: RectBivariateSpline,
    em_fine: np.ndarray,
    ex_fine: np.ndarray,
    target_em: float,
    target_ex: float,
    radius_nm: float = SEARCH_RADIUS_NM,
) -> Tuple[float, float, float]:
    """
    Search for the local maximum within `radius_nm` of (target_em, target_ex)
    on the interpolated surface.

    Returns (best_em, best_ex, max_intensity).
    """
    # Mask: only consider fine-grid points within radius
    em_mask = np.abs(em_fine - target_em) <= radius_nm
    ex_mask = np.abs(ex_fine - target_ex) <= radius_nm

    em_sub = em_fine[em_mask]
    ex_sub = ex_fine[ex_mask]

    if len(em_sub) == 0 or len(ex_sub) == 0:
        # Fallback: nearest point
        val = float(spline(target_em, target_ex)[0, 0])
        return target_em, target_ex, val

    # Evaluate spline on the sub-grid
    Z = spline(em_sub, ex_sub)  # shape (len(em_sub), len(ex_sub))

    # Find argmax
    idx = np.unravel_index(np.argmax(Z), Z.shape)
    best_em = float(em_sub[idx[0]])
    best_ex = float(ex_sub[idx[1]])
    max_val = float(Z[idx])

    return best_em, best_ex, max_val


# ============================================================================
# Extract 1D emission profile at fixed excitation and select consistent
# gradient-decay points from the peak
# ============================================================================

def extract_consistent_gradient_points(
    spline: RectBivariateSpline,
    em_fine: np.ndarray,
    peak_em: float,
    fixed_ex: float,
    reversal_limit: int = GRADIENT_REVERSAL_LIMIT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the 1D emission profile at fixed excitation, then select points
    around the peak that have a consistently degrading gradient.

    Starting from the peak, walk left and right along emission. At each step,
    the intensity must decrease (gradient < 0 going away from peak). Stop when
    the gradient reverses for more than `reversal_limit` consecutive points.

    Returns (em_points, intensity_points) – the selected subset.
    """
    # Full 1D profile along emission at fixed excitation
    profile = spline(em_fine, np.array([fixed_ex]))[:, 0]

    # Find the index closest to peak_em
    peak_idx = int(np.argmin(np.abs(em_fine - peak_em)))

    # Walk RIGHT from peak
    right_indices = [peak_idx]
    consecutive_reversals = 0
    for i in range(peak_idx + 1, len(profile)):
        if profile[i] <= profile[right_indices[-1]]:
            right_indices.append(i)
            consecutive_reversals = 0
        else:
            consecutive_reversals += 1
            if consecutive_reversals >= reversal_limit:
                break
            # Still include this point (minor bump), but count it
            right_indices.append(i)

    # Walk LEFT from peak
    left_indices = []
    consecutive_reversals = 0
    prev_val = profile[peak_idx]
    for i in range(peak_idx - 1, -1, -1):
        if profile[i] <= prev_val:
            left_indices.append(i)
            prev_val = profile[i]
            consecutive_reversals = 0
        else:
            consecutive_reversals += 1
            if consecutive_reversals >= reversal_limit:
                break
            left_indices.append(i)
            prev_val = profile[i]

    # Combine: left (reversed) + right
    left_indices = left_indices[::-1]
    all_indices = left_indices + right_indices
    all_indices = sorted(set(all_indices))

    # Remove any trailing points that were part of reversal sequences
    # by trimming indices where gradient inverted
    # Clean pass: from peak outward, only keep strictly monotone-decay
    clean_right = [peak_idx]
    for i in range(peak_idx + 1, len(profile)):
        if i not in all_indices:
            break
        if profile[i] <= profile[clean_right[-1]]:
            clean_right.append(i)
        else:
            # Start counting reversals from here
            rev_count = 0
            j = i
            while j < len(profile) and j in all_indices:
                if profile[j] > profile[j - 1] if j > peak_idx else profile[j] > profile[j + 1]:
                    rev_count += 1
                else:
                    rev_count = 0
                if rev_count >= reversal_limit:
                    break
                j += 1
            # Include up to but not including the reversal block
            break

    clean_left = []
    prev_val = profile[peak_idx]
    for i in range(peak_idx - 1, -1, -1):
        if i not in all_indices:
            break
        if profile[i] <= prev_val:
            clean_left.append(i)
            prev_val = profile[i]
        else:
            break

    clean_left = clean_left[::-1]
    final_indices = np.array(clean_left + clean_right, dtype=int)

    # Need at least 4 points for a meaningful Gaussian fit
    if len(final_indices) < 4:
        # Fallback: use a wider window around peak
        half_win = 15  # points on the fine grid
        lo = max(0, peak_idx - half_win)
        hi = min(len(em_fine), peak_idx + half_win + 1)
        final_indices = np.arange(lo, hi)

    em_pts = em_fine[final_indices]
    int_pts = profile[final_indices]

    return em_pts, int_pts


# ============================================================================
# Gaussian fitting and descriptor extraction
# ============================================================================

def fit_gaussian_and_extract(
    em_pts: np.ndarray,
    int_pts: np.ndarray,
    peak_em_init: float,
    peak_intensity_init: float,
) -> Dict[str, float]:
    """
    Fit a Gaussian (with baseline) to the selected emission points and extract:
      - gauss_max: fitted amplitude (above baseline)
      - gauss_em_center: fitted emission center (nm)
      - gauss_fwhm: full width at half maximum (nm)
      - gauss_auc: area under the Gaussian (analytical: amplitude * sigma * sqrt(2π))
      - gauss_skew: skewness of the raw data points
      - gauss_kurt: kurtosis of the raw data points

    Falls back to empirical estimates if fitting fails.
    """
    result = {
        "gauss_max": peak_intensity_init,
        "gauss_em_center": peak_em_init,
        "gauss_fwhm": np.nan,
        "gauss_auc": np.nan,
        "gauss_skew": 0.0,
        "gauss_kurt": 0.0,
    }

    if len(em_pts) < 4:
        return result

    # Data-point distribution shape (weighted by intensity)
    # Treat the profile as a probability distribution
    weights = int_pts - int_pts.min()
    w_sum = weights.sum()
    if w_sum > 1e-15:
        w_norm = weights / w_sum
        weighted_mean = np.sum(em_pts * w_norm)
        weighted_var = np.sum(w_norm * (em_pts - weighted_mean) ** 2)
        weighted_std = np.sqrt(weighted_var) if weighted_var > 0 else 1.0
        if weighted_std > 1e-12:
            standardized = (em_pts - weighted_mean) / weighted_std
            result["gauss_skew"] = float(np.sum(w_norm * standardized ** 3))
            result["gauss_kurt"] = float(np.sum(w_norm * standardized ** 4) - 3.0)
        else:
            result["gauss_skew"] = 0.0
            result["gauss_kurt"] = 0.0
    else:
        result["gauss_skew"] = float(sp_skew(int_pts, bias=True))
        result["gauss_kurt"] = float(sp_kurtosis(int_pts, fisher=True, bias=True))

    # Initial guesses for Gaussian fit
    baseline_guess = float(np.min(int_pts))
    amp_guess = float(peak_intensity_init - baseline_guess)
    sigma_guess = float(np.std(em_pts)) if np.std(em_pts) > 0 else 5.0

    try:
        popt, _ = curve_fit(
            gaussian_with_baseline,
            em_pts,
            int_pts,
            p0=[amp_guess, peak_em_init, sigma_guess, baseline_guess],
            bounds=(
                [0, em_pts.min() - 10, 0.1, -np.inf],
                [np.inf, em_pts.max() + 10, 200.0, np.inf],
            ),
            maxfev=5000,
        )
        amp_fit, center_fit, sigma_fit, baseline_fit = popt
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_fit  # ≈ 2.355 * sigma
        auc = amp_fit * sigma_fit * np.sqrt(2.0 * np.pi)       # analytical integral

        result["gauss_max"] = float(amp_fit + baseline_fit)  # total peak height
        result["gauss_em_center"] = float(center_fit)
        result["gauss_fwhm"] = float(fwhm)
        result["gauss_auc"] = float(auc)

    except (RuntimeError, ValueError, TypeError):
        # Fallback: empirical estimates
        result["gauss_max"] = float(peak_intensity_init)
        result["gauss_em_center"] = float(peak_em_init)

        # Empirical FWHM: find where intensity drops to half of (peak - baseline)
        baseline_est = float(np.min(int_pts))
        half_max = (peak_intensity_init + baseline_est) / 2.0
        above_half = em_pts[int_pts >= half_max]
        if len(above_half) >= 2:
            result["gauss_fwhm"] = float(above_half[-1] - above_half[0])
        else:
            result["gauss_fwhm"] = float(np.nan)

        # Empirical AUC via trapezoidal integration (baseline-subtracted)
        baseline_sub = int_pts - baseline_est
        baseline_sub[baseline_sub < 0] = 0
        result["gauss_auc"] = float(np.trapz(baseline_sub, em_pts))

    return result


# ============================================================================
# File finding (same logic as original, adapted for flat directory)
# ============================================================================

def normalize_code(code: str) -> str:
    return code.replace(".", "_")


def _collapse_underscores(s: str) -> str:
    """Collapse runs of underscores so 'a__b' matches 'a_b'."""
    import re
    return re.sub(r"_+", "_", s)


def _search_directory(directory: str, code: str, code_norm: str, tp_tag: str,
                      require_tp_suffix: bool = True):
    """Search a single directory for matching file. Returns path or None.

    If require_tp_suffix is False, also matches files WITHOUT the _0h/_6h/_24h
    suffix (for layouts where timepoint is encoded in the parent directory name).
    """
    if not os.path.isdir(directory):
        return None

    collapsed = _collapse_underscores(code_norm)

    # --- Try direct-name matches first (faster) ---
    # With tp suffix
    for candidate_code in dict.fromkeys([code_norm, collapsed]):  # unique, ordered
        direct = os.path.join(directory, f"{candidate_code}_{tp_tag}.xlsx")
        if os.path.exists(direct):
            return direct
    # Without tp suffix (when directory already encodes the timepoint)
    if not require_tp_suffix:
        for candidate_code in dict.fromkeys([code_norm, collapsed]):
            direct = os.path.join(directory, f"{candidate_code}.xlsx")
            if os.path.exists(direct):
                return direct

    # --- Scan directory with flexible matching ---
    for f in sorted(os.listdir(directory)):
        if not f.lower().endswith(".xlsx") or f.startswith("~$"):
            continue
        basename = os.path.splitext(f)[0]

        # A) Files WITH timepoint suffix
        for suffix in ["_0h", "_6h", "_24h"]:
            if basename.endswith(suffix):
                file_tp = suffix[1:]  # "0h", "6h", "24h"
                file_code = basename[:-len(suffix)]
                if file_tp != tp_tag:
                    continue
                if file_code == code_norm or file_code == code:
                    return os.path.join(directory, f)
                if _collapse_underscores(file_code) == collapsed:
                    return os.path.join(directory, f)
                break  # matched a suffix pattern, no need to check others

        # B) Files WITHOUT timepoint suffix (only when not requiring it)
        if not require_tp_suffix:
            file_code = basename
            if file_code == code_norm or file_code == code:
                return os.path.join(directory, f)
            if _collapse_underscores(file_code) == collapsed:
                return os.path.join(directory, f)

    return None


def find_file_flat(data_dir: str, code: str, tp_tag: str) -> str:
    """
    Find the Excel file for a sample code and timepoint.

    Searches in order:
      0. If data_dir is comma-separated, search each directory directly
      1. data_dir itself (flat layout)
      2. data_dir/out_{tp_tag}  (timepoint subdirectories)
      3. Any immediate subdirectory of data_dir
    """
    code_norm = normalize_code(code)

    # 0. Handle comma-separated list of directories
    if "," in data_dir:
        dirs = [d.strip() for d in data_dir.split(",") if d.strip()]
        # First pass: prefer directories whose name contains the tp_tag
        for d in dirs:
            dirname = os.path.basename(d.rstrip(os.sep))
            if tp_tag in dirname:
                result = _search_directory(d, code, code_norm, tp_tag, require_tp_suffix=False)
                if result:
                    return result
        # Second pass: try all directories (files may have tp suffix)
        for d in dirs:
            result = _search_directory(d, code, code_norm, tp_tag, require_tp_suffix=True)
            if result:
                return result
        raise FileNotFoundError(
            f"No file found for code={code} (norm: {code_norm}), tp={tp_tag} in {data_dir}"
        )

    # 1. Search data_dir directly (files have tp suffix)
    result = _search_directory(data_dir, code, code_norm, tp_tag, require_tp_suffix=True)
    if result:
        return result

    # 2. Search common timepoint subdirectory patterns (files may lack tp suffix)
    tp_subdir_patterns = [
        f"out_{tp_tag}",          # out_0h, out_6h, out_24h
        f"tp_{tp_tag}",           # tp_0h, tp_6h, tp_24h
        tp_tag,                   # 0h, 6h, 24h
    ]
    for pattern in tp_subdir_patterns:
        subdir = os.path.join(data_dir, pattern)
        result = _search_directory(subdir, code, code_norm, tp_tag, require_tp_suffix=False)
        if result:
            return result

    # 3. Fallback: search all immediate subdirectories
    if os.path.isdir(data_dir):
        for entry in sorted(os.listdir(data_dir)):
            subdir = os.path.join(data_dir, entry)
            if os.path.isdir(subdir) and not entry.startswith("."):
                result = _search_directory(subdir, code, code_norm, tp_tag, require_tp_suffix=False)
                if result:
                    return result

    raise FileNotFoundError(
        f"No file found for code={code} (norm: {code_norm}), tp={tp_tag} in {data_dir}"
    )


# ============================================================================
# Main extraction for one sample × one timepoint
# ============================================================================

def extract_sample_tp(
    filepath: str,
    chirality_positions: Dict,
    interp_factor: int = INTERP_FACTOR,
    search_radius: float = SEARCH_RADIUS_NM,
) -> Dict[str, Dict[str, float]]:
    """
    For one EEM file, extract Gaussian descriptors for each chirality.

    Returns dict: {chir_name: {descriptor_name: value, ...}, ...}
    Also adds 'gauss_ex_center' (the excitation coordinate of the local max).
    """
    mat, emission, excitation = load_eem_raw(filepath)

    # Build interpolated surface
    spline, em_fine, ex_fine = build_interpolated_surface(
        mat, emission, excitation, factor=interp_factor
    )

    results = {}

    for chir_name, coords in chirality_positions.items():
        target_em = coords["emission_nm"]
        target_ex = coords["excitation_nm"]

        # 1) Find local max in 2 nm radius on interpolated surface
        best_em, best_ex, max_intensity = find_local_max_in_radius(
            spline, em_fine, ex_fine, target_em, target_ex, search_radius
        )

        # 2) Extract 1D emission profile at the excitation of the local max
        #    and select consistent-gradient points
        em_pts, int_pts = extract_consistent_gradient_points(
            spline, em_fine, best_em, best_ex
        )

        # 3) Gaussian fit and descriptor extraction
        descs = fit_gaussian_and_extract(em_pts, int_pts, best_em, max_intensity)
        descs["gauss_ex_center"] = best_ex

        results[chir_name] = descs

    return results


# ============================================================================
# Pairwise features: intensity ratios and EEM distances
# ============================================================================

def compute_pairwise_features(
    chir_results: Dict[str, Dict[str, float]],
    chir_order: List[str],
) -> Dict[str, float]:
    """
    Compute for every ordered pair (A, B) where A < B alphabetically:
      - ratio = max_A / max_B
      - eemdist = sqrt((em_A - em_B)^2 + (ex_A - ex_B)^2)
    """
    pairwise = {}
    for chir_a, chir_b in combinations(chir_order, 2):
        ra = chir_results[chir_a]
        rb = chir_results[chir_b]

        max_a = ra["gauss_max"]
        max_b = rb["gauss_max"]

        em_a = ra["gauss_em_center"]
        em_b = rb["gauss_em_center"]
        ex_a = ra["gauss_ex_center"]
        ex_b = rb["gauss_ex_center"]

        # Intensity ratio (handle zero)
        if abs(max_b) > 1e-15:
            ratio = max_a / max_b
        else:
            ratio = np.nan

        # Euclidean distance in EEM plane (nm)
        dist = np.sqrt((em_a - em_b) ** 2 + (ex_a - ex_b) ** 2)

        pairwise[f"{chir_a}_vs_{chir_b}_ratio"] = float(ratio)
        pairwise[f"{chir_a}_vs_{chir_b}_eemdist"] = float(dist)

    return pairwise


# ============================================================================
# Full pipeline
# ============================================================================

def run_extraction(
    data_dir: str,
    labels_csv: str,
    output_path: str,
):
    labels_df = pd.read_csv(labels_csv)
    codes = labels_df["code"].astype(str).tolist()
    print(f"Loaded {len(codes)} samples from {labels_csv}")
    print(f"Groups: {labels_df['group'].value_counts().to_dict()}")

    chir_order = list(CHIRALITY_POSITIONS.keys())
    chir_pairs = list(combinations(chir_order, 2))
    per_chir_descs = [
        "gauss_max", "gauss_em_center", "gauss_ex_center",
        "gauss_fwhm", "gauss_auc", "gauss_skew", "gauss_kurt",
    ]

    # Build column order
    columns = ["code"]
    for tp in TIMEPOINT_LABELS:
        for chir in chir_order:
            for desc in per_chir_descs:
                columns.append(f"{chir}_{desc}_{tp}")
    for tp in TIMEPOINT_LABELS:
        for chir_a, chir_b in chir_pairs:
            columns.append(f"{chir_a}_vs_{chir_b}_ratio_{tp}")
            columns.append(f"{chir_a}_vs_{chir_b}_eemdist_{tp}")

    n_per_chir = len(chir_order) * len(per_chir_descs) * len(TIMEPOINT_LABELS)
    n_pairwise = len(chir_pairs) * 2 * len(TIMEPOINT_LABELS)
    print(f"\nFeature breakdown:")
    print(f"  Per-chirality: {len(chir_order)} chiralities × "
          f"{len(per_chir_descs)} descriptors × {len(TIMEPOINT_LABELS)} tp = {n_per_chir}")
    print(f"  Pairwise:      {len(chir_pairs)} pairs × 2 features × "
          f"{len(TIMEPOINT_LABELS)} tp = {n_pairwise}")
    print(f"  Total:         {n_per_chir + n_pairwise} + 1 (code) = {len(columns)}")

    # Validate chirality positions against first file
    first_code = codes[0]
    first_file = find_file_flat(data_dir, first_code, "0h")
    _, em_ax, ex_ax = load_eem_raw(first_file)
    print(f"\nEEM grid: emission [{em_ax[0]:.1f}, {em_ax[-1]:.1f}] nm "
          f"({len(em_ax)} pts, step ~{np.diff(em_ax).mean():.2f} nm)")
    print(f"          excitation [{ex_ax[0]:.1f}, {ex_ax[-1]:.1f}] nm "
          f"({len(ex_ax)} pts, step ~{np.diff(ex_ax).mean():.2f} nm)")
    print(f"Interpolation factor: {INTERP_FACTOR}× → "
          f"em step ~{np.diff(em_ax).mean() / INTERP_FACTOR:.3f} nm, "
          f"ex step ~{np.diff(ex_ax).mean() / INTERP_FACTOR:.3f} nm")
    print(f"Local max search radius: {SEARCH_RADIUS_NM} nm")

    print(f"\nChirality positions:")
    for name, c in CHIRALITY_POSITIONS.items():
        in_range_em = em_ax[0] <= c["emission_nm"] <= em_ax[-1]
        in_range_ex = ex_ax[0] <= c["excitation_nm"] <= ex_ax[-1]
        status = "OK" if (in_range_em and in_range_ex) else "OUT OF RANGE"
        print(f"  {name:8s}: em={c['emission_nm']:8.2f} nm, "
              f"ex={c['excitation_nm']:7.2f} nm  [{status}]")

    all_rows = []

    for i, code in enumerate(codes):
        print(f"\r  Processing {i+1}/{len(codes)}: {normalize_code(code):<40s}", end="", flush=True)

        row_dict = {"code": normalize_code(code)}

        for tp in TIMEPOINT_LABELS:
            try:
                filepath = find_file_flat(data_dir, code, tp)
            except FileNotFoundError as e:
                print(f"\n  WARNING: {e}")
                # Fill NaN for this timepoint
                for chir in chir_order:
                    for desc in per_chir_descs:
                        row_dict[f"{chir}_{desc}_{tp}"] = np.nan
                for chir_a, chir_b in chir_pairs:
                    row_dict[f"{chir_a}_vs_{chir_b}_ratio_{tp}"] = np.nan
                    row_dict[f"{chir_a}_vs_{chir_b}_eemdist_{tp}"] = np.nan
                continue

            # Extract per-chirality descriptors
            chir_results = extract_sample_tp(filepath, CHIRALITY_POSITIONS)

            for chir in chir_order:
                for desc in per_chir_descs:
                    col = f"{chir}_{desc}_{tp}"
                    row_dict[col] = chir_results[chir].get(desc, np.nan)

            # Pairwise features
            pw = compute_pairwise_features(chir_results, chir_order)
            for chir_a, chir_b in chir_pairs:
                row_dict[f"{chir_a}_vs_{chir_b}_ratio_{tp}"] = pw.get(
                    f"{chir_a}_vs_{chir_b}_ratio", np.nan)
                row_dict[f"{chir_a}_vs_{chir_b}_eemdist_{tp}"] = pw.get(
                    f"{chir_a}_vs_{chir_b}_eemdist", np.nan)

        all_rows.append(row_dict)

    print()  # newline after progress

    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Shape: {df.shape}")

    # Summary statistics
    print(f"\n--- Descriptor Summary ---")
    for tp in TIMEPOINT_LABELS:
        max_cols = [f"{c}_gauss_max_{tp}" for c in chir_order]
        fwhm_cols = [f"{c}_gauss_fwhm_{tp}" for c in chir_order]
        max_vals = df[max_cols].values.flatten()
        fwhm_vals = df[fwhm_cols].values.flatten()
        max_vals = max_vals[~np.isnan(max_vals)]
        fwhm_vals = fwhm_vals[~np.isnan(fwhm_vals)]
        print(f"  {tp}: gauss_max range [{max_vals.min():.4e}, {max_vals.max():.4e}], "
              f"FWHM range [{fwhm_vals.min():.2f}, {fwhm_vals.max():.2f}] nm")

    return df


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interpolation-based chirality descriptor extraction from EEM data"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory containing EEM xlsx files. Supports: "
             "(1) flat layout with _0h/_6h/_24h suffixes, "
             "(2) timepoint subdirs like out_0h/out_6h/out_24h"
    )
    parser.add_argument(
        "--labels_csv", type=str, required=True,
        help="CSV with columns: code, group"
    )
    parser.add_argument(
        "--output", type=str, default="chirality_interp_descriptors.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--interp_factor", type=int, default=INTERP_FACTOR,
        help=f"Interpolation upsampling factor (default: {INTERP_FACTOR})"
    )
    parser.add_argument(
        "--search_radius", type=float, default=SEARCH_RADIUS_NM,
        help=f"Local max search radius in nm (default: {SEARCH_RADIUS_NM})"
    )
    args = parser.parse_args()

    # Update module-level constants via the extraction function
    import chirality_interpolated_extraction as _self
    _self.INTERP_FACTOR = args.interp_factor
    _self.SEARCH_RADIUS_NM = args.search_radius

    run_extraction(args.data_dir, args.labels_csv, args.output)


if __name__ == "__main__":
    main()
