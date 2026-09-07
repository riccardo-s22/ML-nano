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
FWHM_WINDOW_FACTOR = 2.5     # fit window extends this × half-width from peak

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


def split_gaussian_with_baseline(x, amplitude, center, sigma_left, sigma_right, baseline):
    """
    Asymmetric (split / bi-) Gaussian with a constant baseline.

    Uses sigma_left for x <= center and sigma_right for x > center, giving
    independent control over the steep short-wavelength edge and the longer
    long-wavelength tail typical of SWCNT fluorescence peaks.
    """
    result = np.empty_like(x, dtype=np.float64)
    left = x <= center
    right = ~left
    result[left] = baseline + amplitude * np.exp(
        -0.5 * ((x[left] - center) / sigma_left) ** 2)
    result[right] = baseline + amplitude * np.exp(
        -0.5 * ((x[right] - center) / sigma_right) ** 2)
    return result


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
    window_factor: float = FWHM_WINDOW_FACTOR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the 1D emission profile at fixed excitation, then select the
    fitting window around the peak using a descending-gradient criterion.

    Algorithm
    ---------
    1.  From the local-maximum index, walk RIGHT along emission (fixed
        excitation).  At each step, compare to the running minimum seen so
        far.  If the new intensity is <= running_min -> descending, reset the
        reversal counter.  Otherwise -> reversal; after `reversal_limit`
        consecutive reversals, stop.
    2.  Repeat walking LEFT.
    3.  The gradient walk gives the maximal extent of monotone descent.
        To prevent including long non-Gaussian tails, the window is then
        capped at `window_factor` x half-width on each side, where the
        half-width is measured as the distance from peak to the 50% level
        (midpoint between peak and local baseline).
    4.  ALL contiguous points within the final window are returned (no
        gaps), preserving the natural profile shape for the fit.

    Returns (em_points, intensity_points).
    """
    # Full 1D profile along emission at fixed excitation
    profile = spline(em_fine, np.array([fixed_ex]))[:, 0]
    peak_idx = int(np.argmin(np.abs(em_fine - peak_em)))
    peak_val = profile[peak_idx]

    # ------------------------------------------------------------------
    # Step 1 & 2: gradient walk to find the maximal monotone-descent span
    # ------------------------------------------------------------------

    # Walk RIGHT
    right_stop = peak_idx
    consec_rev = 0
    running_min = peak_val
    for i in range(peak_idx + 1, len(profile)):
        if profile[i] <= running_min:
            running_min = profile[i]
            right_stop = i
            consec_rev = 0
        else:
            consec_rev += 1
            if consec_rev >= reversal_limit:
                break

    # Walk LEFT
    left_stop = peak_idx
    consec_rev = 0
    running_min = peak_val
    for i in range(peak_idx - 1, -1, -1):
        if profile[i] <= running_min:
            running_min = profile[i]
            left_stop = i
            consec_rev = 0
        else:
            consec_rev += 1
            if consec_rev >= reversal_limit:
                break

    # ------------------------------------------------------------------
    # Step 3: cap the window at window_factor x half-width
    # ------------------------------------------------------------------
    local_baseline = min(profile[left_stop], profile[right_stop])
    half_level = local_baseline + 0.5 * (peak_val - local_baseline)

    # Measure half-width on each side
    hw_right = None
    for i in range(peak_idx, right_stop + 1):
        if profile[i] <= half_level:
            hw_right = em_fine[i] - em_fine[peak_idx]
            break
    hw_left = None
    for i in range(peak_idx, left_stop - 1, -1):
        if profile[i] <= half_level:
            hw_left = em_fine[peak_idx] - em_fine[i]
            break

    # Use symmetric fallback if one side has no crossing
    if hw_right is None and hw_left is not None:
        hw_right = hw_left
    elif hw_left is None and hw_right is not None:
        hw_left = hw_right
    elif hw_left is None and hw_right is None:
        hw_right = hw_left = 15.0  # nm

    # Cap: gradient extent vs window_factor x half-width
    max_right_nm = window_factor * hw_right
    max_left_nm = window_factor * hw_left

    grad_right_nm = em_fine[right_stop] - em_fine[peak_idx]
    actual_right_nm = min(grad_right_nm, max_right_nm)

    grad_left_nm = em_fine[peak_idx] - em_fine[left_stop]
    actual_left_nm = min(grad_left_nm, max_left_nm)

    right_bound = em_fine[peak_idx] + actual_right_nm
    left_bound = em_fine[peak_idx] - actual_left_nm

    mask = (em_fine >= left_bound) & (em_fine <= right_bound)
    final_indices = np.where(mask)[0]

    # ------------------------------------------------------------------
    # Step 4: fallback - ensure at least 6 points for Gaussian + baseline
    # ------------------------------------------------------------------
    if len(final_indices) < 6:
        half_win = 20
        lo = max(0, peak_idx - half_win)
        hi = min(len(em_fine), peak_idx + half_win + 1)
        final_indices = np.arange(lo, hi)

    return em_fine[final_indices], profile[final_indices]


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
    Fit a split (asymmetric) Gaussian with baseline to the selected emission
    points and extract descriptors.

    The data is normalised to [0, 1] before fitting to avoid numerical issues
    with very small fluorescence intensities (~1e-14).  The split Gaussian
    has independent sigma_left and sigma_right to capture the characteristic
    steep short-wavelength edge and long-wavelength tail of SWCNT peaks.

    Falls back to empirical estimates if fitting fails.
    """
    result = {
        "gauss_max": peak_intensity_init,
        "gauss_em_center": peak_em_init,
        "gauss_fwhm": np.nan,
        "gauss_auc": np.nan,
        "gauss_sigma_left": np.nan,
        "gauss_sigma_right": np.nan,
        "gauss_skew": 0.0,
        "gauss_kurt": 0.0,
        "gauss_r2": np.nan,
        "gauss_rmse": np.nan,
        "gauss_npts": float(len(em_pts)),
    }

    if len(em_pts) < 5:
        return result

    # ------------------------------------------------------------------
    # Intensity-weighted shape statistics (skewness, kurtosis)
    # ------------------------------------------------------------------
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
        result["gauss_skew"] = float(sp_skew(int_pts, bias=True))
        result["gauss_kurt"] = float(sp_kurtosis(int_pts, fisher=True, bias=True))

    # ------------------------------------------------------------------
    # Normalise intensity to [0, 1] for numerical stability
    # ------------------------------------------------------------------
    scale_factor = float(int_pts.max())
    if scale_factor < 1e-30:
        return result
    int_norm = int_pts / scale_factor

    baseline_norm = float(int_norm.min())
    amp_norm = float(1.0 - baseline_norm)
    if amp_norm <= 0:
        amp_norm = 0.01

    # Initial sigma estimates from half-widths
    half_level = baseline_norm + 0.5 * amp_norm
    peak_local_idx = int(np.argmax(int_norm))
    peak_em_local = em_pts[peak_local_idx]

    sig_left_init = sig_right_init = 8.0  # fallback
    for i in range(peak_local_idx, -1, -1):
        if int_norm[i] <= half_level:
            sig_left_init = max((peak_em_local - em_pts[i]) / 1.177, 1.0)
            break
    for i in range(peak_local_idx, len(int_norm)):
        if int_norm[i] <= half_level:
            sig_right_init = max((em_pts[i] - peak_em_local) / 1.177, 1.0)
            break

    # ------------------------------------------------------------------
    # Fit split (asymmetric) Gaussian + baseline on normalised data
    # ------------------------------------------------------------------
    try:
        popt, pcov = curve_fit(
            split_gaussian_with_baseline,
            em_pts,
            int_norm,
            p0=[amp_norm, peak_em_local, sig_left_init, sig_right_init, baseline_norm],
            bounds=(
                [0,   em_pts.min() - 5, 0.5, 0.5, 0],
                [2.0, em_pts.max() + 5, 80.0, 80.0, 1.0],
            ),
            maxfev=10000,
        )
        amp_n, center_n, sig_l, sig_r, bl_n = popt

        # FWHM: half-width at half-max on each side is sqrt(2*ln2) * sigma
        fwhm = 1.17741 * (sig_l + sig_r)

        # AUC of split Gaussian (sum of two half-Gaussians)
        auc_norm = amp_n * np.sqrt(2.0 * np.pi) * (sig_l + sig_r) / 2.0
        auc_orig = auc_norm * scale_factor

        # De-normalise peak height
        total_peak = (amp_n + bl_n) * scale_factor

        result["gauss_max"] = float(total_peak)
        result["gauss_em_center"] = float(center_n)
        result["gauss_fwhm"] = float(fwhm)
        result["gauss_auc"] = float(auc_orig)
        result["gauss_sigma_left"] = float(sig_l)
        result["gauss_sigma_right"] = float(sig_r)

        # Goodness-of-fit (on normalised data - R2 is scale-invariant)
        fitted_norm = split_gaussian_with_baseline(em_pts, *popt)
        ss_res = float(np.sum((int_norm - fitted_norm) ** 2))
        ss_tot = float(np.sum((int_norm - np.mean(int_norm)) ** 2))
        result["gauss_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else np.nan

        # RMSE in original intensity units
        fitted_orig = fitted_norm * scale_factor
        result["gauss_rmse"] = float(np.sqrt(np.mean((int_pts - fitted_orig) ** 2)))

    except (RuntimeError, ValueError, TypeError):
        # Fallback: empirical estimates
        result["gauss_max"] = float(peak_intensity_init)
        result["gauss_em_center"] = float(peak_em_init)

        baseline_est = float(np.min(int_pts))
        half_max = (peak_intensity_init + baseline_est) / 2.0
        above_half = em_pts[int_pts >= half_max]
        if len(above_half) >= 2:
            result["gauss_fwhm"] = float(above_half[-1] - above_half[0])

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
        "gauss_fwhm", "gauss_auc", "gauss_sigma_left", "gauss_sigma_right",
        "gauss_skew", "gauss_kurt",
        "gauss_r2", "gauss_rmse", "gauss_npts",
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
        r2_cols = [f"{c}_gauss_r2_{tp}" for c in chir_order]
        npts_cols = [f"{c}_gauss_npts_{tp}" for c in chir_order]

        max_vals = df[max_cols].values.flatten()
        fwhm_vals = df[fwhm_cols].values.flatten()
        r2_vals = df[r2_cols].values.flatten()
        npts_vals = df[npts_cols].values.flatten()

        max_vals = max_vals[~np.isnan(max_vals)]
        fwhm_vals = fwhm_vals[~np.isnan(fwhm_vals)]
        r2_vals = r2_vals[~np.isnan(r2_vals)]
        npts_vals = npts_vals[~np.isnan(npts_vals)]

        print(f"  {tp}: gauss_max range [{max_vals.min():.4e}, {max_vals.max():.4e}], "
              f"FWHM range [{fwhm_vals.min():.2f}, {fwhm_vals.max():.2f}] nm")
        if len(r2_vals) > 0:
            print(f"       R² range [{r2_vals.min():.4f}, {r2_vals.max():.4f}], "
                  f"median R²={np.median(r2_vals):.4f}")
            print(f"       Fit points range [{npts_vals.min():.0f}, {npts_vals.max():.0f}], "
                  f"median={np.median(npts_vals):.0f}")

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
