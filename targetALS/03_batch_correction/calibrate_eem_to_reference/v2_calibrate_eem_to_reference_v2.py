#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibrate_eem_to_reference_v2.py

Reference-based spectral calibration for EEM data prior to proxy latent extraction.
v2 — Fixed for nIR SWCNT data where intensities are at ~1e-14 scale.

WHAT WAS WRONG IN v1:
  1. Absolute thresholds (e.g., `if norm < 1e-12: return 0.0`) killed all 
     computation when data intensities are at 1e-14 scale. Fixed: all 
     thresholds are now RELATIVE to the data.

  2. Single excitation-column profiles had SNR ~0.1 — far too noisy for 
     cross-correlation. Fixed: sum ±N excitation columns around each 
     chirality to boost SNR (SNR goes from ~0.1 to ~3-5).

  3. Excitation intensity variation (column-wise sinusoidal gain, ±40%) 
     was not addressed. Fixed: added per-column gain correction using 
     matched original-surrogate column intensities.

CORRECTS (in order):
  1. Wavelength (emission) shift — cross-correlation of summed chirality profiles
  2. Excitation intensity variation — per-column gain normalization
  3. Baseline drift — polynomial fit to signal-free emission edges
  4. Global intensity scaling — median peak ratio at chiralities

REFERENCE STRATEGIES (--reference_mode):
  "template"    — Mean EEM from training directory (best for synthetic surrogates)
  "designated"  — Specific reference sample per batch (gold standard for real data)
  "self_peaks"  — Intrinsic SWCNT E11 positions (no reference needed)

USAGE:
  python calibrate_eem_to_reference_v2.py \
      --input_dir exp7/3/out_0h \
      --output_dir exp7/3/calibrated/out_0h \
      --reference_mode template \
      --template_dir exp5/out_0h \
      --reference_peaks 6.5 8.3 10.2 \
      --ex_sum_half 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate


# ─── SWCNT Chirality Reference Table ─────────────────────────────────────────
CHIRALITY_CENTERS = {
    '6.5':  (573, 975),
    '7.5':  (647, 1024),
    '7.6':  (645, 1115),
    '8.3':  (667, 952),
    '8.4':  (726, 1100),
    '8.6':  (718, 1170),
    '8.7':  (726, 1260),
    '9.4':  (720, 1100),
    '9.5':  (800, 1240),
    '10.2': (740, 1050),
    '10.3': (800, 1100),
    '10.5': (850, 1250),
}


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_eem(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load EEM Excel file.
    Returns: (data, emission_axis, excitation_axis, excitation_col_names)
    data shape: (n_emission, n_excitation)
    """
    df = pd.read_excel(filepath, engine="openpyxl")
    em_col = df.columns[0]
    emission_axis = df[em_col].values.astype(float)
    ex_cols = [c for c in df.columns if c != em_col]
    excitation_axis = np.array([
        float(str(c).replace("Excitation_", "").strip())
        for c in ex_cols
    ])
    data = df[ex_cols].values.astype(float)
    return data, emission_axis, excitation_axis, ex_cols


def save_eem(filepath: str, data: np.ndarray,
             emission_axis: np.ndarray, ex_col_names: list):
    """Save corrected EEM back to Excel in original format."""
    df = pd.DataFrame(data, columns=ex_col_names)
    df.insert(0, 'Emission', emission_axis)
    df.to_excel(filepath, index=False, engine="openpyxl")


# ─── Relative-threshold helpers ───────────────────────────────────────────────

def _relative_norm_ok(profile: np.ndarray, min_relative_snr: float = 0.5) -> bool:
    """Check if a profile has enough variation to be useful (relative, not absolute)."""
    centered = profile - np.mean(profile)
    if np.max(np.abs(profile)) < 1e-30:   # truly zero data
        return False
    snr = np.std(centered) / (np.mean(np.abs(profile)) + 1e-30)
    return snr > 0.01  # at least 1% relative variation


def _get_summed_profile(data: np.ndarray, excitation_axis: np.ndarray,
                        ex_target_nm: float, ex_sum_half: int = 2) -> np.ndarray:
    """
    Extract emission profile summed over ±ex_sum_half excitation columns 
    around the target wavelength. This boosts SNR from ~0.1 to ~3-5.
    """
    ex_idx = int(np.argmin(np.abs(excitation_axis - ex_target_nm)))
    lo = max(0, ex_idx - ex_sum_half)
    hi = min(data.shape[1], ex_idx + ex_sum_half + 1)
    return np.sum(data[:, lo:hi], axis=1)


# ─── Wavelength shift estimation ─────────────────────────────────────────────

def estimate_shift_xcorr(profile_ref: np.ndarray, profile_new: np.ndarray,
                         max_shift_pixels: int = 20) -> Tuple[float, float]:
    """
    Estimate sub-pixel emission shift via cross-correlation.
    Uses relative normalization — works at any intensity scale.
    
    Returns: (shift_pixels, peak_correlation)
    """
    ref = profile_ref - np.mean(profile_ref)
    new = profile_new - np.mean(profile_new)

    ref_norm = np.linalg.norm(ref)
    new_norm = np.linalg.norm(new)

    # Relative check: is there any variation at all?
    if ref_norm / (np.mean(np.abs(profile_ref)) + 1e-30) < 1e-6:
        return 0.0, 0.0
    if new_norm / (np.mean(np.abs(profile_new)) + 1e-30) < 1e-6:
        return 0.0, 0.0

    ref = ref / ref_norm
    new = new / new_norm

    xcorr = correlate(new, ref, mode='full')
    n = len(ref)
    lags = np.arange(-n + 1, n)

    valid = np.abs(lags) <= max_shift_pixels
    xcorr_valid = xcorr[valid]
    lags_valid = lags[valid]

    peak_idx = np.argmax(xcorr_valid)
    peak_lag = lags_valid[peak_idx]
    peak_corr = float(xcorr_valid[peak_idx])

    # Sub-pixel parabolic refinement
    if 0 < peak_idx < len(xcorr_valid) - 1:
        y_m = xcorr_valid[peak_idx - 1]
        y_c = xcorr_valid[peak_idx]
        y_p = xcorr_valid[peak_idx + 1]
        denom = 2.0 * (2.0 * y_c - y_m - y_p)
        if abs(denom) > 1e-12:
            delta = (y_m - y_p) / denom
            peak_lag = float(peak_lag) + delta

    return float(peak_lag), peak_corr


def estimate_shift_from_chiralities(
    data: np.ndarray,
    ref_data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    chiralities: List[str],
    ex_sum_half: int = 2,
    max_shift_pixels: int = 20,
) -> Tuple[float, dict]:
    """
    Estimate wavelength shift using multiple chirality peaks with summed profiles.
    Returns (median_shift_pixels, per_chirality_details).
    """
    shifts = []
    details = {}

    for chir in chiralities:
        if chir not in CHIRALITY_CENTERS:
            continue
        ex_target, em_target = CHIRALITY_CENTERS[chir]

        # Get summed profiles (boosts SNR dramatically)
        ref_profile = _get_summed_profile(ref_data, excitation_axis, ex_target, ex_sum_half)
        new_profile = _get_summed_profile(data, excitation_axis, ex_target, ex_sum_half)

        if not _relative_norm_ok(ref_profile) or not _relative_norm_ok(new_profile):
            details[chir] = {"status": "low_snr", "shift_px": 0.0}
            continue

        shift_px, corr = estimate_shift_xcorr(
            ref_profile, new_profile, max_shift_pixels=max_shift_pixels
        )
        shifts.append(shift_px)
        em_step = np.mean(np.diff(emission_axis))
        details[chir] = {
            "status": "ok",
            "shift_px": shift_px,
            "shift_nm": shift_px * em_step,
            "correlation": corr,
        }

    if not shifts:
        return 0.0, details

    median_shift = float(np.median(shifts))
    return median_shift, details


def estimate_shift_self_peaks(
    data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    chiralities: List[str],
    ex_sum_half: int = 2,
    search_window_nm: float = 30.0,
) -> Tuple[float, dict]:
    """
    Estimate shift from intrinsic SWCNT peak positions (no reference needed).
    """
    shifts_nm = []
    details = {}

    for chir in chiralities:
        if chir not in CHIRALITY_CENTERS:
            continue
        ex_target, em_expected = CHIRALITY_CENTERS[chir]

        profile = _get_summed_profile(data, excitation_axis, ex_target, ex_sum_half)

        if not _relative_norm_ok(profile):
            details[chir] = {"status": "low_snr", "shift_nm": 0.0}
            continue

        # Find peak in window
        em_lo = em_expected - search_window_nm
        em_hi = em_expected + search_window_nm
        mask = (emission_axis >= em_lo) & (emission_axis <= em_hi)

        if not np.any(mask):
            details[chir] = {"status": "no_window", "shift_nm": 0.0}
            continue

        window_profile = profile[mask]
        window_em = emission_axis[mask]

        peak_idx = np.argmax(window_profile)

        # Parabolic refinement
        if 0 < peak_idx < len(window_profile) - 1:
            y_m = window_profile[peak_idx - 1]
            y_c = window_profile[peak_idx]
            y_p = window_profile[peak_idx + 1]
            denom = 2.0 * (2.0 * y_c - y_m - y_p)
            if abs(denom) > 1e-12:
                delta = (y_m - y_p) / denom
                peak_em = window_em[peak_idx] + delta * np.mean(np.diff(window_em))
            else:
                peak_em = window_em[peak_idx]
        else:
            peak_em = window_em[peak_idx]

        shift_nm = float(peak_em - em_expected)
        shifts_nm.append(shift_nm)
        details[chir] = {"status": "ok", "shift_nm": shift_nm, "peak_em": float(peak_em)}

    if not shifts_nm:
        return 0.0, details

    median_shift_nm = float(np.median(shifts_nm))
    em_step = np.mean(np.diff(emission_axis))
    return median_shift_nm / em_step, details  # return in pixels


# ─── Excitation intensity variation correction ────────────────────────────────

def estimate_excitation_gain(
    data: np.ndarray,
    ref_data: np.ndarray,
) -> np.ndarray:
    """
    Estimate per-excitation-column multiplicative gain: data ≈ gain × ref.
    
    Returns gain_profile: (n_excitation,) array of per-column scale factors.
    Apply correction as: corrected = data / gain_profile
    """
    n_ex = data.shape[1]
    gains = np.ones(n_ex, dtype=float)

    for col in range(n_ex):
        ref_col = ref_data[:, col]
        new_col = data[:, col]

        # Use median ratio of absolute values in upper quartile
        # (avoids near-zero values dominating)
        ref_abs = np.abs(ref_col)
        threshold = np.percentile(ref_abs, 75)

        if threshold < np.max(ref_abs) * 1e-6:
            # Very little signal in this column — use global estimate
            threshold = np.percentile(ref_abs, 50)

        mask = ref_abs > threshold
        if np.sum(mask) < 5:
            # Not enough strong pixels
            gains[col] = 1.0
            continue

        # Robust ratio estimation
        ratios = new_col[mask] / ref_col[mask]
        gains[col] = float(np.median(ratios))

    # Safety: clip extreme gains
    median_gain = np.median(gains)
    gains = np.clip(gains, median_gain * 0.5, median_gain * 2.0)

    return gains


# ─── Baseline drift correction ────────────────────────────────────────────────

def estimate_baseline_drift(
    data: np.ndarray,
    emission_axis: np.ndarray,
    low_cutoff_nm: float = 870.0,
    high_cutoff_nm: float = 1600.0,
    poly_degree: int = 2,
) -> np.ndarray:
    """
    Estimate baseline drift from signal-free edges of emission range.
    Uses RELATIVE thresholds for detection.
    """
    low_mask = emission_axis <= low_cutoff_nm
    high_mask = emission_axis >= high_cutoff_nm

    edge_mask = low_mask | high_mask
    if np.sum(edge_mask) < poly_degree + 2:
        return np.zeros_like(data)

    x_all = np.arange(data.shape[0], dtype=float)
    x_edge = x_all[edge_mask]

    baseline = np.zeros_like(data, dtype=float)

    for col_idx in range(data.shape[1]):
        y_edge = data[edge_mask, col_idx]
        coeffs = np.polyfit(x_edge, y_edge, deg=poly_degree)
        baseline[:, col_idx] = np.polyval(coeffs, x_all)

    return baseline


# ─── Intensity scaling estimation ─────────────────────────────────────────────

def estimate_global_intensity_scale(
    data: np.ndarray,
    ref_data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    chiralities: List[str],
    ex_sum_half: int = 2,
    window_nm: float = 20.0,
) -> float:
    """
    Estimate global intensity scale from chirality peak amplitudes.
    Uses summed profiles for robustness at low intensities.
    """
    ratios = []

    for chir in chiralities:
        if chir not in CHIRALITY_CENTERS:
            continue
        ex_target, em_target = CHIRALITY_CENTERS[chir]

        ref_profile = _get_summed_profile(ref_data, excitation_axis, ex_target, ex_sum_half)
        new_profile = _get_summed_profile(data, excitation_axis, ex_target, ex_sum_half)

        em_mask = (emission_axis >= em_target - window_nm) & (emission_axis <= em_target + window_nm)
        if not np.any(em_mask):
            continue

        ref_peak = np.max(ref_profile[em_mask])
        new_peak = np.max(new_profile[em_mask])

        # Relative check: are peaks meaningful?
        ref_bg = np.median(ref_profile)
        new_bg = np.median(new_profile)

        if (ref_peak - ref_bg) > np.std(ref_profile) * 0.5:
            if (new_peak - new_bg) > np.std(new_profile) * 0.5:
                ratios.append(new_peak / ref_peak)

    if not ratios:
        return 1.0

    return float(np.median(ratios))


# ─── Apply corrections ───────────────────────────────────────────────────────

def apply_emission_shift(data: np.ndarray, shift_pixels: float) -> np.ndarray:
    """Apply emission shift correction (negative of estimated shift)."""
    corrected = np.zeros_like(data)
    for col in range(data.shape[1]):
        corrected[:, col] = scipy_shift(data[:, col], -shift_pixels, mode='nearest')
    return corrected


# ─── Full calibration pipeline ────────────────────────────────────────────────

def calibrate_single_eem(
    data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    reference_data: Optional[np.ndarray],
    chiralities: List[str],
    reference_mode: str,
    ex_sum_half: int = 2,
    correct_shift: bool = True,
    correct_excitation: bool = True,
    correct_baseline: bool = True,
    correct_intensity: bool = True,
    baseline_low_nm: float = 870.0,
    baseline_high_nm: float = 1600.0,
    max_shift_pixels: int = 20,
) -> Tuple[np.ndarray, dict]:
    """
    Full calibration of a single EEM. All thresholds are relative.
    
    Correction order matters:
      1. Wavelength shift (geometric — must come first)
      2. Excitation intensity variation (per-column gain)
      3. Baseline drift (additive polynomial)
      4. Global intensity scaling (multiplicative)
    """
    diag: dict = {
        "shift_nm": 0.0,
        "shift_pixels": 0.0,
        "excitation_gain_range": [1.0, 1.0],
        "intensity_scale": 1.0,
        "baseline_magnitude_relative": 0.0,
        "corrections_applied": [],
    }

    corrected = data.copy()
    em_step = np.mean(np.diff(emission_axis))

    # ── 1. Wavelength shift ───────────────────────────────────────────────────
    if correct_shift:
        if reference_mode == "self_peaks":
            shift_px, shift_details = estimate_shift_self_peaks(
                corrected, emission_axis, excitation_axis,
                chiralities, ex_sum_half=ex_sum_half,
            )
        elif reference_mode in ("template", "designated") and reference_data is not None:
            shift_px, shift_details = estimate_shift_from_chiralities(
                corrected, reference_data, emission_axis, excitation_axis,
                chiralities, ex_sum_half=ex_sum_half,
                max_shift_pixels=max_shift_pixels,
            )
        else:
            shift_px, shift_details = 0.0, {}

        if abs(shift_px) > 0.05:  # more than 0.05 pixels ≈ 0.08 nm
            corrected = apply_emission_shift(corrected, shift_px)
            diag["corrections_applied"].append("wavelength_shift")

        diag["shift_pixels"] = shift_px
        diag["shift_nm"] = shift_px * em_step
        diag["shift_details"] = shift_details

    # ── 2. Excitation intensity variation ─────────────────────────────────────
    if correct_excitation and reference_data is not None:
        gains = estimate_excitation_gain(corrected, reference_data)
        gain_range = [float(np.min(gains)), float(np.max(gains))]

        # Only correct if there's meaningful variation
        gain_cv = np.std(gains) / (np.mean(gains) + 1e-30)
        if gain_cv > 0.01:  # >1% coefficient of variation
            corrected = corrected / gains[np.newaxis, :]
            diag["corrections_applied"].append("excitation_variation")

        diag["excitation_gain_range"] = gain_range
        diag["excitation_gain_cv"] = float(gain_cv)

    # ── 3. Baseline drift ─────────────────────────────────────────────────────
    if correct_baseline:
        baseline = estimate_baseline_drift(
            corrected, emission_axis,
            low_cutoff_nm=baseline_low_nm,
            high_cutoff_nm=baseline_high_nm,
        )
        data_scale = np.std(corrected)
        baseline_relative = np.mean(np.abs(baseline)) / (data_scale + 1e-30)
        
        if baseline_relative > 0.001:  # >0.1% of signal
            corrected = corrected - baseline
            diag["corrections_applied"].append("baseline_drift")

        diag["baseline_magnitude_relative"] = float(baseline_relative)

    # ── 4. Global intensity scaling ───────────────────────────────────────────
    if correct_intensity and reference_data is not None:
        scale = estimate_global_intensity_scale(
            corrected, reference_data, emission_axis, excitation_axis,
            chiralities, ex_sum_half=ex_sum_half,
        )
        if 0.1 < scale < 10.0 and abs(scale - 1.0) > 0.01:
            corrected = corrected / scale
            diag["corrections_applied"].append("intensity_scaling")

        diag["intensity_scale"] = scale

    return corrected, diag


# ─── Batch processing ────────────────────────────────────────────────────────

def build_template(template_dir: str, max_files: int = 100
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Build mean EEM template from training directory."""
    template_dir = Path(template_dir)
    files = sorted(template_dir.glob("*.xlsx"))[:max_files]
    if not files:
        raise FileNotFoundError(f"No .xlsx in {template_dir}")

    mats = []
    em_axis = ex_axis = ex_cols = None

    for f in files:
        d, em, ex, cols = load_eem(str(f))
        if em_axis is None:
            em_axis, ex_axis, ex_cols = em, ex, cols
        mats.append(d)

    mean_data = np.mean(np.stack(mats), axis=0)
    print(f"  Template: {len(mats)} files, shape {mean_data.shape}, "
          f"intensity range [{np.min(mean_data):.2e}, {np.max(mean_data):.2e}]")
    return mean_data, em_axis, ex_axis, ex_cols


def main():
    ap = argparse.ArgumentParser(
        description="Reference-based EEM calibration v2 (handles nIR-scale data)."
    )
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--reference_mode", required=True,
                    choices=["template", "designated", "self_peaks"])
    ap.add_argument("--template_dir", default=None)
    ap.add_argument("--designated_ref_train", default=None)
    ap.add_argument("--designated_ref_new", default=None)
    ap.add_argument("--reference_peaks", nargs="+", default=["6.5", "8.3", "10.2"])
    ap.add_argument("--ex_sum_half", type=int, default=2,
                    help="Sum ±N excitation columns around each chirality for SNR boost. "
                         "Default 2 (sums 5 columns × 5nm = 25nm band).")
    ap.add_argument("--no_shift_correction", action="store_true")
    ap.add_argument("--no_excitation_correction", action="store_true")
    ap.add_argument("--no_baseline_correction", action="store_true")
    ap.add_argument("--no_intensity_correction", action="store_true")
    ap.add_argument("--baseline_low_nm", type=float, default=870.0)
    ap.add_argument("--baseline_high_nm", type=float, default=1600.0)
    ap.add_argument("--max_shift_pixels", type=int, default=20)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate chiralities
    for chir in args.reference_peaks:
        if chir not in CHIRALITY_CENTERS:
            raise ValueError(f"Unknown chirality '{chir}'")

    # ── Build reference ───────────────────────────────────────────────────────
    reference_data = None

    if args.reference_mode == "template":
        if not args.template_dir:
            raise ValueError("--template_dir required for template mode")
        print(f"Building template from {args.template_dir}...")
        reference_data, ref_em, ref_ex, ref_cols = build_template(args.template_dir)

    elif args.reference_mode == "designated":
        if not args.designated_ref_train:
            raise ValueError("--designated_ref_train required")
        reference_data, ref_em, ref_ex, ref_cols = load_eem(args.designated_ref_train)

    # ── Process ───────────────────────────────────────────────────────────────
    input_files = sorted(input_dir.glob("*.xlsx"))
    if not input_files:
        raise FileNotFoundError(f"No .xlsx in {input_dir}")

    print(f"\nCalibrating {len(input_files)} files...")
    print(f"  Mode: {args.reference_mode}")
    print(f"  Chiralities: {args.reference_peaks}")
    print(f"  Ex sum half: ±{args.ex_sum_half} columns "
          f"({2*args.ex_sum_half + 1} × 5nm = {(2*args.ex_sum_half + 1)*5}nm band)")
    print(f"  Corrections: shift={not args.no_shift_correction}, "
          f"excitation={not args.no_excitation_correction}, "
          f"baseline={not args.no_baseline_correction}, "
          f"intensity={not args.no_intensity_correction}")

    report = {"reference_mode": args.reference_mode, "files": {}}

    for i, filepath in enumerate(input_files):
        data, em_axis, ex_axis, ex_cols = load_eem(str(filepath))

        corrected, diag = calibrate_single_eem(
            data, em_axis, ex_axis,
            reference_data=reference_data,
            chiralities=args.reference_peaks,
            reference_mode=args.reference_mode,
            ex_sum_half=args.ex_sum_half,
            correct_shift=not args.no_shift_correction,
            correct_excitation=not args.no_excitation_correction,
            correct_baseline=not args.no_baseline_correction,
            correct_intensity=not args.no_intensity_correction,
            baseline_low_nm=args.baseline_low_nm,
            baseline_high_nm=args.baseline_high_nm,
            max_shift_pixels=args.max_shift_pixels,
        )

        save_eem(str(output_dir / filepath.name), corrected, em_axis, ex_cols)
        report["files"][filepath.name] = diag

        if (i + 1) % 10 == 0 or i == 0:
            corrs = [c for c in diag["corrections_applied"]]
            print(f"  [{i+1}/{len(input_files)}] {filepath.name}: "
                  f"shift={diag['shift_nm']:.3f}nm, "
                  f"ex_gain=[{diag['excitation_gain_range'][0]:.3f},{diag['excitation_gain_range'][1]:.3f}], "
                  f"scale={diag['intensity_scale']:.4f}  "
                  f"applied: {corrs}")

    # ── Save report ───────────────────────────────────────────────────────────
    # Make report JSON-serializable
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(_clean(report), f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    shifts = [d.get("shift_nm", 0) for d in report["files"].values()]
    scales = [d.get("intensity_scale", 1) for d in report["files"].values()]
    gains = [d.get("excitation_gain_range", [1, 1]) for d in report["files"].values()]
    baselines = [d.get("baseline_magnitude_relative", 0) for d in report["files"].values()]

    gain_mins = [g[0] for g in gains]
    gain_maxs = [g[1] for g in gains]

    print(f"\n{'='*60}")
    print(f"Calibration complete: {len(input_files)} files")
    print(f"  Shift (nm):     median={np.median(shifts):.3f}  range=[{np.min(shifts):.3f}, {np.max(shifts):.3f}]")
    print(f"  Intensity:      median={np.median(scales):.4f}  range=[{np.min(scales):.4f}, {np.max(scales):.4f}]")
    print(f"  Ex gain range:  [{np.median(gain_mins):.3f}, {np.median(gain_maxs):.3f}]")
    print(f"  Baseline (rel): median={np.median(baselines):.4f}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
