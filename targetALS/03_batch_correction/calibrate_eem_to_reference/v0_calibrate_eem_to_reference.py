#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibrate_eem_to_reference.py

Reference-based spectral calibration for EEM data prior to proxy latent extraction.

Corrects three types of batch effects at the raw EEM level:
  1. Wavelength (emission) shift — estimated via cross-correlation of chirality peaks
  2. Intensity scaling — estimated from reference peak amplitudes
  3. Baseline drift — estimated from signal-free spectral regions

WHY THIS MATTERS:
Your proxy latent pipeline defines ROIs in pixel coordinates. A wavelength shift
moves spectral content across ROI boundaries, fundamentally changing which physical
signal the feature extraction kernels operate on. No downstream z-score correction
can fix this — it must be corrected at the image level before feature extraction.

REFERENCE STRATEGIES (--reference_mode):

  "template"  (RECOMMENDED for synthetic surrogates)
      Build a mean EEM template from training data. For each new-batch sample,
      estimate the shift by cross-correlating emission profiles at strong
      chirality peaks with the template. Best when you have the original
      training data available.

  "designated"
      Use a specific designated reference sample (e.g., SWCNT + buffer,
      no serum). Include this same reference in every measurement batch.
      Compare it to the training reference to estimate batch parameters.
      Gold standard for real experimental batches.

  "self_peaks"
      Use SWCNT chirality peak positions as intrinsic references. The E11
      emission of each chirality is a fixed property of the nanotube species;
      any deviation from expected position = instrumental shift. Does NOT
      require a reference sample. Works sample-by-sample.

OUTPUTS:
  - Corrected EEM Excel files (same format as input)
  - calibration_report.json (per-sample shift/scale estimates for QC)

USAGE EXAMPLES:

  # Using training data as template reference
  python calibrate_eem_to_reference.py \
      --input_dir synthetic_surrogates/out_0h \
      --output_dir calibrated/out_0h \
      --reference_mode template \
      --template_dir original_data/out_0h \
      --reference_peaks 6.5 8.3 10.2

  # Using a designated reference sample
  python calibrate_eem_to_reference.py \
      --input_dir synthetic_surrogates/out_0h \
      --output_dir calibrated/out_0h \
      --reference_mode designated \
      --designated_ref_new batch_reference.xlsx \
      --designated_ref_train training_reference.xlsx

  # Using intrinsic peak positions (no reference sample needed)
  python calibrate_eem_to_reference.py \
      --input_dir synthetic_surrogates/out_0h \
      --output_dir calibrated/out_0h \
      --reference_mode self_peaks \
      --reference_peaks 6.5 8.3 10.2
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
# (excitation_nm, emission_nm) — E22/E11 pairs
# These are intrinsic to the nanotube species and do NOT change with protein corona

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

def load_eem_with_axes(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EEM Excel file and extract axes.

    Returns:
        data: (n_emission, n_excitation) array
        emission_axis: wavelengths in nm (row axis)
        excitation_axis: wavelengths in nm (column axis)
    """
    df = pd.read_excel(filepath, engine="openpyxl")

    # First column is typically 'Emission' (wavelength axis)
    em_col = df.columns[0]
    emission_axis = df[em_col].values.astype(float)

    # Remaining columns are excitation wavelengths
    excitation_cols = [c for c in df.columns if c != em_col]
    excitation_axis = np.array([float(str(c).replace("Excitation_", ""))
                                for c in excitation_cols])

    data = df[excitation_cols].values.astype(float)

    return data, emission_axis, excitation_axis


def save_eem_with_axes(filepath: str, data: np.ndarray,
                       emission_axis: np.ndarray, excitation_cols: list):
    """Save corrected EEM back to Excel in original format."""
    df = pd.DataFrame(data, columns=excitation_cols)
    df.insert(0, 'Emission', emission_axis)
    df.to_excel(filepath, index=False)


# ─── Cross-correlation shift estimation ───────────────────────────────────────

def estimate_shift_xcorr(profile_ref: np.ndarray, profile_new: np.ndarray,
                         max_shift_pixels: int = 20) -> float:
    """
    Estimate sub-pixel emission shift via cross-correlation.

    Args:
        profile_ref: 1D emission profile from reference (template or designated)
        profile_new: 1D emission profile from new-batch sample
        max_shift_pixels: maximum allowed shift in pixels (safety bound)

    Returns:
        shift_pixels: estimated shift (positive = new data shifted to higher indices)
                      Apply -shift_pixels to correct.
    """
    # Normalize to zero-mean, unit-variance for correlation
    ref = profile_ref - np.mean(profile_ref)
    new = profile_new - np.mean(profile_new)

    ref_norm = np.linalg.norm(ref)
    new_norm = np.linalg.norm(new)

    if ref_norm < 1e-12 or new_norm < 1e-12:
        return 0.0

    ref = ref / ref_norm
    new = new / new_norm

    # Full cross-correlation
    xcorr = correlate(new, ref, mode='full')
    n = len(ref)
    lags = np.arange(-n + 1, n)

    # Restrict to reasonable shift range
    valid = np.abs(lags) <= max_shift_pixels
    xcorr_valid = xcorr[valid]
    lags_valid = lags[valid]

    # Peak of cross-correlation
    peak_idx = np.argmax(xcorr_valid)
    peak_lag = lags_valid[peak_idx]

    # Sub-pixel refinement via parabolic interpolation
    if 0 < peak_idx < len(xcorr_valid) - 1:
        y_minus = xcorr_valid[peak_idx - 1]
        y_center = xcorr_valid[peak_idx]
        y_plus = xcorr_valid[peak_idx + 1]
        denom = 2.0 * (2.0 * y_center - y_minus - y_plus)
        if abs(denom) > 1e-12:
            delta = (y_minus - y_plus) / denom
            peak_lag = float(peak_lag) + delta

    return float(peak_lag)


def estimate_shift_from_peak_position(
    data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    chirality: str,
    search_window_nm: float = 30.0,
) -> float:
    """
    Estimate emission shift from a single chirality peak position.

    Finds the peak maximum in a local window around the expected position
    and returns the offset from the expected E11 emission.

    Returns:
        shift_nm: observed - expected emission position (nm)
    """
    ex_expected, em_expected = CHIRALITY_CENTERS[chirality]

    # Find closest excitation index
    ex_idx = int(np.argmin(np.abs(excitation_axis - ex_expected)))

    # Extract emission profile at this excitation
    profile = data[:, ex_idx]

    # Define search window around expected emission
    em_lo = em_expected - search_window_nm
    em_hi = em_expected + search_window_nm
    mask = (emission_axis >= em_lo) & (emission_axis <= em_hi)

    if not np.any(mask):
        return 0.0

    window_profile = profile[mask]
    window_em = emission_axis[mask]

    # Find peak (use parabolic refinement for sub-pixel)
    peak_idx = np.argmax(window_profile)

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

    return float(peak_em - em_expected)


# ─── Baseline drift estimation ────────────────────────────────────────────────

def estimate_baseline_drift(data: np.ndarray, emission_axis: np.ndarray,
                            low_cutoff_nm: float = 910.0,
                            high_cutoff_nm: float = 1340.0,
                            poly_degree: int = 2) -> np.ndarray:
    """
    Estimate baseline drift from signal-free edges of the emission range.

    Uses the extreme low and high emission regions (where SWCNT fluorescence
    is minimal) to fit a polynomial baseline per excitation column.

    Returns:
        baseline: same shape as data, to be subtracted
    """
    low_mask = emission_axis <= low_cutoff_nm
    high_mask = emission_axis >= high_cutoff_nm

    edge_mask = low_mask | high_mask
    if np.sum(edge_mask) < poly_degree + 1:
        # Not enough edge points; return zero baseline
        return np.zeros_like(data)

    x_all = np.arange(data.shape[0], dtype=float)
    x_edge = x_all[edge_mask]

    baseline = np.zeros_like(data, dtype=float)

    for col_idx in range(data.shape[1]):
        y_edge = data[edge_mask, col_idx]
        # Fit polynomial to edge regions
        coeffs = np.polyfit(x_edge, y_edge, deg=poly_degree)
        baseline[:, col_idx] = np.polyval(coeffs, x_all)

    return baseline


# ─── Intensity scaling estimation ─────────────────────────────────────────────

def estimate_intensity_scale(
    data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    ref_data: np.ndarray,
    chiralities: List[str],
    window_nm: float = 20.0,
) -> float:
    """
    Estimate multiplicative intensity scale factor from reference peak amplitudes.

    Compares peak intensities at specified chirality positions between
    the new data and reference, returns the median ratio.
    """
    ratios = []

    for chir in chiralities:
        ex_expected, em_expected = CHIRALITY_CENTERS[chir]
        ex_idx = int(np.argmin(np.abs(excitation_axis - ex_expected)))

        # Find peak in window for both reference and new data
        em_lo = em_expected - window_nm
        em_hi = em_expected + window_nm
        mask = (emission_axis >= em_lo) & (emission_axis <= em_hi)

        if not np.any(mask):
            continue

        ref_peak = np.max(ref_data[mask, ex_idx])
        new_peak = np.max(data[mask, ex_idx])

        if ref_peak > 1e-12 and new_peak > 1e-12:
            ratios.append(new_peak / ref_peak)

    if not ratios:
        return 1.0

    return float(np.median(ratios))


# ─── Apply corrections ───────────────────────────────────────────────────────

def apply_emission_shift(data: np.ndarray, shift_pixels: float) -> np.ndarray:
    """Apply emission axis shift correction to entire EEM."""
    corrected = np.zeros_like(data)
    for col in range(data.shape[1]):
        corrected[:, col] = scipy_shift(data[:, col], -shift_pixels, mode='nearest')
    return corrected


def calibrate_single_eem(
    data: np.ndarray,
    emission_axis: np.ndarray,
    excitation_axis: np.ndarray,
    reference_data: Optional[np.ndarray],
    reference_chiralities: List[str],
    reference_mode: str,
    correct_shift: bool = True,
    correct_baseline: bool = True,
    correct_intensity: bool = True,
    baseline_low_nm: float = 910.0,
    baseline_high_nm: float = 1340.0,
    max_shift_pixels: int = 20,
) -> Tuple[np.ndarray, dict]:
    """
    Full calibration of a single EEM.

    Returns:
        corrected_data: calibrated EEM
        diagnostics: dict with estimated parameters
    """
    diag = {
        "shift_nm": 0.0,
        "shift_pixels": 0.0,
        "intensity_scale": 1.0,
        "baseline_magnitude": 0.0,
        "corrections_applied": [],
    }

    corrected = data.copy()

    # ── Step 1: Wavelength shift correction ───────────────────────────────────
    if correct_shift:
        if reference_mode == "self_peaks":
            # Estimate from intrinsic peak positions
            shifts_nm = []
            for chir in reference_chiralities:
                if chir in CHIRALITY_CENTERS:
                    s = estimate_shift_from_peak_position(
                        corrected, emission_axis, excitation_axis, chir
                    )
                    shifts_nm.append(s)

            if shifts_nm:
                median_shift_nm = float(np.median(shifts_nm))
                em_step = np.mean(np.diff(emission_axis))
                shift_px = median_shift_nm / em_step
                diag["shift_nm"] = median_shift_nm
                diag["shift_pixels"] = shift_px
                diag["shift_estimates_per_peak"] = {
                    chir: float(s) for chir, s in zip(reference_chiralities, shifts_nm)
                }
                corrected = apply_emission_shift(corrected, shift_px)
                diag["corrections_applied"].append("wavelength_shift")

        elif reference_mode in ("template", "designated"):
            if reference_data is None:
                raise ValueError(f"reference_mode='{reference_mode}' requires reference_data")

            # Estimate shift via cross-correlation at multiple chirality peaks
            shifts_px = []
            for chir in reference_chiralities:
                if chir not in CHIRALITY_CENTERS:
                    continue
                ex_expected, em_expected = CHIRALITY_CENTERS[chir]
                ex_idx = int(np.argmin(np.abs(excitation_axis - ex_expected)))

                # Extract emission profiles
                ref_profile = reference_data[:, ex_idx]
                new_profile = corrected[:, ex_idx]

                s = estimate_shift_xcorr(ref_profile, new_profile,
                                         max_shift_pixels=max_shift_pixels)
                shifts_px.append(s)

            if shifts_px:
                median_shift_px = float(np.median(shifts_px))
                em_step = np.mean(np.diff(emission_axis))
                diag["shift_pixels"] = median_shift_px
                diag["shift_nm"] = median_shift_px * em_step
                diag["shift_estimates_per_peak"] = {
                    chir: float(s) * em_step
                    for chir, s in zip(reference_chiralities, shifts_px)
                }
                corrected = apply_emission_shift(corrected, median_shift_px)
                diag["corrections_applied"].append("wavelength_shift")

    # ── Step 2: Baseline drift correction ─────────────────────────────────────
    if correct_baseline:
        baseline = estimate_baseline_drift(
            corrected, emission_axis,
            low_cutoff_nm=baseline_low_nm,
            high_cutoff_nm=baseline_high_nm,
        )
        diag["baseline_magnitude"] = float(np.mean(np.abs(baseline)))
        corrected = corrected - baseline
        diag["corrections_applied"].append("baseline_drift")

    # ── Step 3: Intensity scaling correction ──────────────────────────────────
    if correct_intensity and reference_data is not None:
        scale = estimate_intensity_scale(
            corrected, emission_axis, excitation_axis,
            reference_data, reference_chiralities,
        )
        if scale > 0.1:  # safety bound
            diag["intensity_scale"] = scale
            corrected = corrected / scale
            diag["corrections_applied"].append("intensity_scaling")

    return corrected, diag


# ─── Batch processing ────────────────────────────────────────────────────────

def build_template_from_dir(template_dir: str,
                            max_files: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build mean EEM template from a directory of training EEM files.
    Returns (mean_data, emission_axis, excitation_axis).
    """
    template_dir = Path(template_dir)
    files = sorted(template_dir.glob("*.xlsx"))[:max_files]

    if not files:
        raise FileNotFoundError(f"No .xlsx files in {template_dir}")

    mats = []
    em_axis = None
    ex_cols = None

    for f in files:
        data, em, ex = load_eem_with_axes(str(f))
        if em_axis is None:
            em_axis = em
            ex_cols = ex
        mats.append(data)

    # Average
    mean_data = np.mean(np.stack(mats), axis=0)
    return mean_data, em_axis, ex_cols


def main():
    ap = argparse.ArgumentParser(
        description="Reference-based EEM calibration before proxy latent extraction."
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing EEM .xlsx files to calibrate.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory for calibrated output files.")

    # Reference mode
    ap.add_argument("--reference_mode", required=True,
                    choices=["template", "designated", "self_peaks"],
                    help="How to establish the reference for shift estimation.")
    ap.add_argument("--template_dir", default=None,
                    help="Directory of training EEM files (for --reference_mode template).")
    ap.add_argument("--designated_ref_train", default=None,
                    help="Path to training-batch reference .xlsx (for --reference_mode designated).")
    ap.add_argument("--designated_ref_new", default=None,
                    help="Path to new-batch reference .xlsx (for --reference_mode designated). "
                         "If provided, shift is estimated once from this file and applied to all.")

    # Which chiralities to use as references
    ap.add_argument("--reference_peaks", nargs="+", default=["6.5", "8.3", "10.2"],
                    help="Chirality labels to use for shift estimation. "
                         "Choose strong, well-isolated peaks. Default: 6.5 8.3 10.2")

    # Correction toggles
    ap.add_argument("--no_shift_correction", action="store_true",
                    help="Skip wavelength shift correction.")
    ap.add_argument("--no_baseline_correction", action="store_true",
                    help="Skip baseline drift correction.")
    ap.add_argument("--no_intensity_correction", action="store_true",
                    help="Skip intensity scaling correction.")

    # Baseline parameters
    ap.add_argument("--baseline_low_nm", type=float, default=910.0,
                    help="Low-emission cutoff for baseline estimation (nm).")
    ap.add_argument("--baseline_high_nm", type=float, default=1340.0,
                    help="High-emission cutoff for baseline estimation (nm).")

    # Safety
    ap.add_argument("--max_shift_pixels", type=int, default=20,
                    help="Maximum allowed shift in pixels.")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate chiralities
    for chir in args.reference_peaks:
        if chir not in CHIRALITY_CENTERS:
            raise ValueError(f"Unknown chirality '{chir}'. Available: {list(CHIRALITY_CENTERS.keys())}")

    # ── Build reference ───────────────────────────────────────────────────────
    reference_data = None
    ref_em_axis = None
    ref_ex_axis = None

    if args.reference_mode == "template":
        if not args.template_dir:
            raise ValueError("--template_dir required for reference_mode='template'")
        print(f"Building template from {args.template_dir}...")
        reference_data, ref_em_axis, ref_ex_axis = build_template_from_dir(args.template_dir)
        print(f"  Template shape: {reference_data.shape}")

    elif args.reference_mode == "designated":
        if not args.designated_ref_train:
            raise ValueError("--designated_ref_train required for reference_mode='designated'")
        reference_data, ref_em_axis, ref_ex_axis = load_eem_with_axes(args.designated_ref_train)

        # If a new-batch reference is provided, estimate the batch shift once
        batch_shift_px = None
        if args.designated_ref_new:
            new_ref_data, _, _ = load_eem_with_axes(args.designated_ref_new)
            shifts_px = []
            for chir in args.reference_peaks:
                ex_exp, _ = CHIRALITY_CENTERS[chir]
                ex_idx = int(np.argmin(np.abs(ref_ex_axis - ex_exp)))
                s = estimate_shift_xcorr(
                    reference_data[:, ex_idx], new_ref_data[:, ex_idx],
                    max_shift_pixels=args.max_shift_pixels
                )
                shifts_px.append(s)
            batch_shift_px = float(np.median(shifts_px))
            em_step = np.mean(np.diff(ref_em_axis))
            print(f"  Batch shift from designated reference: {batch_shift_px:.3f} px ({batch_shift_px * em_step:.2f} nm)")

    elif args.reference_mode == "self_peaks":
        # No reference data needed; uses intrinsic peak positions
        pass

    # ── Process all files ─────────────────────────────────────────────────────
    input_files = sorted(input_dir.glob("*.xlsx"))
    if not input_files:
        raise FileNotFoundError(f"No .xlsx files in {input_dir}")

    print(f"\nCalibrating {len(input_files)} files...")
    print(f"  Reference mode: {args.reference_mode}")
    print(f"  Reference peaks: {args.reference_peaks}")
    print(f"  Corrections: shift={not args.no_shift_correction}, "
          f"baseline={not args.no_baseline_correction}, "
          f"intensity={not args.no_intensity_correction}")

    report = {"reference_mode": args.reference_mode, "files": {}}

    for filepath in input_files:
        data, em_axis, ex_axis = load_eem_with_axes(str(filepath))

        # For designated mode with pre-estimated batch shift, apply it directly
        if args.reference_mode == "designated" and batch_shift_px is not None:
            corrected = data.copy()
            diag = {"corrections_applied": []}

            if not args.no_shift_correction:
                corrected = apply_emission_shift(corrected, batch_shift_px)
                em_step = np.mean(np.diff(em_axis))
                diag["shift_pixels"] = batch_shift_px
                diag["shift_nm"] = batch_shift_px * em_step
                diag["corrections_applied"].append("wavelength_shift")

            if not args.no_baseline_correction:
                baseline = estimate_baseline_drift(
                    corrected, em_axis,
                    low_cutoff_nm=args.baseline_low_nm,
                    high_cutoff_nm=args.baseline_high_nm,
                )
                diag["baseline_magnitude"] = float(np.mean(np.abs(baseline)))
                corrected = corrected - baseline
                diag["corrections_applied"].append("baseline_drift")

            if not args.no_intensity_correction and reference_data is not None:
                scale = estimate_intensity_scale(
                    corrected, em_axis, ex_axis,
                    reference_data, args.reference_peaks,
                )
                if scale > 0.1:
                    diag["intensity_scale"] = scale
                    corrected = corrected / scale
                    diag["corrections_applied"].append("intensity_scaling")
        else:
            # Per-sample estimation
            corrected, diag = calibrate_single_eem(
                data, em_axis, ex_axis,
                reference_data=reference_data,
                reference_chiralities=args.reference_peaks,
                reference_mode=args.reference_mode,
                correct_shift=not args.no_shift_correction,
                correct_baseline=not args.no_baseline_correction,
                correct_intensity=not args.no_intensity_correction,
                baseline_low_nm=args.baseline_low_nm,
                baseline_high_nm=args.baseline_high_nm,
                max_shift_pixels=args.max_shift_pixels,
            )

        # Save corrected EEM (reconstruct original column names)
        df_orig = pd.read_excel(str(filepath), engine="openpyxl")
        ex_cols = [c for c in df_orig.columns if c != df_orig.columns[0]]
        save_eem_with_axes(str(output_dir / filepath.name), corrected, em_axis, ex_cols)

        report["files"][filepath.name] = diag

    # ── Save calibration report ───────────────────────────────────────────────
    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────────
    shifts = [d.get("shift_nm", 0.0) for d in report["files"].values()]
    scales = [d.get("intensity_scale", 1.0) for d in report["files"].values()]
    baselines = [d.get("baseline_magnitude", 0.0) for d in report["files"].values()]

    print(f"\n{'='*60}")
    print(f"Calibration complete: {len(input_files)} files processed")
    print(f"  Shift (nm):    median={np.median(shifts):.3f}, range=[{np.min(shifts):.3f}, {np.max(shifts):.3f}]")
    print(f"  Intensity:     median={np.median(scales):.3f}, range=[{np.min(scales):.3f}, {np.max(scales):.3f}]")
    print(f"  Baseline mag:  median={np.median(baselines):.5f}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
