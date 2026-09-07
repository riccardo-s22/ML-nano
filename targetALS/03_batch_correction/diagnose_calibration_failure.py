#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_calibration_failure.py

Run this to understand why the calibration script detected zero batch effects.

Checks:
  1. Are the files actually different? (raw pixel comparison)
  2. Do the chirality excitation wavelengths match the data's excitation axis?
  3. Are the emission profiles at chirality peaks strong enough for xcorr?
  4. What is the actual shift between original and surrogate at each chirality?
  5. What are the actual intensity scaling and baseline differences?
  6. Was the template built from the correct directory?

USAGE:
  python diagnose_calibration_failure.py \
      --original_dir path/to/exp5/out_0h \
      --surrogate_dir path/to/exp7/3/out_0h

  OR compare specific files:

  python diagnose_calibration_failure.py \
      --original_dir path/to/exp5/out_0h \
      --surrogate_dir path/to/exp7/3/out_0h \
      --original_file 1.3d__570.opj_with_emission.xlsx \
      --surrogate_file 1_surrogate_03_3d__570_opj_with_emission.xlsx
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import correlate


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


def load_eem(filepath):
    """Load EEM Excel file."""
    df = pd.read_excel(filepath, engine="openpyxl")
    em_col = df.columns[0]
    emission_axis = df[em_col].values.astype(float)
    excitation_cols = [c for c in df.columns if c != em_col]
    ex_values = []
    for c in excitation_cols:
        s = str(c).replace("Excitation_", "").strip()
        try:
            ex_values.append(float(s))
        except ValueError:
            ex_values.append(np.nan)
    excitation_axis = np.array(ex_values)
    data = df[excitation_cols].values.astype(float)
    return data, emission_axis, excitation_axis, excitation_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", required=True)
    ap.add_argument("--surrogate_dir", required=True)
    ap.add_argument("--original_file", default=None)
    ap.add_argument("--surrogate_file", default=None)
    ap.add_argument("--chiralities", nargs="+", default=["6.5", "8.3", "10.2"])
    args = ap.parse_args()

    orig_dir = Path(args.original_dir)
    surr_dir = Path(args.surrogate_dir)

    # ── Find files ───────────────────────────────────────────────────────────
    if args.original_file and args.surrogate_file:
        orig_path = orig_dir / args.original_file
        surr_path = surr_dir / args.surrogate_file
    else:
        orig_files = sorted(orig_dir.glob("*.xlsx"))
        surr_files = sorted(surr_dir.glob("*.xlsx"))
        if not orig_files or not surr_files:
            print("ERROR: No .xlsx files found"); return
        orig_path = orig_files[0]
        surr_path = surr_files[0]
        print(f"Auto-selected: {orig_path.name} vs {surr_path.name}")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 1: FILE COMPARISON")
    print(f"{'='*70}")

    orig_data, orig_em, orig_ex, orig_cols = load_eem(str(orig_path))
    surr_data, surr_em, surr_ex, surr_cols = load_eem(str(surr_path))

    print(f"\n  Original:  {orig_data.shape}, em=[{orig_em[0]:.1f},{orig_em[-1]:.1f}], "
          f"ex=[{orig_ex[0]:.1f},{orig_ex[-1]:.1f}], step={np.mean(np.diff(orig_em)):.3f}nm")
    print(f"  Surrogate: {surr_data.shape}, em=[{surr_em[0]:.1f},{surr_em[-1]:.1f}], "
          f"ex=[{surr_ex[0]:.1f},{surr_ex[-1]:.1f}], step={np.mean(np.diff(surr_em)):.3f}nm")
    print(f"  Col names (first 3): {orig_cols[:3]}")

    if orig_data.shape == surr_data.shape:
        diff = surr_data - orig_data
        identical = np.allclose(orig_data, surr_data, atol=1e-10)
        print(f"\n  Identical? {'YES ⚠️  — PROBLEM: no batch effects applied!' if identical else 'NO ✓'}")
        if not identical:
            print(f"  Max |diff|:    {np.max(np.abs(diff)):.6e}")
            print(f"  Mean |diff|:   {np.mean(np.abs(diff)):.6e}")
            print(f"  Relative RMS:  {np.sqrt(np.mean(diff**2))/np.std(orig_data):.4f}")

            # Estimate global scale + offset
            o_flat = orig_data.flatten()
            s_flat = surr_data.flatten()
            A = np.vstack([o_flat, np.ones_like(o_flat)]).T
            (scale, offset), _, _, _ = np.linalg.lstsq(A, s_flat, rcond=None)
            print(f"  Global fit: surr ≈ {scale:.6f} × orig + {offset:.4e}")
    else:
        print(f"\n  ⚠️  Shape mismatch!")

    # ── Chirality peak analysis ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 2: CHIRALITY PEAK ANALYSIS")
    print(f"{'='*70}")

    ex_axis_values = orig_ex[~np.isnan(orig_ex)]
    print(f"\n  Data excitation axis ({len(ex_axis_values)} values):")
    print(f"  {ex_axis_values}")

    for chir in args.chiralities:
        ex_exp, em_exp = CHIRALITY_CENTERS[chir]
        ex_idx = int(np.argmin(np.abs(orig_ex - ex_exp)))
        ex_actual = orig_ex[ex_idx]

        print(f"\n  ─── Chirality ({chir}): expected ex={ex_exp}nm, em={em_exp}nm ───")
        print(f"  Nearest excitation: {ex_actual:.1f}nm (Δ={abs(ex_actual-ex_exp):.1f}nm) at col index {ex_idx}")

        # Profiles
        orig_prof = orig_data[:, ex_idx]
        surr_prof = surr_data[:, ex_idx]

        # Signal check
        orig_snr = (np.max(orig_prof) - np.median(orig_prof)) / max(np.std(orig_prof), 1e-12)
        surr_snr = (np.max(surr_prof) - np.median(surr_prof)) / max(np.std(surr_prof), 1e-12)
        print(f"  Profile SNR:  orig={orig_snr:.1f}  surr={surr_snr:.1f}")

        # Peak positions
        em_lo, em_hi = em_exp - 30, em_exp + 30
        mask = (orig_em >= em_lo) & (orig_em <= em_hi)
        if np.any(mask):
            orig_peak_em = orig_em[mask][np.argmax(orig_prof[mask])]
            surr_peak_em = surr_em[mask][np.argmax(surr_prof[mask])]
            print(f"  Peak position: orig={orig_peak_em:.2f}nm  surr={surr_peak_em:.2f}nm  (Δ={surr_peak_em-orig_peak_em:.2f}nm)")

            # Intensity ratio
            orig_peak_val = np.max(orig_prof[mask])
            surr_peak_val = np.max(surr_prof[mask])
            if orig_peak_val > 1e-12:
                print(f"  Peak intensity: orig={orig_peak_val:.4e}  surr={surr_peak_val:.4e}  ratio={surr_peak_val/orig_peak_val:.4f}")

        # Cross-correlation
        ref = orig_prof - np.mean(orig_prof)
        new = surr_prof - np.mean(surr_prof)
        ref_norm = np.linalg.norm(ref)
        new_norm = np.linalg.norm(new)

        if ref_norm < 1e-12 or new_norm < 1e-12:
            print(f"  ⚠️  Profile norm too small — xcorr will fail!")
            continue

        ref /= ref_norm
        new /= new_norm

        xcorr = correlate(new, ref, mode='full')
        n = len(ref)
        lags = np.arange(-n + 1, n)
        valid = np.abs(lags) <= 20
        xcorr_v = xcorr[valid]
        lags_v = lags[valid]

        peak_idx = np.argmax(xcorr_v)
        peak_lag = lags_v[peak_idx]
        peak_corr = xcorr_v[peak_idx]

        # Sub-pixel refinement
        if 0 < peak_idx < len(xcorr_v) - 1:
            y_m, y_c, y_p = xcorr_v[peak_idx-1], xcorr_v[peak_idx], xcorr_v[peak_idx+1]
            denom = 2*(2*y_c - y_m - y_p)
            if abs(denom) > 1e-12:
                delta = (y_m - y_p) / denom
                refined = float(peak_lag) + delta
            else:
                refined = float(peak_lag)
        else:
            refined = float(peak_lag)

        em_step = np.mean(np.diff(orig_em))
        print(f"  Xcorr: lag={peak_lag}px  refined={refined:.3f}px  ({refined*em_step:.3f}nm)  corr={peak_corr:.6f}")

    # ── Template check ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 3: TEMPLATE CROSS-CHECK")
    print(f"{'='*70}")
    print(f"\n  CRITICAL QUESTION: What did you pass as --template_dir?")
    print(f"  If you passed the SURROGATE directory, the template is the mean")
    print(f"  of the batch-affected data, and calibration finds zero shift!")

    orig_files = sorted(orig_dir.glob("*.xlsx"))[:50]
    surr_files = sorted(surr_dir.glob("*.xlsx"))[:50]

    if len(orig_files) > 1 and len(surr_files) > 1:
        # Build both templates
        orig_mats = []
        for f in orig_files:
            try:
                d, _, _, _ = load_eem(str(f))
                orig_mats.append(d)
            except: pass

        surr_mats = []
        for f in surr_files:
            try:
                d, _, _, _ = load_eem(str(f))
                surr_mats.append(d)
            except: pass

        if orig_mats and surr_mats:
            orig_template = np.mean(np.stack(orig_mats), axis=0)
            surr_template = np.mean(np.stack(surr_mats), axis=0)

            diff_templates = surr_template - orig_template
            print(f"\n  Templates comparison:")
            print(f"    Original template (from {len(orig_mats)} files)")
            print(f"    Surrogate template (from {len(surr_mats)} files)")
            print(f"    Max |diff|:   {np.max(np.abs(diff_templates)):.6e}")
            print(f"    Mean |diff|:  {np.mean(np.abs(diff_templates)):.6e}")

            # Check each chirality shift between templates
            for chir in args.chiralities:
                ex_exp, em_exp = CHIRALITY_CENTERS[chir]
                ex_idx = int(np.argmin(np.abs(orig_ex - ex_exp)))

                ref_p = orig_template[:, ex_idx] - np.mean(orig_template[:, ex_idx])
                new_p = surr_template[:, ex_idx] - np.mean(surr_template[:, ex_idx])
                ref_n = np.linalg.norm(ref_p)
                new_n = np.linalg.norm(new_p)
                if ref_n > 1e-12 and new_n > 1e-12:
                    ref_p /= ref_n; new_p /= new_n
                    xc = correlate(new_p, ref_p, mode='full')
                    n = len(ref_p)
                    lg = np.arange(-n+1, n)
                    v = np.abs(lg) <= 20
                    pi = np.argmax(xc[v])
                    em_step = np.mean(np.diff(orig_em))
                    print(f"    Chirality ({chir}): template-to-template shift = {lg[v][pi]}px ({lg[v][pi]*em_step:.2f}nm)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("""
If files ARE different but calibration found zero:
  → Most likely --template_dir pointed at the wrong directory
  → Fix: use the ORIGINAL exp5 data directory as template

If xcorr detects shift here but calibration didn't:
  → Possible data loading format mismatch in calibration script
  → Check that calibrate_eem_to_reference.py loads the Emission column correctly

If files are IDENTICAL:
  → The surrogate generation didn't write to this directory
  → Or you're looking at the original data, not the surrogates
""")


if __name__ == "__main__":
    main()
