#!/usr/bin/env python
# force_correction_v6_auto_invert.py
#
# v6 improvements over v5:
# 1) COORDS FILE HANDLING (same as v5)
#    - If coords contains nm (Emission/Excitation or Em/Ex): snap to closest pixel on Batch-2 grid.
#    - If coords contains only chirality labels (e.g., 8,3): auto-detect peak anchors from Batch-1 PBS mean.
#
# 2) AUTO-INVERT SURFACE (new)
#    - After building a surface from anchor ratios, v6 tests BOTH directions:
#         surface       and   1/surface
#      by applying them to the Batch-2 PBS mean and measuring error vs Batch-1 PBS mean.
#      It automatically picks the one that BEST aligns Batch-2 mean to Batch-1 mean.
#
# 3) EPS HANDLING
#    - --eps still exists, but default is smaller (1e-18) for your ~1e-14 scale.
#
# Outputs:
#    correction_surface_used.csv/.npy
#    debug_anchor_points.csv, debug_stats.txt (when --debug)
#
# Example:
# python force_correction_v6_auto_invert.py ^
#   --batch1_pbs_folder "...exp5\\tech_controls\\0h" --batch1_prefix P ^
#   --batch2_pbs_folder "...exp7\\tech_controls\\0h\\03" --batch2_prefix P ^
#   --coords_file "...\Coordinates_DNA.txt" ^
#   --input_dir "...exp7\\tech_controls\\0h\\03" ^
#   --output_dir "...exp7\\0h_corrected_force_v6" ^
#   --recursive --debug
#
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Force batch correction surface (XLSX/CSV). Uses nm coords if present; "
                    "otherwise auto-detects anchor peaks from PBS mean. Auto-inverts surface if needed."
    )

    p.add_argument("--batch1_pbs_folder", required=True, help="Folder containing Batch 1 PBS controls (reference)")
    p.add_argument("--batch2_pbs_folder", required=True, help="Folder containing Batch 2 PBS controls (target)")
    p.add_argument("--coords_file", required=True, help="Path to Coordinates_DNA.txt (nm coords or chirality list)")

    p.add_argument("--batch1_prefix", default="", help="Prefix filter for Batch 1 PBS files (e.g. 'P'). '' = all.")
    p.add_argument("--batch2_prefix", default="", help="Prefix filter for Batch 2 PBS files (e.g. 'P'). '' = all.")

    p.add_argument("--input_dir", required=True, help="Root folder of Batch 2 data to correct")
    p.add_argument("--output_dir", required=True, help="Where to save corrected files")

    p.add_argument("--recursive", action="store_true", help="Recurse through input_dir to find files")
    p.add_argument("--debug", action="store_true", help="Write debug CSV/TXT files into output_dir")

    sheet = p.add_mutually_exclusive_group()
    sheet.add_argument("--sheet_name", default=None, help="Sheet name for XLSX")
    sheet.add_argument("--sheet_index", type=int, default=0, help="Sheet index for XLSX")

    # Auto-peak fallback controls
    p.add_argument("--auto_peaks_n", type=int, default=0,
                   help="Override number of auto peaks. 0 => infer from coords_file (chirality count) or 12.")
    p.add_argument("--auto_peak_min_em_sep", type=int, default=12,
                   help="Min separation (emission index) between auto peaks")
    p.add_argument("--auto_peak_min_ex_sep", type=int, default=6,
                   help="Min separation (excitation index) between auto peaks")

    p.add_argument("--eps", type=float, default=1e-18, help="Small epsilon to avoid divide-by-zero in ratios")

    return p.parse_args()


# -------------------------
# EEM I/O (sorted axes)
# -------------------------
def _find_emission_col(cols):
    for c in cols:
        if str(c).strip().lower() == "emission":
            return c
    for c in cols:
        if "emission" in str(c).strip().lower():
            return c
    raise ValueError("Could not find an 'Emission' column.")


def _parse_excitation_nm(colname: str) -> float:
    s = str(colname)
    m = re.search(r"excitation[^0-9]*([\d\.]+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.fullmatch(r"\s*([\d\.]+)\s*", s)
    if m2:
        return float(m2.group(1))
    nums = re.findall(r"[\d\.]+", s)
    if not nums:
        raise ValueError(f"Could not parse excitation wavelength from column '{colname}'")
    return float(nums[0])


def load_eem(file_path: str, sheet_name=None, sheet_index=0):
    if os.path.getsize(file_path) == 0:
        return None, None, None, None, None, None

    try:
        if file_path.lower().endswith(".xlsx"):
            if sheet_name is not None:
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_index, engine="openpyxl")
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Warning: Could not load {os.path.basename(file_path)}: {e}")
        return None, None, None, None, None, None

    try:
        em_col = _find_emission_col(df.columns)
        ex_cols = []
        for c in df.columns:
            cs = str(c).lower()
            if "excitation" in cs:
                ex_cols.append(c)
            else:
                if re.fullmatch(r"\s*[\d\.]+\s*", str(c)):
                    ex_cols.append(c)

        if not ex_cols:
            return None, None, None, None, None, None

        em_axis = df[em_col].to_numpy(dtype=float)
        ex_axis = np.array([_parse_excitation_nm(c) for c in ex_cols], dtype=float)
        data = df[ex_cols].to_numpy(dtype=float)

        # sort emission
        em_order = np.argsort(em_axis)
        em_axis = em_axis[em_order]
        data = data[em_order, :]

        # sort excitation
        ex_order = np.argsort(ex_axis)
        ex_axis = ex_axis[ex_order]
        ex_cols_sorted = [ex_cols[i] for i in ex_order]
        data = data[:, ex_order]

        df_sorted = df.sort_values(by=em_col, ascending=True).reset_index(drop=True)
        return em_axis, ex_axis, data, df_sorted, em_col, ex_cols_sorted
    except Exception as e:
        print(f"Warning: Could not parse EEM matrix from {os.path.basename(file_path)}: {e}")
        return None, None, None, None, None, None


def list_data_files(folder: str, prefix: str, recursive=False):
    valid_exts = (".xlsx", ".csv")
    prefix = (prefix or "").upper()

    paths = []
    folderp = Path(folder)
    it = folderp.rglob("*") if recursive else folderp.iterdir()

    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("~$"):
            continue
        if not p.name.lower().endswith(valid_exts):
            continue
        if prefix and (not p.name.upper().startswith(prefix)):
            continue
        paths.append(str(p))
    return sorted(paths)


def get_mean_matrix(folder_path: str, prefix: str, sheet_name=None, sheet_index=0, recursive=False):
    files = list_data_files(folder_path, prefix, recursive=recursive)
    if not files:
        raise ValueError(f"No data files found in {folder_path} with prefix '{prefix}'")

    print(f"Found {len(files)} PBS replicates in {os.path.basename(folder_path)} (Prefix='{prefix}')")
    stack = []
    em_ref = ex_ref = None

    for fp in files:
        em, ex, data, *_ = load_eem(fp, sheet_name=sheet_name, sheet_index=sheet_index)
        if data is None:
            continue
        if em_ref is None:
            em_ref, ex_ref = em, ex
        else:
            if em.shape != em_ref.shape or ex.shape != ex_ref.shape or \
               (not np.allclose(em, em_ref)) or (not np.allclose(ex, ex_ref)):
                raise ValueError(
                    "PBS replicates do not share identical grids.\n"
                    f"Offending file: {fp}\n"
                    "Re-export on a consistent wavelength grid."
                )
        stack.append(data)

    if not stack:
        raise ValueError("Could not load any valid PBS replicates.")
    return em_ref, ex_ref, np.mean(np.stack(stack, axis=0), axis=0), files


# -------------------------
# Coordinates parsing + snapping
# -------------------------
_NUM = r"([0-9]+(?:\.[0-9]+)?)"


def _kw_value(line: str, key_regex: str):
    m = re.search(key_regex + r"[^0-9]*" + _NUM, line, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(m.lastindex))


def parse_nm_coords_or_chiralities(coords_file: str):
    text = Path(coords_file).read_text(errors="ignore")
    nm_points = []
    chiralities = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        em = _kw_value(line, r"\b(emission|em)\b")
        ex = _kw_value(line, r"\b(excitation|ex)\b")
        if (em is not None) and (ex is not None):
            nm_points.append((float(ex), float(em)))
        for a, b in re.findall(r"\b(\d+)\s*,\s*(\d+)\b", line):
            chiralities.append((int(a), int(b)))

    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return dedup(nm_points), dedup(chiralities)


def snap_points_to_grid(points_ex_em, ex_axis, em_axis):
    snapped = []
    for ex, em in points_ex_em:
        c_idx = int(np.abs(ex_axis - ex).argmin())
        r_idx = int(np.abs(em_axis - em).argmin())
        snapped.append((float(ex_axis[c_idx]), float(em_axis[r_idx])))
    return np.unique(np.array(snapped, dtype=float), axis=0)


# -------------------------
# Auto-detect strongest peaks
# -------------------------
def auto_detect_anchor_peaks(matrix: np.ndarray, em_axis: np.ndarray, ex_axis: np.ndarray,
                             n_peaks: int, min_em_sep: int, min_ex_sep: int) -> np.ndarray:
    mat = np.array(matrix, dtype=float)
    mat[~np.isfinite(mat)] = -np.inf

    flat_idx = np.argsort(mat.ravel())[::-1]
    n_rows, n_cols = mat.shape
    selected = []

    def far_enough(r, c):
        for rr, cc in selected:
            if abs(r - rr) < min_em_sep and abs(c - cc) < min_ex_sep:
                return False
        return True

    for k in flat_idx:
        r = int(k // n_cols)
        c = int(k % n_cols)
        if not np.isfinite(mat[r, c]):
            continue
        if far_enough(r, c):
            selected.append((r, c))
            if len(selected) >= n_peaks:
                break

    if len(selected) < 3:
        raise RuntimeError(f"Auto-peak detection found only {len(selected)} peaks; need >=3.")
    pts = np.array([[float(ex_axis[c]), float(em_axis[r])] for (r, c) in selected], dtype=float)
    return np.unique(pts, axis=0)


# -------------------------
# Sampling ratios at points
# -------------------------
def sample_matrix_at_points(em_axis, ex_axis, matrix, points_ex_em):
    interp = RegularGridInterpolator((em_axis, ex_axis), matrix, bounds_error=False, fill_value=None)
    pts_em_ex = np.column_stack([points_ex_em[:, 1], points_ex_em[:, 0]])
    return np.asarray(interp(pts_em_ex), dtype=float)


def rel_rmse(ref: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(ref) & np.isfinite(x)
    refv = ref[m].ravel()
    xv = x[m].ravel()
    denom = np.mean(np.abs(refv)) + 1e-30
    return float(np.sqrt(np.mean((refv - xv) ** 2)) / denom)


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- 1) Load PBS means ---")
    try:
        print("Processing Batch 1 (Ref PBS mean)...")
        em_ref, ex_ref, ref_mean, b1_files = get_mean_matrix(
            args.batch1_pbs_folder, args.batch1_prefix,
            sheet_name=args.sheet_name, sheet_index=args.sheet_index,
            recursive=args.recursive
        )
        print("Processing Batch 2 (Target PBS mean)...")
        em_tgt, ex_tgt, tgt_mean, b2_files = get_mean_matrix(
            args.batch2_pbs_folder, args.batch2_prefix,
            sheet_name=args.sheet_name, sheet_index=args.sheet_index,
            recursive=args.recursive
        )
    except ValueError as e:
        print(f"\nSTOPPING: {e}")
        sys.exit(1)

    if ref_mean.shape != tgt_mean.shape or (not np.allclose(em_ref, em_tgt)) or (not np.allclose(ex_ref, ex_tgt)):
        print("WARNING: Batch1 and Batch2 PBS grids are not identical. Proceeding with interpolators (surface on Batch2 grid).")

    print("\n--- 2) Determine anchor coordinates (take closest / auto-peaks) ---")
    nm_points, chir = parse_nm_coords_or_chiralities(args.coords_file)

    if nm_points:
        anchors = snap_points_to_grid(nm_points, ex_tgt, em_tgt)
        anchor_mode = "coords_nm_snapped_to_batch2_grid"
        print(f"Parsed {len(nm_points)} nm points; snapped to {len(anchors)} unique grid points.")
    else:
        inferred = len(chir) if chir else 12
        n_anchors = args.auto_peaks_n if args.auto_peaks_n > 0 else inferred
        n_anchors = max(n_anchors, 8)
        anchors = auto_detect_anchor_peaks(
            ref_mean, em_ref, ex_ref,
            n_peaks=n_anchors,
            min_em_sep=args.auto_peak_min_em_sep,
            min_ex_sep=args.auto_peak_min_ex_sep
        )
        anchor_mode = "auto_peaks_from_batch1_ref_mean"
        print(f"No nm coords found. Detected {len(anchors)} anchor peaks from Batch1 PBS mean (inferred={inferred}).")

    if anchors.shape[0] < 3:
        raise RuntimeError(f"Too few anchor points ({anchors.shape[0]}). Need >=3.")

    ref_vals = sample_matrix_at_points(em_ref, ex_ref, ref_mean, anchors)
    tgt_vals = sample_matrix_at_points(em_tgt, ex_tgt, tgt_mean, anchors)
    tgt_safe = np.where(np.abs(tgt_vals) < args.eps, np.sign(tgt_vals) * args.eps + (tgt_vals == 0) * args.eps, tgt_vals)

    ratios = ref_vals / tgt_safe
    ratios = np.where(np.isfinite(ratios), ratios, 1.0)

    log_ratios = np.log(np.maximum(np.abs(ratios), args.eps)) * np.sign(ratios)  # keep sign if needed

    # Interpolate log-ratios on Batch-2 grid
    grid_x, grid_y = np.meshgrid(ex_tgt, em_tgt)
    surf_log = griddata(anchors, log_ratios, (grid_x, grid_y), method="linear")
    mask = np.isnan(surf_log)
    if np.any(mask):
        surf_log[mask] = griddata(anchors, log_ratios, (grid_x[mask], grid_y[mask]), method="nearest")

    surface = np.exp(surf_log)

    # -------------------------
    # AUTO-INVERT: choose surface direction that best maps Batch2 mean -> Batch1 mean
    # -------------------------
    rmse_before = rel_rmse(ref_mean, tgt_mean)
    rmse_after = rel_rmse(ref_mean, tgt_mean * surface)

    surface_inv = 1.0 / np.where(surface == 0, np.nan, surface)
    rmse_after_inv = rel_rmse(ref_mean, tgt_mean * surface_inv)

    chosen = "surface"
    if rmse_after_inv < rmse_after:
        surface = surface_inv
        chosen = "inverse_surface"

    print("\n--- 3) Surface QC (auto-invert) ---")
    print(f"RMSE vs Batch1 mean: before={rmse_before:.6g}  after={rmse_after:.6g}  after_inv={rmse_after_inv:.6g}")
    print(f"Chosen: {chosen}")

    surf_mean = float(np.nanmean(surface))
    surf_std = float(np.nanstd(surface))
    surf_min = float(np.nanmin(surface))
    surf_max = float(np.nanmax(surface))
    print(f"Surface stats: mean={surf_mean:.6g} std={surf_std:.6g} min={surf_min:.6g} max={surf_max:.6g}")

    # Save surface with axes
    surf_df = pd.DataFrame(surface, columns=[f"Excitation {x:g}" for x in ex_tgt])
    surf_df.insert(0, "Emission", em_tgt)
    surf_df.to_csv(out_dir / "correction_surface_used.csv", index=False)
    np.save(out_dir / "correction_surface_used.npy", surface)

    if args.debug:
        dbg = pd.DataFrame({
            "Ex": anchors[:, 0],
            "Em": anchors[:, 1],
            "ref_val": ref_vals,
            "tgt_val": tgt_vals,
            "ratio": ratios,
        })
        dbg.to_csv(out_dir / "debug_anchor_points.csv", index=False)
        pd.DataFrame({"nm_points_ex_em": [str(x) for x in nm_points]}).to_csv(out_dir / "debug_nm_points_raw.csv", index=False)
        pd.DataFrame({"chirality_pairs": [str(x) for x in chir]}).to_csv(out_dir / "debug_chiralities_raw.csv", index=False)

        (out_dir / "debug_stats.txt").write_text(
            "\n".join([
                f"anchor_mode={anchor_mode}",
                f"chosen_surface={chosen}",
                f"n_anchors={anchors.shape[0]}",
                f"rmse_before={rmse_before}",
                f"rmse_after={rmse_after}",
                f"rmse_after_inv={rmse_after_inv}",
                f"surface_mean={surf_mean}",
                f"surface_std={surf_std}",
                f"surface_min={surf_min}",
                f"surface_max={surf_max}",
                f"batch2_ex_min={float(np.min(ex_tgt))}",
                f"batch2_ex_max={float(np.max(ex_tgt))}",
                f"batch2_em_min={float(np.min(em_tgt))}",
                f"batch2_em_max={float(np.max(em_tgt))}",
                f"eps={args.eps}",
            ])
        )

    print("\n--- 4) Apply correction to Batch-2 dataset ---")
    valid_exts = (".xlsx", ".csv")
    input_root = Path(args.input_dir)

    it = input_root.rglob("*") if args.recursive else input_root.iterdir()

    count = 0
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("~$"):
            continue
        if not p.name.lower().endswith(valid_exts):
            continue
        if "correction_surface" in p.name.lower():
            continue

        em, ex, data, df_sorted, em_col, ex_cols_sorted = load_eem(
            str(p), sheet_name=args.sheet_name, sheet_index=args.sheet_index
        )
        if data is None:
            continue

        if data.shape != surface.shape or (not np.allclose(em, em_tgt)) or (not np.allclose(ex, ex_tgt)):
            raise ValueError(
                "Data grid mismatch between this file and the Batch-2 PBS grid used to build the surface.\n"
                f"File: {p}\n"
                f"Data shape: {data.shape}, Surface shape: {surface.shape}\n"
                "Fix by ensuring all Batch-2 files share a single grid, or build separate surfaces per grid."
            )

        corrected = data * surface
        df_sorted[ex_cols_sorted] = corrected

        rel = p.relative_to(input_root)
        save_path = (out_dir / rel).with_suffix(".xlsx")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_sorted.to_excel(save_path, index=False, engine="openpyxl")

        count += 1

    print(f"\nDone! {count} files corrected.")
    print(f"Output Directory: {args.output_dir}")


if __name__ == "__main__":
    main()
