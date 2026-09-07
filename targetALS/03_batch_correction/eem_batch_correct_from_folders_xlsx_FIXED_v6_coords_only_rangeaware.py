# eem_batch_correct_from_folders_xlsx_FIXED_v4_coords_only_rangeaware.py
# Batch correction for EEM XLSX using PBS-derived multiplicative surface (P*=PBS, F*=FBS).
#
# v4 fixes the v3 failure ("Too few in-range peaks") by making the coordinates parser RANGE-AWARE:
# - Coordinates_DNA.txt is treated as COORDINATES ONLY (any heights/intensities ignored).
# - If a line contains extra numbers (e.g., chirality label 8,3 + wavelengths 920 570),
#   v4 selects the numeric pair that best fits the Batch-2 EEM grid ranges.
# - Supports keyword formats: Emission/Excitation and also Em/Ex abbreviations.
# - If still no peaks in-range, prints the parsed points and the EEM axis ranges to debug quickly.

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator


# -------------------------
# Robust EEM parsing
# -------------------------
def _find_emission_col(columns):
    for c in columns:
        if str(c).strip().lower() == "emission":
            return c
    for c in columns:
        if "emission" in str(c).strip().lower():
            return c
    raise ValueError("Could not find an Emission column. Expected a column named like 'Emission'.")


def _parse_excitation_nm(colname: str) -> float:
    s = str(colname)
    m = re.search(r"excitation[^0-9]*([\d\.]+)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # allow pure numeric headers like "570" or "570.0"
    m2 = re.fullmatch(r"\s*([\d\.]+)\s*", s)
    if m2:
        return float(m2.group(1))
    nums = re.findall(r"[\d\.]+", s)
    if not nums:
        raise ValueError(f"Could not parse excitation wavelength from column: {colname}")
    return float(nums[0])


def load_eem_matrix_xlsx(xlsx_path, sheet_name=None, sheet_index=0):
    if sheet_name is not None:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    else:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_index, engine="openpyxl")

    em_col = _find_emission_col(df.columns)

    # excitation columns: contains "Excitation" OR purely numeric header
    ex_cols = []
    for c in df.columns:
        cs = str(c).lower()
        if "excitation" in cs:
            ex_cols.append(c)
        else:
            if re.fullmatch(r"\s*[\d\.]+\s*", str(c)):
                ex_cols.append(c)

    if not ex_cols:
        raise ValueError(f"No excitation columns found in {xlsx_path}")

    em_axis = df[em_col].to_numpy(dtype=float)
    ex_axis = np.array([_parse_excitation_nm(c) for c in ex_cols], dtype=float)
    data = df[ex_cols].to_numpy(dtype=float)

    # Sort emission (rows)
    em_order = np.argsort(em_axis)
    em_axis = em_axis[em_order]
    data = data[em_order, :]

    # Sort excitation (cols)
    ex_order = np.argsort(ex_axis)
    ex_axis = ex_axis[ex_order]
    ex_cols_sorted = [ex_cols[i] for i in ex_order]
    data = data[:, ex_order]

    return em_axis, ex_axis, data, df, em_col, ex_cols_sorted


def get_average_matrix_xlsx(file_list, sheet_name=None, sheet_index=0):
    stack = []
    em_ref = ex_ref = None

    print(f"Averaging {len(file_list)} replicates...")
    for f in file_list:
        em, ex, data, *_ = load_eem_matrix_xlsx(f, sheet_name=sheet_name, sheet_index=sheet_index)

        if em_ref is None:
            em_ref, ex_ref = em, ex
        else:
            if em.shape != em_ref.shape or ex.shape != ex_ref.shape or \
               not np.allclose(em, em_ref) or not np.allclose(ex, ex_ref):
                raise ValueError(
                    "Replicates do not share identical emission/excitation grids.\n"
                    f"Offending file: {f}\n"
                    "Fix by exporting all XLSX on the same wavelength grid."
                )

        stack.append(data)

    return em_ref, ex_ref, np.mean(np.stack(stack, axis=0), axis=0)


# -------------------------
# Coordinates parser (coordinates ONLY, range-aware)
# -------------------------
_NUM = r"([0-9]+(?:\.[0-9]+)?)"


def _kw_value(line: str, key_regex: str):
    """
    Extract numeric value after a keyword (e.g., Emission/Excitation or Em/Ex).
    Returns float or None.
    """
    m = re.search(key_regex + r"[^0-9]*" + _NUM, line, flags=re.IGNORECASE)
    if not m:
        return None
    # key_regex may contain capturing groups; numeric value is the last captured group
    return float(m.group(m.lastindex))


def _score_pair(ex: float, em: float, ex_min: float, ex_max: float, em_min: float, em_max: float) -> float:
    """
    0 is best (inside ranges). Otherwise penalize distance outside range.
    """
    s = 0.0
    if ex < ex_min:
        s += (ex_min - ex)
    elif ex > ex_max:
        s += (ex - ex_max)
    if em < em_min:
        s += (em_min - em)
    elif em > em_max:
        s += (em - em_max)
    return s


def parse_peak_coords_coords_only_rangeaware(txt_path: str, ex_axis: np.ndarray, em_axis: np.ndarray):
    """
    Returns points as (Ex, Em) using ONLY coordinates; ignores any heights/intensities.
    If a line has many numbers, chooses the pair that best fits (ex_range, em_range).
    """
    text = Path(txt_path).read_text(errors="ignore")

    ex_min, ex_max = float(np.min(ex_axis)), float(np.max(ex_axis))
    em_min, em_max = float(np.min(em_axis)), float(np.max(em_axis))

    points = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # 1) Try keyword-based parse (Emission/Excitation or Em/Ex)
        em = _kw_value(line, r"\b(emission|em)\b")
        ex = _kw_value(line, r"\b(excitation|ex)\b")

        if (em is not None) and (ex is not None):
            points.append([ex, em])
            continue

        # 2) Otherwise: numeric tokens, choose best pair that fits grid
        nums = [float(x) for x in re.findall(_NUM, line)]
        if len(nums) < 2:
            continue

        best = None
        best_score = None

        # consider all pairs and both assignments
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]

                # assignment 1: ex=a, em=b
                s1 = _score_pair(a, b, ex_min, ex_max, em_min, em_max)
                # assignment 2: ex=b, em=a
                s2 = _score_pair(b, a, ex_min, ex_max, em_min, em_max)

                if best_score is None or s1 < best_score:
                    best_score = s1
                    best = (a, b)  # (ex, em)
                if s2 < best_score:
                    best_score = s2
                    best = (b, a)  # (ex, em)

        if best is not None:
            points.append([best[0], best[1]])  # (Ex, Em)

    if not points:
        raise ValueError(
            "No peak coordinates could be parsed from the coords file.\n"
            "Expected keyword lines like 'Emission = 920  Excitation = 570' OR lines containing numeric coordinates."
        )

    pts = np.array(points, dtype=float)

    # De-duplicate coordinates (optional but helps interpolation stability)
    pts = np.unique(pts, axis=0)

    return pts


def filter_points_to_grid(points_ex_em: np.ndarray, ex_axis: np.ndarray, em_axis: np.ndarray):
    ex_min, ex_max = float(np.min(ex_axis)), float(np.max(ex_axis))
    em_min, em_max = float(np.min(em_axis)), float(np.max(em_axis))

    keep = (
        (points_ex_em[:, 0] >= ex_min) & (points_ex_em[:, 0] <= ex_max) &
        (points_ex_em[:, 1] >= em_min) & (points_ex_em[:, 1] <= em_max)
    )
    kept = points_ex_em[keep]
    dropped = points_ex_em[~keep]
    return kept, dropped, (ex_min, ex_max, em_min, em_max)


def sample_matrix_at_points(em_axis, ex_axis, matrix, points_ex_em):
    interp = RegularGridInterpolator((em_axis, ex_axis), matrix, bounds_error=False, fill_value=None)
    pts_em_ex = np.column_stack([points_ex_em[:, 1], points_ex_em[:, 0]])
    return np.asarray(interp(pts_em_ex), dtype=float)


# -------------------------
# Core: robust surface (global scale + shape)
# -------------------------
def generate_robust_surface_xlsx(
    batch1_pbs_files, batch2_pbs_files, coords_path,
    sheet_name=None, sheet_index=0,
    eps=1e-6,
    separate_global_scale=True,
    clamp_shape=True,
    shape_min=0.5,
    shape_max=2.0,
    debug_out_dir=None
):
    em1, ex1, ref_mean = get_average_matrix_xlsx(batch1_pbs_files, sheet_name=sheet_name, sheet_index=sheet_index)
    em2, ex2, tgt_mean = get_average_matrix_xlsx(batch2_pbs_files, sheet_name=sheet_name, sheet_index=sheet_index)

    # Parse coordinates using the Batch-2 grid ranges (range-aware)
    points_all = parse_peak_coords_coords_only_rangeaware(coords_path, ex2, em2)
    points, dropped, ranges = filter_points_to_grid(points_all, ex2, em2)

    ex_min, ex_max, em_min, em_max = ranges
    if len(dropped) > 0:
        print(f"WARNING: Dropped {len(dropped)} peak coordinates outside the Batch-2 EEM grid.")

    if len(points) < 3:
        # Provide actionable debug
        preview = points_all[:min(10, len(points_all))]
        msg = (
            f"Too few in-range peaks after filtering: {len(points)} (need >=3).\n"
            f"Batch-2 grid ranges: Ex [{ex_min:.3f}, {ex_max:.3f}], Em [{em_min:.3f}, {em_max:.3f}]\n"
            f"Parsed coordinates preview (first {len(preview)}):\n{preview}\n"
            "This usually means Coordinates_DNA.txt contains values not in nm (e.g., chirality labels like 8,3), "
            "or the file uses a different wavelength domain than your current EEM grid.\n"
        )
        raise RuntimeError(msg)

    ref_vals = sample_matrix_at_points(em1, ex1, ref_mean, points)
    tgt_vals = sample_matrix_at_points(em2, ex2, tgt_mean, points)

    tgt_vals = np.where(np.abs(tgt_vals) < eps, eps, tgt_vals)
    ratios = ref_vals / tgt_vals
    ratios = np.where(np.isfinite(ratios), ratios, 1.0)

    log_ratios = np.log(np.maximum(ratios, eps))

    if separate_global_scale:
        global_shift = float(np.median(log_ratios))
        log_shape = log_ratios - global_shift
    else:
        global_shift = 0.0
        log_shape = log_ratios

    if clamp_shape:
        lo = np.log(max(shape_min, eps))
        hi = np.log(max(shape_max, eps))
        log_shape = np.clip(log_shape, lo, hi)

    grid_x, grid_y = np.meshgrid(ex2, em2)
    surface_log_shape = griddata(points, log_shape, (grid_x, grid_y), method="linear")

    mask = np.isnan(surface_log_shape)
    if np.any(mask):
        surface_log_shape[mask] = griddata(points, log_shape, (grid_x[mask], grid_y[mask]), method="nearest")

    surface = np.exp(surface_log_shape + global_shift)

    if debug_out_dir is not None:
        debug_out_dir = Path(debug_out_dir)
        debug_out_dir.mkdir(parents=True, exist_ok=True)

        dbg = pd.DataFrame({
            "Ex": points[:, 0],
            "Em": points[:, 1],
            "ref_val": ref_vals,
            "tgt_val": tgt_vals,
            "ratio": ratios,
            "global_scale": np.exp(global_shift),
            "shape_used": np.exp(log_shape),
        })
        dbg.to_csv(debug_out_dir / "debug_peak_ratios.csv", index=False)

        parsed_df = pd.DataFrame(points_all, columns=["Ex", "Em"])
        parsed_df.to_csv(debug_out_dir / "debug_parsed_coordinates.csv", index=False)

        stats = {
            "ratios_min": float(np.min(ratios)),
            "ratios_med": float(np.median(ratios)),
            "ratios_max": float(np.max(ratios)),
            "global_scale": float(np.exp(global_shift)),
            "shape_used_min": float(np.min(np.exp(log_shape))),
            "shape_used_max": float(np.max(np.exp(log_shape))),
            "n_peaks_parsed": int(len(points_all)),
            "n_peaks_used": int(len(points)),
            "ex_min": ex_min,
            "ex_max": ex_max,
            "em_min": em_min,
            "em_max": em_max,
        }
        (debug_out_dir / "debug_stats.txt").write_text("\n".join([f"{k}={v}" for k, v in stats.items()]))

    return em2, ex2, surface


# -------------------------
# Folder discovery + matching
# -------------------------
def classify_prefix(path: Path) -> str:
    name = path.name.strip()
    if not name:
        return "OTHER"
    first = name[0].upper()
    if first == "P":
        return "PBS"
    if first == "F":
        return "FBS"
    return "OTHER"


def canonical_sample_id(path: Path) -> str:
    s = path.name
    s = re.sub(r"\.xlsx$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*Sheet\d+\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()

    if s and s[0].upper() in ("P", "F"):
        s = s[1:]

    s = re.sub(r"[^A-Za-z0-9]+", "", s).lower()
    return s


def collect_xlsx_case_insensitive(root: str, recursive=True):
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")

    it = rootp.rglob("*") if recursive else rootp.glob("*")
    files = []
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("~$"):
            continue
        if p.suffix.lower() == ".xlsx":
            files.append(p)
    return sorted(files)


def build_index(files):
    idx = {"PBS": {}, "FBS": {}, "OTHER": {}}
    for p in files:
        cls = classify_prefix(p)
        sid = canonical_sample_id(p)
        idx[cls].setdefault(sid, []).append(p)
    return idx


def matched_file_lists(idx1, idx2, cls):
    ids1 = set(idx1[cls].keys())
    ids2 = set(idx2[cls].keys())
    common = sorted(ids1.intersection(ids2))
    b1 = [str(p) for sid in common for p in idx1[cls][sid]]
    b2 = [str(p) for sid in common for p in idx2[cls][sid]]
    return b1, b2, common


def output_path_for_file(batch2_root: str, f: Path, out_root: str, cls: str, mirror=True):
    out_root = Path(out_root)
    if mirror:
        rel = f.relative_to(Path(batch2_root))
        out_dir = out_root / cls / rel.parent
    else:
        out_dir = out_root / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (f.stem + "_corrected.xlsx")


# -------------------------
# Apply correction
# -------------------------
def apply_correction_xlsx(target_xlsx, correction_surface, output_xlsx, sheet_name=None, sheet_index=0):
    em, ex, data, df, em_col, ex_cols_sorted = load_eem_matrix_xlsx(
        target_xlsx, sheet_name=sheet_name, sheet_index=sheet_index
    )

    if correction_surface.shape != data.shape:
        raise ValueError(
            f"Surface shape {correction_surface.shape} does not match data shape {data.shape}\n"
            f"File: {target_xlsx}\n"
            "Usually means Batch-2 sample grid differs from Batch-2 PBS grid."
        )

    corrected = data * correction_surface

    df_sorted = df.sort_values(by=em_col, ascending=True).reset_index(drop=True)
    df_sorted[ex_cols_sorted] = corrected
    Path(output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_excel(output_xlsx, index=False, engine="openpyxl")
    print(f"  -> Saved: {output_xlsx}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Batch correction for EEM XLSX using PBS-derived multiplicative surface (P*=PBS, F*=FBS). "
                    "Coordinates file is treated as coordinates ONLY; extra numeric fields are ignored."
    )
    ap.add_argument("--batch1_root", required=True)
    ap.add_argument("--batch2_root", required=True)
    ap.add_argument("--coords_file", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--no_mirror", action="store_true")
    ap.add_argument("--include_other", action="store_true")

    sheet = ap.add_mutually_exclusive_group()
    sheet.add_argument("--sheet_name", default=None)
    sheet.add_argument("--sheet_index", type=int, default=0)

    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--no_global_scale", action="store_true",
                    help="Do NOT separate global brightness scaling from shape.")
    ap.add_argument("--no_shape_clamp", action="store_true",
                    help="Do NOT clamp shape residual.")
    ap.add_argument("--shape_min", type=float, default=0.5)
    ap.add_argument("--shape_max", type=float, default=2.0)
    ap.add_argument("--debug_peaks", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror = not args.no_mirror

    b1_files = collect_xlsx_case_insensitive(args.batch1_root, recursive=args.recursive)
    b2_files = collect_xlsx_case_insensitive(args.batch2_root, recursive=args.recursive)

    print(f"Batch1: found {len(b1_files)} XLSX under {args.batch1_root}")
    print(f"Batch2: found {len(b2_files)} XLSX under {args.batch2_root}")

    idx1 = build_index(b1_files)
    idx2 = build_index(b2_files)

    b1_pbs, b2_pbs, pbs_ids = matched_file_lists(idx1, idx2, "PBS")
    print(f"Matched PBS sample IDs: {len(pbs_ids)}")
    if len(b1_pbs) == 0 or len(b2_pbs) == 0:
        raise RuntimeError("No matched PBS (P*) files found across Batch1 and Batch2.")

    print("\n--- Generating correction surface from matched PBS means (coords-only v4, range-aware) ---")
    em2, ex2, surface = generate_robust_surface_xlsx(
        b1_pbs, b2_pbs, args.coords_file,
        sheet_name=args.sheet_name,
        sheet_index=args.sheet_index,
        eps=args.eps,
        separate_global_scale=(not args.no_global_scale),
        clamp_shape=(not args.no_shape_clamp),
        shape_min=args.shape_min,
        shape_max=args.shape_max,
        debug_out_dir=(out_dir if args.debug_peaks else None)
    )

    np.save(out_dir / "correction_surface.npy", surface)
    surface_df = pd.DataFrame(surface, columns=[f"Excitation {x:g}" for x in ex2])
    surface_df.insert(0, "Emission", em2)
    surface_df.to_csv(out_dir / "correction_surface.csv", index=False)
    print(f"Saved surface to: {out_dir / 'correction_surface.npy'} and correction_surface.csv")

    print("\n--- Applying correction to Batch2 ---")
    classes_to_correct = ["PBS", "FBS"]
    if args.include_other:
        classes_to_correct.append("OTHER")

    for cls in classes_to_correct:
        for _, paths in idx2[cls].items():
            for p in paths:
                out_xlsx = output_path_for_file(args.batch2_root, p, args.output_dir, cls, mirror=mirror)
                apply_correction_xlsx(
                    str(p), surface, str(out_xlsx),
                    sheet_name=args.sheet_name,
                    sheet_index=args.sheet_index
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
