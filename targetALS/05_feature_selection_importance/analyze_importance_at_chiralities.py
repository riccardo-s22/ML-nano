#!/usr/bin/env python3
"""
Analyze importance maps at chirality positions vs neighboring regions.

For each (model, dim, timepoint) importance map from Step 3:
  - Extract mean importance in a 7x7 patch centered on each chirality position
  - Extract mean importance in a surrounding annulus (ring) around the patch
  - Compute enrichment ratio = chirality_patch / neighbor_ring
  - Also compute: percentile rank of chirality patch within the full map

Outputs:
  chirality_importance_detail.csv   — per (model, dim, tp, chirality) rows
  chirality_importance_summary.csv  — aggregated across dims and models
  chirality_enrichment_heatmap.html — interactive visualization

USAGE:
------
python analyze_importance_at_chiralities.py \
    --step3_dir ./step3_rois_3 \
    --step1_dir ./step1_latent_features \
    --tp_dirs ./out_0h,./out_6h,./out_24h \
    --output_dir ./chirality_importance_analysis \
    --latent_dim 512 \
    --top_k 10 \
    --patch_radius 3 \
    --ring_width 3
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from openpyxl import load_workbook
import re

# ============================================================================
# Chirality positions
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

TP_NAMES = ["0h", "6h", "24h"]


# ============================================================================
# Spectral axis loading
# ============================================================================

def load_spectral_axes(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load emission and excitation axes from an EEM Excel file."""
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
    excitation = np.array(excitation)

    emission = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        emission.append(float(row[0]))
    emission = np.array(emission)

    return emission, excitation


def nm_to_index(axis: np.ndarray, target_nm: float) -> int:
    return int(np.argmin(np.abs(axis - target_nm)))


def find_eem_file(directory: str, code: str) -> str:
    """Find the first file matching a sample code."""
    d = Path(directory)
    for f in d.iterdir():
        if f.suffix.lower() in (".xlsx", ".xls") and not f.name.startswith("~"):
            return str(f)
    raise FileNotFoundError(f"No xlsx file found in {directory}")


# ============================================================================
# Patch and ring extraction
# ============================================================================

def get_patch_mask(H: int, W: int, row_c: int, col_c: int, radius: int) -> np.ndarray:
    """Boolean mask for a (2r+1)x(2r+1) patch, clipped to image bounds."""
    mask = np.zeros((H, W), dtype=bool)
    r0 = max(0, row_c - radius)
    r1 = min(H, row_c + radius + 1)
    c0 = max(0, col_c - radius)
    c1 = min(W, col_c + radius + 1)
    mask[r0:r1, c0:c1] = True
    return mask


def get_ring_mask(H: int, W: int, row_c: int, col_c: int,
                  inner_radius: int, ring_width: int) -> np.ndarray:
    """Boolean mask for an annulus: outer_patch minus inner_patch."""
    outer_r = inner_radius + ring_width
    outer = get_patch_mask(H, W, row_c, col_c, outer_r)
    inner = get_patch_mask(H, W, row_c, col_c, inner_radius)
    return outer & ~inner


# ============================================================================
# Main analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze importance maps at chirality positions vs neighbors"
    )
    parser.add_argument("--step3_dir", type=str, required=True)
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--tp_dirs", type=str, required=True,
                        help="Comma-separated: dir_0h,dir_6h,dir_24h")
    parser.add_argument("--output_dir", type=str, default="./chirality_importance_analysis")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--patch_radius", type=int, default=3,
                        help="Radius for chirality patch (3 = 7x7)")
    parser.add_argument("--ring_width", type=int, default=3,
                        help="Width of surrounding annulus for comparison")
    args = parser.parse_args()

    step3_dir = Path(args.step3_dir)
    step1_dir = Path(args.step1_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]
    assert len(tp_dirs) == 3

    # -------------------------------------------------------------------------
    # Load spectral axes
    # -------------------------------------------------------------------------
    first_file = find_eem_file(tp_dirs[0], "any")
    emission_axis, excitation_axis = load_spectral_axes(first_file)
    H = len(emission_axis)
    W = len(excitation_axis)
    print(f"Spectral grid: {H} emission x {W} excitation")
    print(f"Emission:   {emission_axis[0]:.1f} - {emission_axis[-1]:.1f} nm")
    print(f"Excitation: {excitation_axis[0]:.1f} - {excitation_axis[-1]:.1f} nm")

    # -------------------------------------------------------------------------
    # Map chirality positions to pixel coords
    # -------------------------------------------------------------------------
    chir_pixel_coords = {}
    print(f"\nChirality positions → pixel coordinates:")
    for name, pos in CHIRALITY_POSITIONS.items():
        row = nm_to_index(emission_axis, pos["emission_nm"])
        col = nm_to_index(excitation_axis, pos["excitation_nm"])
        chir_pixel_coords[name] = (row, col)
        em_actual = emission_axis[row]
        ex_actual = excitation_axis[col]
        print(f"  {name:8s}: ({pos['emission_nm']:7.1f}, {pos['excitation_nm']:6.1f}) nm "
              f"→ pixel ({row:3d}, {col:2d})  actual=({em_actual:.1f}, {ex_actual:.1f}) nm")

    # Precompute masks for each chirality
    patch_masks = {}
    ring_masks = {}
    for name, (row, col) in chir_pixel_coords.items():
        patch_masks[name] = get_patch_mask(H, W, row, col, args.patch_radius)
        ring_masks[name] = get_ring_mask(H, W, row, col, args.patch_radius, args.ring_width)

    # Combined mask: union of all chirality patches
    all_chir_mask = np.zeros((H, W), dtype=bool)
    for m in patch_masks.values():
        all_chir_mask |= m
    n_chir_pixels = int(all_chir_mask.sum())
    n_total = H * W
    print(f"\nAll chirality patches: {n_chir_pixels}/{n_total} pixels "
          f"({100 * n_chir_pixels / n_total:.2f}%)")

    # -------------------------------------------------------------------------
    # Discover models and their dims
    # -------------------------------------------------------------------------
    model_dirs = sorted([
        d for d in step3_dir.iterdir()
        if d.is_dir() and d.name.startswith("best_model_fold")
    ])
    print(f"\nFound {len(model_dirs)} model folders in {step3_dir}")

    # -------------------------------------------------------------------------
    # Analyze each importance map
    # -------------------------------------------------------------------------
    detail_rows = []

    for model_dir in model_dirs:
        fold_name = model_dir.name

        # Load top dims from Step 1
        imp_csv = step1_dir / fold_name / "dim_importance.csv"
        if not imp_csv.exists():
            imp_csv = step1_dir / "ensemble_dim_importance.csv"
        if not imp_csv.exists():
            print(f"  [WARN] No dim_importance.csv for {fold_name}, skipping")
            continue
        imp_df = pd.read_csv(imp_csv)
        top_dims = imp_df.head(args.top_k)["dim"].astype(int).tolist()

        print(f"\n{fold_name}: dims={top_dims[:5]}{'...' if len(top_dims) > 5 else ''}")

        for d in top_dims:
            for tp in TP_NAMES:
                imp_path = model_dir / f"importance_dim{d}_{tp}.npy"
                if not imp_path.exists():
                    continue
                imp_map = np.load(imp_path)

                # Global stats
                global_mean = float(imp_map.mean())
                global_std = float(imp_map.std())

                # All-chirality vs non-chirality
                chir_mean_all = float(imp_map[all_chir_mask].mean())
                non_chir_mean = float(imp_map[~all_chir_mask].mean())

                # Per chirality position
                for chir_name in CHIRALITY_POSITIONS:
                    pm = patch_masks[chir_name]
                    rm = ring_masks[chir_name]

                    patch_vals = imp_map[pm]
                    ring_vals = imp_map[rm]

                    patch_mean = float(patch_vals.mean())
                    patch_max = float(patch_vals.max())
                    ring_mean = float(ring_vals.mean()) if ring_vals.size > 0 else 0.0
                    ring_std = float(ring_vals.std()) if ring_vals.size > 0 else 1e-12

                    # Enrichment metrics
                    enrichment = patch_mean / ring_mean if ring_mean > 1e-12 else 0.0
                    z_score = (patch_mean - ring_mean) / ring_std if ring_std > 1e-12 else 0.0

                    # Percentile rank of patch mean within the full map
                    pctl = float(np.mean(imp_map.ravel() <= patch_mean) * 100)

                    # Is this chirality inside the ROI?
                    roi_path = model_dir / f"roi_mask_dim{d}_{tp}.npy"
                    in_roi = False
                    roi_overlap_frac = 0.0
                    if roi_path.exists():
                        roi_mask = np.load(roi_path).astype(bool)
                        row_c, col_c = chir_pixel_coords[chir_name]
                        in_roi = bool(roi_mask[row_c, col_c])
                        roi_overlap_frac = float((roi_mask & pm).sum() / pm.sum())

                    detail_rows.append({
                        "model": fold_name,
                        "dim": d,
                        "timepoint": tp,
                        "chirality": chir_name,
                        "em_nm": CHIRALITY_POSITIONS[chir_name]["emission_nm"],
                        "exc_nm": CHIRALITY_POSITIONS[chir_name]["excitation_nm"],
                        "patch_mean": patch_mean,
                        "patch_max": patch_max,
                        "ring_mean": ring_mean,
                        "enrichment": enrichment,
                        "z_score": z_score,
                        "percentile": pctl,
                        "in_roi": in_roi,
                        "roi_overlap_frac": roi_overlap_frac,
                        "global_mean": global_mean,
                        "chir_all_mean": chir_mean_all,
                        "non_chir_mean": non_chir_mean,
                    })

    detail_df = pd.DataFrame(detail_rows)
    print(f"\nTotal detail rows: {len(detail_df)}")

    # -------------------------------------------------------------------------
    # Save detail CSV
    # -------------------------------------------------------------------------
    detail_df.to_csv(outdir / "chirality_importance_detail.csv", index=False)
    print(f"Saved: chirality_importance_detail.csv")

    # -------------------------------------------------------------------------
    # Summary: average across dims within each (model, tp, chirality)
    # -------------------------------------------------------------------------
    summary_by_model_tp = detail_df.groupby(["model", "timepoint", "chirality"]).agg(
        mean_enrichment=("enrichment", "mean"),
        mean_z_score=("z_score", "mean"),
        mean_percentile=("percentile", "mean"),
        mean_patch=("patch_mean", "mean"),
        mean_ring=("ring_mean", "mean"),
        frac_in_roi=("in_roi", "mean"),
        mean_roi_overlap=("roi_overlap_frac", "mean"),
    ).reset_index()
    summary_by_model_tp.to_csv(outdir / "chirality_importance_by_model_tp.csv", index=False)
    print(f"Saved: chirality_importance_by_model_tp.csv")

    # -------------------------------------------------------------------------
    # Summary: average across models → (tp, chirality)
    # -------------------------------------------------------------------------
    summary_by_tp = detail_df.groupby(["timepoint", "chirality"]).agg(
        mean_enrichment=("enrichment", "mean"),
        std_enrichment=("enrichment", "std"),
        mean_z_score=("z_score", "mean"),
        mean_percentile=("percentile", "mean"),
        std_percentile=("percentile", "std"),
        mean_patch=("patch_mean", "mean"),
        mean_ring=("ring_mean", "mean"),
        frac_in_roi=("in_roi", "mean"),
        n_maps=("enrichment", "count"),
    ).reset_index()
    summary_by_tp.to_csv(outdir / "chirality_importance_by_tp.csv", index=False)
    print(f"Saved: chirality_importance_by_tp.csv")

    # -------------------------------------------------------------------------
    # Grand summary: average across everything → per chirality
    # -------------------------------------------------------------------------
    summary_chir = detail_df.groupby("chirality").agg(
        mean_enrichment=("enrichment", "mean"),
        mean_z_score=("z_score", "mean"),
        mean_percentile=("percentile", "mean"),
        frac_in_roi=("in_roi", "mean"),
    ).reset_index().sort_values("mean_enrichment", ascending=False)
    summary_chir.to_csv(outdir / "chirality_importance_grand.csv", index=False)
    print(f"Saved: chirality_importance_grand.csv")

    # -------------------------------------------------------------------------
    # Global: all chirality patches vs non-chirality
    # -------------------------------------------------------------------------
    global_df = detail_df.groupby(["model", "dim", "timepoint"]).first().reset_index()
    chir_vs_non = global_df[["model", "dim", "timepoint",
                              "chir_all_mean", "non_chir_mean", "global_mean"]].copy()
    chir_vs_non["enrichment_global"] = chir_vs_non["chir_all_mean"] / chir_vs_non["non_chir_mean"]

    print(f"\n{'='*70}")
    print("CHIRALITY vs NON-CHIRALITY IMPORTANCE (averaged across all maps)")
    print(f"{'='*70}")
    for tp in TP_NAMES:
        tp_data = chir_vs_non[chir_vs_non["timepoint"] == tp]
        c_mean = tp_data["chir_all_mean"].mean()
        nc_mean = tp_data["non_chir_mean"].mean()
        ratio = c_mean / nc_mean if nc_mean > 1e-12 else 0
        print(f"  {tp}: chirality={c_mean:.4f}, non-chirality={nc_mean:.4f}, "
              f"ratio={ratio:.3f}")

    # -------------------------------------------------------------------------
    # Per-chirality enrichment table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PER-CHIRALITY ENRICHMENT (patch/ring, averaged across models and dims)")
    print(f"{'='*70}")
    print(f"{'Chirality':>10s}  {'Em(nm)':>8s}  {'Exc(nm)':>8s}  ", end="")
    for tp in TP_NAMES:
        print(f"{'enrich_'+tp:>10s}  {'pctl_'+tp:>8s}  {'inROI_'+tp:>9s}  ", end="")
    print()
    print("-" * 120)

    for _, row in summary_chir.iterrows():
        chir = row["chirality"]
        pos = CHIRALITY_POSITIONS[chir]
        print(f"{chir:>10s}  {pos['emission_nm']:8.1f}  {pos['excitation_nm']:8.1f}  ", end="")
        for tp in TP_NAMES:
            tp_row = summary_by_tp[
                (summary_by_tp["chirality"] == chir) & (summary_by_tp["timepoint"] == tp)
            ]
            if len(tp_row) > 0:
                e = tp_row["mean_enrichment"].values[0]
                p = tp_row["mean_percentile"].values[0]
                r = tp_row["frac_in_roi"].values[0]
                print(f"{e:10.3f}  {p:8.1f}  {r:9.1%}  ", end="")
            else:
                print(f"{'N/A':>10s}  {'N/A':>8s}  {'N/A':>9s}  ", end="")
        print()

    # -------------------------------------------------------------------------
    # Per-dim enrichment (which dims care about chirality?)
    # -------------------------------------------------------------------------
    dim_enrich = detail_df.groupby(["dim", "timepoint"]).agg(
        mean_enrichment=("enrichment", "mean"),
        mean_percentile=("percentile", "mean"),
        frac_in_roi=("in_roi", "mean"),
    ).reset_index()
    dim_enrich.to_csv(outdir / "chirality_importance_by_dim.csv", index=False)

    print(f"\n{'='*70}")
    print("TOP DIMS BY CHIRALITY ENRICHMENT")
    print(f"{'='*70}")
    dim_grand = dim_enrich.groupby("dim")["mean_enrichment"].mean().sort_values(ascending=False)
    for d, e in dim_grand.head(15).items():
        pctl = dim_enrich[dim_enrich["dim"] == d]["mean_percentile"].mean()
        roi_frac = dim_enrich[dim_enrich["dim"] == d]["frac_in_roi"].mean()
        print(f"  dim{d:3d}: enrichment={e:.3f}, percentile={pctl:.1f}, ROI overlap={roi_frac:.1%}")

    print(f"\nAll outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
