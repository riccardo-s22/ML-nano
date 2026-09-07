#!/usr/bin/env python3
"""
Test whether chirality positions have significantly higher importance scores
than neighboring regions in the autoencoder importance maps.

For each importance map (model × dim × timepoint):
  - 7×7 patch at each chirality position → mean importance
  - Surrounding annulus (ring) → mean importance
  - Enrichment = patch / ring

Statistical tests:
  - Paired Wilcoxon: chirality patch vs ring (paired by position)
  - One-sample t-test: enrichment ratios > 1
  - Permutation test: shuffle chirality vs non-chirality labels

USAGE:
------
python chirality_importance_test.py \
    --step3_dir ./step3_rois_3 \
    --step1_dir ./step1_latent_features \
    --tp_dirs ./out_0h,./out_6h,./out_24h \
    --output_dir ./chirality_importance_test
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from openpyxl import load_workbook
from scipy import stats
import re
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Chirality positions (emission_nm, excitation_nm)
# ============================================================================

CHIRALITY_POSITIONS = {
    "ch8_3":  (973.98,  673.94),
    "ch6_5":  (987.82,  577.12),
    "ch7_5":  (1047.81, 653.32),
    "ch10_2": (1080.60, 745.92),
    "ch9_4":  (1131.96, 731.39),
    "ch8_4":  (1130.34, 599.78),
    "ch7_6":  (1138.19, 659.79),
    "ch8_6":  (1200.03, 727.40),
    "ch8_7":  (1288.27, 740.87),
    "ch9_5":  (1262.98, 685.15),
    "ch10_3": (1267.70, 648.97),
    "ch10_5": (1282.97, 801.23),
}

TP_NAMES = ["0h", "6h", "24h"]
PATCH_RADIUS = 3   # 7×7
RING_WIDTH = 3     # annulus around patch


def load_spectral_axes(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
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
    emission = [float(row[0]) for row in rows[1:] if row[0] is not None]
    return np.array(emission), np.array(excitation)


def nm_to_index(axis, target):
    return int(np.argmin(np.abs(axis - target)))


def patch_mean(imp_map, row_c, col_c, radius):
    H, W = imp_map.shape
    r0, r1 = max(0, row_c - radius), min(H, row_c + radius + 1)
    c0, c1 = max(0, col_c - radius), min(W, col_c + radius + 1)
    return float(imp_map[r0:r1, c0:c1].mean())


def ring_mean(imp_map, row_c, col_c, inner_r, ring_w):
    H, W = imp_map.shape
    outer_r = inner_r + ring_w
    # outer patch
    r0o, r1o = max(0, row_c - outer_r), min(H, row_c + outer_r + 1)
    c0o, c1o = max(0, col_c - outer_r), min(W, col_c + outer_r + 1)
    # inner patch
    r0i, r1i = max(0, row_c - inner_r), min(H, row_c + inner_r + 1)
    c0i, c1i = max(0, col_c - inner_r), min(W, col_c + inner_r + 1)

    outer_mask = np.zeros((H, W), dtype=bool)
    outer_mask[r0o:r1o, c0o:c1o] = True
    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[r0i:r1i, c0i:c1i] = True
    ring = outer_mask & ~inner_mask

    vals = imp_map[ring]
    return float(vals.mean()) if vals.size > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step3_dir", type=str, required=True)
    parser.add_argument("--step1_dir", type=str, required=True)
    parser.add_argument("--tp_dirs", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./chirality_importance_test")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    step3_dir = Path(args.step3_dir)
    step1_dir = Path(args.step1_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]

    # Load spectral axes
    first_file = next(Path(tp_dirs[0]).glob("*.xlsx"))
    emission_axis, excitation_axis = load_spectral_axes(str(first_file))
    H, W = len(emission_axis), len(excitation_axis)
    print(f"Grid: {H} x {W}  |  Emission: {emission_axis[0]:.0f}-{emission_axis[-1]:.0f} nm  |  "
          f"Excitation: {excitation_axis[0]:.0f}-{excitation_axis[-1]:.0f} nm")

    # Map chirality → pixel
    chir_px = {}
    for name, (em, exc) in CHIRALITY_POSITIONS.items():
        chir_px[name] = (nm_to_index(emission_axis, em), nm_to_index(excitation_axis, exc))

    # Discover models
    model_dirs = sorted([
        d for d in step3_dir.iterdir()
        if d.is_dir() and d.name.startswith("best_model_fold")
    ])
    print(f"Models: {len(model_dirs)}")

    # =========================================================================
    # Collect enrichment data from every importance map
    # =========================================================================
    rows = []

    for mdir in model_dirs:
        fold = mdir.name
        imp_csv = step1_dir / fold / "dim_importance.csv"
        if not imp_csv.exists():
            imp_csv = step1_dir / "ensemble_dim_importance.csv"
        top_dims = pd.read_csv(imp_csv).head(args.top_k)["dim"].astype(int).tolist()

        for d in top_dims:
            for tp in TP_NAMES:
                imp_path = mdir / f"importance_dim{d}_{tp}.npy"
                if not imp_path.exists():
                    continue
                imp_map = np.load(imp_path)

                for chir_name, (rc, cc) in chir_px.items():
                    p = patch_mean(imp_map, rc, cc, PATCH_RADIUS)
                    r = ring_mean(imp_map, rc, cc, PATCH_RADIUS, RING_WIDTH)
                    enrichment = p / r if r > 1e-12 else np.nan
                    rows.append({
                        "model": fold, "dim": d, "timepoint": tp,
                        "chirality": chir_name,
                        "patch_importance": p,
                        "ring_importance": r,
                        "enrichment": enrichment,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "chirality_vs_ring_detail.csv", index=False)
    n_maps = df.groupby(["model", "dim", "timepoint"]).ngroups
    print(f"\nAnalyzed {n_maps} importance maps × {len(CHIRALITY_POSITIONS)} chiralities = {len(df)} comparisons")

    # =========================================================================
    # Statistical tests
    # =========================================================================
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS: Chirality patch vs surrounding ring")
    print(f"{'='*70}")

    # --- 1. Overall ---
    print(f"\n--- OVERALL (all maps, all timepoints) ---")
    _run_tests(df, outdir, "overall")

    # --- 2. Per timepoint ---
    for tp in TP_NAMES:
        print(f"\n--- Timepoint: {tp} ---")
        _run_tests(df[df["timepoint"] == tp], outdir, f"tp_{tp}")

    # --- 3. Per chirality position ---
    print(f"\n{'='*70}")
    print("PER-CHIRALITY ENRICHMENT")
    print(f"{'='*70}")
    print(f"{'Position':>10s}  {'Em(nm)':>7s}  {'Exc(nm)':>7s}  "
          f"{'Patch':>7s}  {'Ring':>7s}  {'Enrich':>7s}  {'p(Wilc)':>9s}  {'Sig':>4s}")
    print("-" * 85)

    chir_summary_rows = []
    for chir_name in sorted(CHIRALITY_POSITIONS.keys()):
        sub = df[df["chirality"] == chir_name]
        em, exc = CHIRALITY_POSITIONS[chir_name]
        p_mean = sub["patch_importance"].mean()
        r_mean = sub["ring_importance"].mean()
        e_mean = sub["enrichment"].mean()

        # Paired Wilcoxon per chirality
        try:
            stat, pval = stats.wilcoxon(sub["patch_importance"], sub["ring_importance"],
                                        alternative="greater")
        except Exception:
            pval = 1.0

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"{chir_name:>10s}  {em:7.1f}  {exc:7.1f}  "
              f"{p_mean:7.4f}  {r_mean:7.4f}  {e_mean:7.3f}  {pval:9.2e}  {sig:>4s}")

        chir_summary_rows.append({
            "chirality": chir_name, "emission_nm": em, "excitation_nm": exc,
            "mean_patch": p_mean, "mean_ring": r_mean, "mean_enrichment": e_mean,
            "wilcoxon_p": pval, "significant": sig,
        })

    chir_summary = pd.DataFrame(chir_summary_rows).sort_values("mean_enrichment", ascending=False)
    chir_summary.to_csv(outdir / "chirality_enrichment_summary.csv", index=False)

    # --- 4. Per model (averaged across dims) ---
    print(f"\n{'='*70}")
    print("PER-MODEL ENRICHMENT (averaged across dims and chiralities)")
    print(f"{'='*70}")
    for mdir in model_dirs:
        fold = mdir.name
        sub = df[df["model"] == fold]
        for tp in TP_NAMES:
            tp_sub = sub[sub["timepoint"] == tp]
            if len(tp_sub) == 0:
                continue
            e = tp_sub["enrichment"].mean()
            p_mean = tp_sub["patch_importance"].mean()
            r_mean = tp_sub["ring_importance"].mean()
            print(f"  {fold} {tp}: patch={p_mean:.4f}  ring={r_mean:.4f}  enrichment={e:.3f}")

    # --- 5. Per dim (which dims show most chirality enrichment?) ---
    print(f"\n{'='*70}")
    print("PER-DIM ENRICHMENT (averaged across chiralities and models)")
    print(f"{'='*70}")
    dim_enrich = df.groupby("dim").agg(
        mean_enrichment=("enrichment", "mean"),
        mean_patch=("patch_importance", "mean"),
        mean_ring=("ring_importance", "mean"),
    ).sort_values("mean_enrichment", ascending=False)
    dim_enrich.to_csv(outdir / "chirality_enrichment_by_dim.csv")

    for d, row in dim_enrich.head(15).iterrows():
        print(f"  dim{d:3d}: enrichment={row['mean_enrichment']:.3f}  "
              f"patch={row['mean_patch']:.4f}  ring={row['mean_ring']:.4f}")

    print(f"\nAll outputs saved to: {outdir}")


def _run_tests(df_sub, outdir, label):
    """Run paired Wilcoxon, one-sample t-test on enrichment, and print results."""
    patch = df_sub["patch_importance"].values
    ring = df_sub["ring_importance"].values
    enrichment = df_sub["enrichment"].dropna().values

    mean_patch = patch.mean()
    mean_ring = ring.mean()
    mean_enrich = enrichment.mean()

    # Paired Wilcoxon signed-rank (one-sided: patch > ring)
    try:
        w_stat, w_p = stats.wilcoxon(patch, ring, alternative="greater")
    except Exception:
        w_stat, w_p = np.nan, 1.0

    # One-sample t-test: enrichment > 1
    try:
        t_stat, t_p_two = stats.ttest_1samp(enrichment, 1.0)
        t_p = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2  # one-sided
    except Exception:
        t_stat, t_p = np.nan, 1.0

    # Effect size: Cohen's d for paired comparison
    diff = patch - ring
    d_cohen = diff.mean() / diff.std() if diff.std() > 1e-12 else 0.0

    print(f"  N comparisons:       {len(patch)}")
    print(f"  Mean patch:          {mean_patch:.4f}")
    print(f"  Mean ring:           {mean_ring:.4f}")
    print(f"  Mean enrichment:     {mean_enrich:.3f}")
    print(f"  Paired Wilcoxon:     W={w_stat:.0f}, p={w_p:.2e}  "
          f"{'***' if w_p < 0.001 else '**' if w_p < 0.01 else '*' if w_p < 0.05 else 'ns'}")
    print(f"  Enrichment > 1:      t={t_stat:.2f}, p={t_p:.2e}  "
          f"{'***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'ns'}")
    print(f"  Cohen's d (paired):  {d_cohen:.3f}")


if __name__ == "__main__":
    main()
