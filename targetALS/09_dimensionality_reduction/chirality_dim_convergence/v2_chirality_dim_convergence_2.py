#!/usr/bin/env python3
"""
Chirality–Latent Dimension Convergence Analysis
================================================

Tests whether autoencoder latent dimensions that correlate with NFL
concentration also encode information from the SWCNT chirality positions
(ch8_7, ch9_4, ch9_5) that physically drive NFL-modulated fluorescence.

ANALYSES:
  1. Correlation of ALL 720 chirality descriptors with each target dim (N=39)
  2. Enrichment: NFL-top chirality features vs background (Mann-Whitney)
  3. Specificity: target dim vs 200 random dims (null distribution)
  4. Mediation: chirality → dim → NFL (Sobel test, N=20 ALS)
  5. Chirality group importance (mean |r| per SWCNT species)

USAGE:
  python chirality_dim_convergence.py \
      --z_agg          z_agg1.csv \
      --chirality_csv  chirality_full_descriptors.csv \
      --clinical_csv   merged_proxy_clinical.csv \
      --target_dims    35 \
      --model_label    "Model 1" \
      --output_dir     ./convergence_results

  # For multiple dims in Model 3:
  python chirality_dim_convergence.py \
      --z_agg          z_agg3.csv \
      --chirality_csv  chirality_full_descriptors.csv \
      --clinical_csv   merged_proxy_clinical.csv \
      --target_dims    477,350 \
      --model_label    "Model 3" \
      --output_dir     ./convergence_results_m3

INPUTS:
  --z_agg           CSV with columns: code, group, dim0..dim511
  --chirality_csv   CSV with columns: code, [group], ch*_*_*_*h (720 features)
  --clinical_csv    CSV with columns: code, nfl_conc, age, sex, ...
  --target_dims     Comma-separated dim indices to analyze (e.g. 35 or 477,350)
  --n_null_dims     Number of random dims for null distribution (default: 200)
  --model_label     Label for the model in plots (default: "Model")
  --output_dir      Output directory (default: ./convergence_results)

OUTPUTS:
  Per dimension:
    chirality_{dim}_correlations.csv   — All 720 features × dim correlations
    mediation_{dim}_nfl.csv            — Sobel mediation results
    chirality_group_{dim}.csv          — Mean |r| per chirality type

  Combined:
    convergence_comparison.png         — 3-row panel (enrichment, specificity, mediation)
    chirality_group_importance.png     — Group-level bar charts
    convergence_summary.txt            — Text summary of all results

Author: Generated for Riccardo's ALS EEM pipeline
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# =============================================================================
# Top 10 NFL-correlated chirality features (from Figure 1 analysis)
# =============================================================================
TOP_NFL_CHIRALITY = [
    'ch9_5_grad_center_24h', 'ch9_5_kurtosis_24h', 'ch8_7_cv_6h',
    'ch9_5_cv_24h', 'ch8_7_std_6h', 'ch8_7_range_6h',
    'ch9_4_skewness_6h', 'ch9_4_cv_6h', 'ch9_5_range_24h', 'ch8_7_iqr_6h',
]

NFL_CHIRALITY_SPECIES = ['ch8_7', 'ch9_4', 'ch9_5']


# =============================================================================
# UTILITIES
# =============================================================================

def fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    ranked = np.argsort(pvals)
    fdr_sorted = pvals[ranked] * n / (np.arange(n) + 1)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr = np.zeros(n)
    fdr[ranked] = np.minimum(fdr_sorted, 1.0)
    return fdr


def clean_code(code_series):
    """Standardize sample codes (replace . with _ for matching)."""
    return code_series.astype(str).str.replace('.', '_', regex=False)


# =============================================================================
# CORE ANALYSIS FUNCTION
# =============================================================================

def run_convergence_analysis(dim_name, dim_idx, z_df, ch_df, clin_df,
                              chir_cols, top_nfl, n_null_dims, outdir, model_label):
    """
    Run all convergence analyses for a single latent dimension.

    Returns dict with all results needed for comparison tables and plots.
    """
    results = {'dim': dim_name, 'dim_idx': dim_idx}

    print(f"\n{'#' * 70}")
    print(f"# CONVERGENCE: {model_label} — {dim_name}")
    print(f"{'#' * 70}")

    # --- Merge z_agg with chirality (N=39, all samples) ---
    merged = z_df[['code', 'code_clean', 'group', dim_name]].merge(
        ch_df, on='code_clean'
    )

    # --- Merge with clinical for NFL (N=20, ALS only) ---
    merged_nfl = merged.merge(
        clin_df[['code_clean', 'nfl_conc', 'age', 'sex']],
        on='code_clean', how='inner'
    )
    nfl_vals = merged_nfl['nfl_conc'].values.astype(float)

    N_all = len(merged)
    N_nfl = len(merged_nfl)
    print(f"  N(all samples) = {N_all}, N(ALS + NFL) = {N_nfl}")

    # Confirm dim-NFL correlation
    dim_nfl = merged_nfl[dim_name].values.astype(float)
    ok = np.isfinite(dim_nfl) & np.isfinite(nfl_vals)
    r_nfl, p_nfl = stats.pearsonr(dim_nfl[ok], nfl_vals[ok])
    rho_nfl, sp_nfl = stats.spearmanr(dim_nfl[ok], nfl_vals[ok])
    print(f"  {dim_name} vs NFL: r={r_nfl:.3f} (p={p_nfl:.4f}), "
          f"ρ={rho_nfl:.3f} (p={sp_nfl:.4f})")
    results['r_nfl'] = r_nfl
    results['p_nfl'] = p_nfl
    results['rho_nfl'] = rho_nfl

    # ================================================================
    # ANALYSIS 1: Correlations with all chirality features (N=39)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 1: Chirality correlations with {dim_name} (N={N_all})")
    print(f"{'=' * 60}")

    corr_rows = []
    for feat in chir_cols:
        v = merged[feat].values.astype(float)
        d = merged[dim_name].values.astype(float)
        ok = np.isfinite(v) & np.isfinite(d)
        if ok.sum() < 5:
            continue
        r_p, p_p = stats.pearsonr(d[ok], v[ok])
        r_s, p_s = stats.spearmanr(d[ok], v[ok])
        corr_rows.append({
            'feature': feat,
            'n': int(ok.sum()),
            'pearson_r': r_p,
            'pearson_p': p_p,
            'spearman_rho': r_s,
            'spearman_p': p_s,
            'abs_pearson_r': abs(r_p),
            'is_nfl_top': feat in top_nfl,
        })

    corr_df = pd.DataFrame(corr_rows).sort_values('abs_pearson_r', ascending=False).reset_index(drop=True)
    corr_df['pearson_fdr'] = fdr_bh(corr_df['pearson_p'].values)
    corr_df.to_csv(outdir / f'chirality_{dim_name}_correlations.csv', index=False)

    # Print top 15
    print(f"\n  Top 15 chirality features correlated with {dim_name}:")
    for i, row in corr_df.head(15).iterrows():
        star = " ★" if row['is_nfl_top'] else ""
        print(f"    {row['feature']:<32s}  r={row['pearson_r']:+.3f}  "
              f"p={row['pearson_p']:.4f}  FDR={row['pearson_fdr']:.3f}{star}")

    # Where do NFL-top features rank?
    print(f"\n  NFL-top features ranking among {len(corr_df)} chirality features:")
    for feat in top_nfl:
        idx = corr_df[corr_df['feature'] == feat].index
        if len(idx) > 0:
            rank = idx[0] + 1
            ar = corr_df.loc[idx[0], 'abs_pearson_r']
            r_val = corr_df.loc[idx[0], 'pearson_r']
            print(f"    {feat:<32s}  |r|={ar:.3f}  rank=#{rank}")

    results['corr_df'] = corr_df

    # ================================================================
    # ANALYSIS 2: Enrichment test
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 2: Enrichment Test")
    print(f"{'=' * 60}")

    nfl_abs = corr_df[corr_df['is_nfl_top']]['abs_pearson_r'].dropna().values
    other_abs = corr_df[~corr_df['is_nfl_top']]['abs_pearson_r'].dropna().values

    if len(nfl_abs) > 0 and len(other_abs) > 0:
        mw_stat, mw_p = stats.mannwhitneyu(nfl_abs, other_abs, alternative='greater')
    else:
        mw_stat, mw_p = np.nan, np.nan

    print(f"  NFL-top chirality: mean |r| = {nfl_abs.mean():.4f} (n={len(nfl_abs)})")
    print(f"  Other chirality:   mean |r| = {other_abs.mean():.4f} (n={len(other_abs)})")
    print(f"  Mann-Whitney U: p = {mw_p:.2e}")

    results['enrich_nfl_mean'] = float(nfl_abs.mean())
    results['enrich_other_mean'] = float(other_abs.mean())
    results['enrich_p'] = float(mw_p)
    results['nfl_abs'] = nfl_abs
    results['other_abs'] = other_abs

    # ================================================================
    # ANALYSIS 3: Specificity vs random dims
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 3: Specificity — {dim_name} vs {n_null_dims} random dims")
    print(f"{'=' * 60}")

    np.random.seed(SEED)
    random_dims = np.random.choice(
        [d for d in range(z_df.shape[1] - 3) if d != dim_idx],  # exclude code, group, code_clean
        size=min(n_null_dims, z_df.shape[1] - 4),
        replace=False
    )

    # Pre-compute chirality arrays aligned with merged
    chir_arrays = {}
    for feat in top_nfl:
        if feat in merged.columns:
            chir_arrays[feat] = merged[feat].values.astype(float)

    # Pre-compute code → z_df row mapping
    code_to_zidx = {code: idx for idx, code in z_df['code'].items()}
    merged_codes = merged['code'].values

    null_mean_r = []
    for i, d in enumerate(random_dims):
        if (i + 1) % 50 == 0:
            print(f"    Processing null dim {i + 1}/{n_null_dims}...")
        dcol = f'dim{d}'
        if dcol not in z_df.columns:
            continue
        # Get dim values aligned with merged
        dim_d_vals = np.array([
            float(z_df.loc[code_to_zidx[code], dcol]) if code in code_to_zidx else np.nan
            for code in merged_codes
        ])

        r_vals = []
        for feat, x_f in chir_arrays.items():
            ok_d = np.isfinite(x_f) & np.isfinite(dim_d_vals)
            if ok_d.sum() > 5:
                r_vals.append(abs(stats.pearsonr(x_f[ok_d], dim_d_vals[ok_d])[0]))
        if r_vals:
            null_mean_r.append(np.mean(r_vals))

    null_mean_r = np.array(null_mean_r)
    target_mean_r = float(nfl_abs.mean())
    percentile = float((null_mean_r < target_mean_r).mean() * 100)
    p_spec = 1 - percentile / 100

    print(f"  {dim_name}: mean |r| with NFL chirality = {target_mean_r:.4f}")
    print(f"  Null: mean={null_mean_r.mean():.4f} ± {null_mean_r.std():.4f}")
    print(f"  Percentile: {percentile:.1f}th, p_empirical={p_spec:.4f}")

    results['spec_percentile'] = percentile
    results['spec_p'] = p_spec
    results['null_mean_r'] = null_mean_r
    results['target_mean_r'] = target_mean_r

    # ================================================================
    # ANALYSIS 4: Mediation — chirality → dim → NFL (ALS only, N=20)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 4: Mediation — chirality → {dim_name} → NFL (N={N_nfl})")
    print(f"{'=' * 60}")

    med_rows = []
    avail_top = [f for f in top_nfl if f in merged_nfl.columns]

    for feat in avail_top:
        x = merged_nfl[feat].values.astype(float)
        dm = merged_nfl[dim_name].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(nfl_vals) & np.isfinite(dm)
        if ok.sum() < 10:
            continue

        x_ok, nfl_ok, dm_ok = x[ok], nfl_vals[ok], dm[ok]
        # Standardize
        x_z = (x_ok - x_ok.mean()) / (x_ok.std() + 1e-12)
        nfl_z = (nfl_ok - nfl_ok.mean()) / (nfl_ok.std() + 1e-12)
        dm_z = (dm_ok - dm_ok.mean()) / (dm_ok.std() + 1e-12)

        # Paths
        r_c, p_c = stats.pearsonr(x_z, nfl_z)           # c: total effect
        r_a, p_a = stats.pearsonr(x_z, dm_z)             # a: chirality → dim
        X_med = np.column_stack([x_z, dm_z])
        reg = LinearRegression().fit(X_med, nfl_z)
        c_prime = reg.coef_[0]                             # c': direct effect
        b = reg.coef_[1]                                   # b: dim → NFL (controlling chirality)
        indirect = r_a * b                                 # ab: indirect effect

        # Sobel test
        n = len(x_z)
        se_a = np.sqrt((1 - r_a ** 2) / (n - 2))
        r2_full = reg.score(X_med, nfl_z)
        se_b = np.sqrt((1 - r2_full) / (n - 3))
        se_indirect = np.sqrt(r_a ** 2 * se_b ** 2 + b ** 2 * se_a ** 2)
        z_sobel = indirect / (se_indirect + 1e-12)
        p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))

        # Proportion mediated
        prop_med = indirect / r_c if abs(r_c) > 0.01 else np.nan

        med_rows.append({
            'feature': feat,
            'total_effect_c': r_c,
            'total_p': p_c,
            'path_a_chir_to_dim': r_a,
            'path_a_p': p_a,
            'path_b_dim_to_nfl': b,
            'direct_effect_c_prime': c_prime,
            'indirect_effect_ab': indirect,
            'sobel_z': z_sobel,
            'sobel_p': p_sobel,
            'proportion_mediated': prop_med,
        })

    med_df = pd.DataFrame(med_rows)
    med_df.to_csv(outdir / f'mediation_{dim_name}_nfl.csv', index=False)

    # Print
    print(f"\n  {'Feature':<30s} {'c(tot)':>7s} {'a':>7s} {'b':>7s} "
          f"{'ab':>8s} {'%med':>6s} {'Sobel p':>8s}")
    print(f"  {'-' * 80}")
    for _, r in med_df.iterrows():
        pm = f"{r['proportion_mediated'] * 100:.0f}%" if np.isfinite(r['proportion_mediated']) else "N/A"
        sig = " *" if r['sobel_p'] < 0.05 else ""
        print(f"  {r['feature']:<30s} {r['total_effect_c']:+.3f}  "
              f"{r['path_a_chir_to_dim']:+.3f}  {r['path_b_dim_to_nfl']:+.3f}  "
              f"{r['indirect_effect_ab']:+.4f} {pm:>6s} {r['sobel_p']:.4f}{sig}")

    # Safety check: Only calculate significance if the sobel_p column actually exists
if 'sobel_p' in med_df.columns and not med_df.empty:
    n_sig = int((med_df['sobel_p'] < 0.05).sum())
else:
    n_sig = 0
    print("  No features met the criteria to calculate sobel_p.")
    print(f"\n  Significant Sobel tests: {n_sig}/{len(med_df)}")

    results['n_sig_mediations'] = n_sig
    results['n_total_mediations'] = len(med_df)
    results['med_df'] = med_df

    # ================================================================
    # ANALYSIS 5: Chirality group importance
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS 5: Chirality group importance for {dim_name}")
    print(f"{'=' * 60}")

    groups = {}
    for feat in corr_df['feature']:
        parts = feat.split('_')
        ct = parts[0] + '_' + parts[1] if len(parts) >= 2 else 'unknown'
        groups.setdefault(ct, []).append(feat)

    grp_rows = []
    for ct, feats in groups.items():
        sub = corr_df[corr_df['feature'].isin(feats)]
        grp_rows.append({
            'chirality': ct,
            'n_features': len(sub),
            'mean_abs_r': float(sub['abs_pearson_r'].mean()),
            'max_abs_r': float(sub['abs_pearson_r'].max()),
            'n_sig_raw': int((sub['pearson_p'] < 0.05).sum()),
            'is_nfl_species': ct in NFL_CHIRALITY_SPECIES,
        })

    grp_df = pd.DataFrame(grp_rows).sort_values('mean_abs_r', ascending=False).reset_index(drop=True)
    grp_df.to_csv(outdir / f'chirality_group_{dim_name}.csv', index=False)

    print(f"\n  {'Chirality':<10s} {'n':>4s} {'mean|r|':>8s} {'max|r|':>8s} {'n_sig':>6s}")
    print(f"  {'-' * 42}")
    for _, r in grp_df.iterrows():
        star = "  ← NFL" if r['is_nfl_species'] else ""
        print(f"  {r['chirality']:<10s} {r['n_features']:>4d} {r['mean_abs_r']:>8.3f} "
              f"{r['max_abs_r']:>8.3f} {r['n_sig_raw']:>4d}/{r['n_features']:<2d}{star}")

    results['grp_df'] = grp_df

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_figure(all_results, outdir, model_label):
    """
    Create a comparison figure with rows:
      Row 0: Enrichment boxplots
      Row 1: Specificity null distributions
      Row 2: Mediation proportion mediated
    """
    n_dims = len(all_results)
    fig, axes = plt.subplots(3, n_dims, figsize=(6 * n_dims, 16))
    if n_dims == 1:
        axes = axes.reshape(3, 1)

    for col, res in enumerate(all_results):
        dim_name = res['dim']

        # --- Row 0: Enrichment ---
        ax = axes[0, col]
        nfl_r = res['nfl_abs']
        oth_r = res['other_abs']
        bp = ax.boxplot(
            [oth_r, nfl_r],
            tick_labels=['Other\nchirality', 'NFL-top\nchirality'],
            patch_artist=True, widths=0.5
        )
        bp['boxes'][0].set_facecolor('#bdc3c7')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel('|Pearson r| with latent dim', fontsize=10)
        ax.scatter([1], [oth_r.mean()], color='black', s=60, zorder=5, marker='D')
        ax.scatter([2], [nfl_r.mean()], color='black', s=60, zorder=5, marker='D')
        ax.text(2.3, nfl_r.mean(), f'{nfl_r.mean():.3f}', va='center', fontsize=9)
        ax.text(0.7, oth_r.mean(), f'{oth_r.mean():.3f}', va='center', fontsize=9, ha='right')
        p_str = f"{res['enrich_p']:.1e}" if res['enrich_p'] < 0.001 else f"{res['enrich_p']:.3f}"
        ax.set_title(f"{dim_name}\nEnrichment p = {p_str}", fontweight='bold', fontsize=12)

        # --- Row 1: Specificity ---
        ax = axes[1, col]
        null_arr = res['null_mean_r']
        tgt = res['target_mean_r']
        ax.hist(null_arr, bins=25, color='#bdc3c7', edgecolor='k', alpha=0.8, label='Random dims')
        ax.axvline(tgt, color='red', linewidth=2.5, label=f'{dim_name} = {tgt:.3f}')
        ax.set_xlabel('Mean |r| with NFL chirality features')
        ax.set_ylabel('Count')
        pct = res['spec_percentile']
        ax.set_title(f'Specificity: {pct:.1f}th percentile', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)

        # --- Row 2: Mediation ---
        ax = axes[2, col]
        mdf = res['med_df']
        if len(mdf) > 0:
            mdf_sorted = mdf.sort_values('proportion_mediated', ascending=True).reset_index(drop=True)
            prop_vals = mdf_sorted['proportion_mediated'].values * 100
            prop_vals = np.nan_to_num(prop_vals, nan=0)
            colors = ['#e74c3c' if p < 0.05 else '#3498db' for p in mdf_sorted['sobel_p']]

            y_pos = range(len(mdf_sorted))
            ax.barh(y_pos, prop_vals, color=colors, edgecolor='k', linewidth=0.5)
            ax.set_yticks(list(y_pos))

            # Shortened feature labels
            labels = []
            for f in mdf_sorted['feature']:
                short = f.replace('ch', '').replace('_grad_center_', ' grad ').replace('_kurtosis_', ' kurt ')
                short = short.replace('_skewness_', ' skew ').replace('_range_', ' rng ')
                short = short.replace('_cv_', ' cv ').replace('_std_', ' std ')
                short = short.replace('_iqr_', ' iqr ')
                labels.append(short)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('% mediated through latent dim')
        else:
            ax.text(0.5, 0.5, 'No mediation data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        n_sig = res['n_sig_mediations']
        n_tot = res['n_total_mediations']
        ax.set_title(f'Mediation: {n_sig}/{n_tot} significant', fontweight='bold', fontsize=12)

    # Legend for mediation
    sig_patch = mpatches.Patch(color='#e74c3c', label='Sobel p < 0.05')
    ns_patch = mpatches.Patch(color='#3498db', label='Not significant')
    fig.legend(handles=[sig_patch, ns_patch], loc='lower center', ncol=2, fontsize=11)

    dims_str = ', '.join([r['dim'] for r in all_results])
    plt.suptitle(f'{model_label}: Chirality–Latent Dim Convergence ({dims_str})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(outdir / 'convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ convergence_comparison.png")


def create_group_figure(all_results, outdir, model_label):
    """Bar chart of chirality group importance per dim."""
    n_dims = len(all_results)
    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 6))
    if n_dims == 1:
        axes = [axes]

    for col, res in enumerate(all_results):
        ax = axes[col]
        gdf = res['grp_df'].sort_values('mean_abs_r', ascending=True).reset_index(drop=True)
        colors = ['#e74c3c' if nfl else '#3498db' for nfl in gdf['is_nfl_species']]
        ax.barh(range(len(gdf)), gdf['mean_abs_r'].values, color=colors,
                edgecolor='k', linewidth=0.5)
        ax.set_yticks(range(len(gdf)))
        ax.set_yticklabels(gdf['chirality'].values, fontsize=10)
        ax.set_xlabel('Mean |r| across all descriptors')
        ax.set_title(res['dim'], fontweight='bold', fontsize=12)

    fig.legend(handles=[
        mpatches.Patch(color='#e74c3c', label='NFL-correlated (ch8_7, ch9_4, ch9_5)'),
        mpatches.Patch(color='#3498db', label='Other chiralities'),
    ], loc='lower center', ncol=2, fontsize=11)

    plt.suptitle(f'{model_label}: Chirality Group Importance by Latent Dimension',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(outdir / 'chirality_group_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ chirality_group_importance.png")


def create_correlation_heatmaps(all_results, z_df, ch_df, clin_df, chir_cols,
                                 top_nfl, outdir, model_label):
    """
    Create correlation heatmaps per dimension showing:
      1. Intercorrelation matrix of NFL-top chirality features + latent dim + NFL
      2. Chirality-type × timepoint heatmap of mean |r| with latent dim
    """
    for res in all_results:
        dim_name = res['dim']

        # ----------------------------------------------------------------
        # Heatmap 1: Intercorrelation matrix (ALS only, includes NFL)
        # ----------------------------------------------------------------
        merged_nfl = z_df[['code', 'code_clean', 'group', dim_name]].merge(
            ch_df, on='code_clean'
        ).merge(
            clin_df[['code_clean', 'nfl_conc']], on='code_clean', how='inner'
        )

        avail_feats = [f for f in top_nfl if f in merged_nfl.columns]
        cols_for_heatmap = avail_feats + [dim_name, 'nfl_conc']
        mat = merged_nfl[cols_for_heatmap].astype(float).dropna()

        # Shorten labels
        short_labels = []
        for c in cols_for_heatmap:
            if c == dim_name:
                short_labels.append(dim_name)
            elif c == 'nfl_conc':
                short_labels.append('NFL conc.')
            else:
                short = c.replace('ch', '').replace('_grad_center_', ' grad ')
                short = short.replace('_kurtosis_', ' kurt ').replace('_skewness_', ' skew ')
                short = short.replace('_range_', ' rng ').replace('_cv_', ' cv ')
                short = short.replace('_std_', ' std ').replace('_iqr_', ' iqr ')
                short_labels.append(short)

        corr_mat = mat.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        sns.heatmap(
            corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
            ax=ax
        )
        ax.set_title(f'{model_label} — {dim_name}: Intercorrelation Matrix\n'
                     f'(NFL-top chirality features + latent dim + NFL, N={len(mat)})',
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(outdir / f'heatmap_intercorrelation_{dim_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # ----------------------------------------------------------------
        # Heatmap 2: Chirality-type × timepoint |r| with latent dim (N=39)
        # ----------------------------------------------------------------
        corr_df = res['corr_df']

        # Parse chirality type and timepoint from feature names
        chirality_types = sorted(set(
            '_'.join(f.split('_')[:2]) for f in corr_df['feature']
            if len(f.split('_')) >= 2
        ))
        timepoints = ['0h', '6h', '24h']
        descriptors_all = set()
        for f in corr_df['feature']:
            parts = f.split('_')
            if len(parts) >= 4:
                # e.g. ch8_7_cv_6h -> descriptor = cv
                desc = '_'.join(parts[2:-1])
                descriptors_all.add(desc)

        # Build matrix: chirality_type × timepoint, aggregated across descriptors
        heat_data = pd.DataFrame(index=chirality_types, columns=timepoints, dtype=float)
        heat_count = pd.DataFrame(index=chirality_types, columns=timepoints, dtype=float)

        for _, row in corr_df.iterrows():
            feat = row['feature']
            parts = feat.split('_')
            if len(parts) < 4:
                continue
            ct = parts[0] + '_' + parts[1]
            tp = parts[-1]
            if ct in chirality_types and tp in timepoints:
                if pd.isna(heat_data.loc[ct, tp]):
                    heat_data.loc[ct, tp] = 0.0
                    heat_count.loc[ct, tp] = 0.0
                heat_data.loc[ct, tp] += row['abs_pearson_r']
                heat_count.loc[ct, tp] += 1

        heat_mean = heat_data / heat_count.replace(0, np.nan)

        # Sort by overall mean |r|
        row_order = heat_mean.mean(axis=1).sort_values(ascending=False).index
        heat_mean = heat_mean.loc[row_order]

        fig, ax = plt.subplots(figsize=(6, 8))
        sns.heatmap(
            heat_mean.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white', square=False,
            cbar_kws={'label': 'Mean |Pearson r|', 'shrink': 0.7},
            ax=ax
        )
        # Highlight NFL species
        for i, ct in enumerate(heat_mean.index):
            if ct in NFL_CHIRALITY_SPECIES:
                ax.add_patch(plt.Rectangle((0, i), 3, 1, fill=False,
                             edgecolor='red', linewidth=2.5))
        ax.set_ylabel('SWCNT chirality')
        ax.set_xlabel('Timepoint')
        ax.set_title(f'{model_label} — {dim_name}: Mean |r| by Chirality × Timepoint\n'
                     f'(red box = NFL-correlated species, N={len(corr_df["feature"].unique())} features)',
                     fontweight='bold', fontsize=11)
        plt.tight_layout()
        plt.savefig(outdir / f'heatmap_chirality_timepoint_{dim_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # ----------------------------------------------------------------
        # Heatmap 3: Full descriptor × chirality-type heatmap (best timepoint)
        # ----------------------------------------------------------------
        # For each chirality × descriptor, take the max |r| across timepoints
        descriptors_sorted = sorted(descriptors_all)
        heat_full = pd.DataFrame(index=chirality_types, columns=descriptors_sorted, dtype=float)

        for _, row in corr_df.iterrows():
            feat = row['feature']
            parts = feat.split('_')
            if len(parts) < 4:
                continue
            ct = parts[0] + '_' + parts[1]
            desc = '_'.join(parts[2:-1])
            if ct in chirality_types and desc in descriptors_sorted:
                current = heat_full.loc[ct, desc]
                if pd.isna(current) or row['abs_pearson_r'] > current:
                    heat_full.loc[ct, desc] = row['abs_pearson_r']

        heat_full = heat_full.loc[row_order]  # keep same chirality order
        # Sort descriptors by mean |r|
        desc_order = heat_full.mean(axis=0).sort_values(ascending=False).index
        heat_full = heat_full[desc_order]

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(
            heat_full.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.3, linecolor='white', square=False,
            cbar_kws={'label': 'Max |Pearson r| across timepoints', 'shrink': 0.6},
            ax=ax, annot_kws={'fontsize': 7}
        )
        # Highlight NFL species rows
        for i, ct in enumerate(heat_full.index):
            if ct in NFL_CHIRALITY_SPECIES:
                ax.add_patch(plt.Rectangle((0, i), len(desc_order), 1, fill=False,
                             edgecolor='red', linewidth=2.5))
        ax.set_ylabel('SWCNT chirality')
        ax.set_xlabel('Descriptor (best timepoint)')
        ax.set_title(f'{model_label} — {dim_name}: Max |r| by Chirality × Descriptor\n'
                     f'(red box = NFL-correlated species)',
                     fontweight='bold', fontsize=11)
        plt.tight_layout()
        plt.savefig(outdir / f'heatmap_chirality_descriptor_{dim_name}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ heatmap_intercorrelation_{dim_name}.png")
        print(f"  ✓ heatmap_chirality_timepoint_{dim_name}.png")
        print(f"  ✓ heatmap_chirality_descriptor_{dim_name}.png")


def write_summary(all_results, outdir, model_label):
    """Write text summary file."""
    lines = [
        f"CHIRALITY-LATENT DIMENSION CONVERGENCE ANALYSIS",
        f"{'=' * 60}",
        f"Model: {model_label}",
        f"Dimensions analyzed: {', '.join(r['dim'] for r in all_results)}",
        f"",
    ]

    # Header
    dims = [r['dim'] for r in all_results]
    w = 14
    hdr = f"{'Metric':<35s}" + ''.join(f"{d:>{w}s}" for d in dims)
    lines.append(hdr)
    lines.append('-' * len(hdr))

    # Rows
    def fmt_row(label, key, fmt_func):
        vals = ''.join(f"{fmt_func(r.get(key, 'N/A')):>{w}s}" for r in all_results)
        return f"{label:<35s}{vals}"

    lines.append(fmt_row('r with NFL', 'r_nfl', lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
    lines.append(fmt_row('Enrichment p', 'enrich_p', lambda x: f"{x:.2e}" if isinstance(x, float) else str(x)))
    lines.append(fmt_row('NFL-chir mean |r|', 'enrich_nfl_mean', lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    lines.append(fmt_row('Background mean |r|', 'enrich_other_mean', lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    lines.append(fmt_row('Specificity percentile', 'spec_percentile', lambda x: f"{x:.1f}th" if isinstance(x, float) else str(x)))

    for r in all_results:
        r['med_str'] = f"{r['n_sig_mediations']}/{r['n_total_mediations']}"
    lines.append(fmt_row('Significant mediations', 'med_str', str))

    lines.append('')
    lines.append('Top chirality groups per dimension:')
    for r in all_results:
        lines.append(f"\n  {r['dim']}:")
        gdf = r['grp_df']
        for _, row in gdf.head(5).iterrows():
            star = " ← NFL" if row['is_nfl_species'] else ""
            lines.append(f"    {row['chirality']:<10s}  mean|r|={row['mean_abs_r']:.3f}  "
                         f"n_sig={row['n_sig_raw']}/{row['n_features']}{star}")

    txt = '\n'.join(lines)
    (outdir / 'convergence_summary.txt').write_text(txt, encoding='utf-8')
    print(f"  ✓ convergence_summary.txt")
    print(f"\n{txt}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chirality–Latent Dimension Convergence Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chirality_dim_convergence.py \\
      --z_agg z_agg1.csv --target_dims 35 --model_label "Model 1"

  python chirality_dim_convergence.py \\
      --z_agg z_agg3.csv --target_dims 477,350 --model_label "Model 3"
        """
    )
    parser.add_argument('--z_agg', type=str, required=True,
                        help='Path to z_agg CSV (code, group, dim0..dim511)')
    parser.add_argument('--chirality_csv', type=str, required=True,
                        help='Path to chirality_full_descriptors.csv')
    parser.add_argument('--clinical_csv', type=str, required=True,
                        help='Path to early_slope.csv')
    parser.add_argument('--target_dims', type=str, required=True,
                        help='Comma-separated dim indices (e.g. 35 or 477,350)')
    parser.add_argument('--model_label', type=str, default='Model',
                        help='Label for the model in output titles')
    parser.add_argument('--n_null_dims', type=int, default=200,
                        help='Number of random dims for null distribution')
    parser.add_argument('--output_dir', type=str, default='./convergence_results',
                        help='Output directory')

    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse target dims
    target_dims = [int(d.strip()) for d in args.target_dims.split(',')]
    print(f"{'=' * 70}")
    print(f"CHIRALITY–LATENT DIM CONVERGENCE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Model: {args.model_label}")
    print(f"Target dims: {target_dims}")
    print(f"Null dims: {args.n_null_dims}")
    print(f"Output: {outdir}")

    # --- Load data ---
    print(f"\nLoading data...")
    z_df = pd.read_csv(args.z_agg)
    z_df['code_clean'] = clean_code(z_df['code'])
    print(f"  z_agg: {z_df.shape[0]} samples × {z_df.shape[1] - 3} dims")

    ch_df = pd.read_csv(args.chirality_csv)
    ch_df = ch_df.drop(columns=['group'], errors='ignore')
    ch_code = ch_df['code']
    ch_df = ch_df.drop(columns=['code'])
    chir_cols = ch_df.columns.tolist()
    ch_df['code_clean'] = ch_code
    print(f"  chirality: {len(chir_cols)} features")

    clin_df = pd.read_csv(args.clinical_csv)
    clin_df['code_clean'] = clean_code(clin_df['code'])
    print(f"  clinical: {clin_df.shape[0]} ALS patients")

    # --- Run analysis for each target dim ---
    all_results = []
    for dim_idx in target_dims:
        dim_name = f'dim{dim_idx}'
        if dim_name not in z_df.columns:
            print(f"\n  ⚠ {dim_name} not found in z_agg! Skipping.")
            continue
        res = run_convergence_analysis(
            dim_name=dim_name,
            dim_idx=dim_idx,
            z_df=z_df,
            ch_df=ch_df,
            clin_df=clin_df,
            chir_cols=chir_cols,
            top_nfl=TOP_NFL_CHIRALITY,
            n_null_dims=args.n_null_dims,
            outdir=outdir,
            model_label=args.model_label,
        )
        all_results.append(res)

    if not all_results:
        print("\nNo valid dimensions found. Exiting.")
        return

    # --- Create visualizations ---
    print(f"\n{'=' * 70}")
    print("Creating visualizations...")
    print(f"{'=' * 70}")

    create_comparison_figure(all_results, outdir, args.model_label)
    create_group_figure(all_results, outdir, args.model_label)
    create_correlation_heatmaps(all_results, z_df, ch_df, clin_df, chir_cols,
                                 TOP_NFL_CHIRALITY, outdir, args.model_label)
    write_summary(all_results, outdir, args.model_label)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {outdir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
