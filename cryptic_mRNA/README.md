# cryptic_mRNA

STMN2 cryptic exon detection: SWCNT sensor assays + molecular modeling

**97 scripts** across **11 analysis types**.

## Analysis types

| Folder | Scripts | What it covers |
|---|---|---|
| [`01_preprocessing/`](./01_preprocessing/) | 17 | Data ingestion, renaming, cropping, cache building and system/structure setup |
| [`02_peak_fitting/`](./02_peak_fitting/) | 11 | Gaussian / free-form spectral peak fitting and fit diagnostics |
| [`03_batch_correction/`](./03_batch_correction/) | 1 | EEM batch-effect correction, reference calibration and detrending |
| [`05_feature_selection_importance/`](./05_feature_selection_importance/) | 5 | Recursive feature selection, saliency, permutation and latent-dimension importance |
| [`07_regression/`](./07_regression/) | 1 | ALSFRS score / slope regression and PLS regression models |
| [`08_statistics/`](./08_statistics/) | 8 | PERMANOVA, PERMDISP, linear mixed models, ANOVA, confounder and dependence testing |
| [`09_dimensionality_reduction/`](./09_dimensionality_reduction/) | 2 | PCA, UMAP and latent-dimension convergence analyses |
| [`10_qc_validation/`](./10_qc_validation/) | 9 | Quality control, technical controls, surrogate generation and pipeline smoke tests |
| [`11_response_analysis/`](./11_response_analysis/) | 22 | Dose-response, serum/water assay response and blinded/unblinded readouts |
| [`12_molecular_modeling/`](./12_molecular_modeling/) | 12 | Docking, hybridization thermodynamics, conformer generation and MD-derived descriptors |
| [`13_figures/`](./13_figures/) | 9 | Publication and diagnostic figure generation |

## 01_preprocessing

Data ingestion, renaming, cropping, cache building and system/structure setup

### Standalone scripts

| File | Notes |
|---|---|
| `00_build_cnt.py` | 00_build_cnt.py  —  Build the (n,m) SWCNT, periodic along z, and emit: |
| `01_build_oligos.sh` | 01_build_oligos.sh — build the STMN2-CE sensor nucleic acids with AmberTools |
| `02_assemble_system.py` | 02_assemble_system.py — place the sensor next to the frozen (9,4) tube and emit |
| `03_solvate_ions.sh` | 03_solvate_ions.sh <hyb|unhyb> — CHARMM36 topology for the nucleic acids, |
| `04_make_index.sh` | 04_make_index.sh <hyb|unhyb> — build index groups used by the .mdp files and |
| `amber2charmm_names.py` | amber2charmm_names.py  —  translate an Amber-named nucleic-acid PDB into names |
| `archive_results.py` | Timestamped archive (§21): code, config, provenance, result tables, figures, |
| `buildProject.sh` |  |
| `build_structures.py` | Conceptual full-sensor starting structures (§12, Gate 5 = conceptual). |
| `build_swcnts.py` | Build + validate the 5 core SWCNT chiralities (§9). |
| `fetch_reference.py` | Fetch the authoritative STMN2 transcript live (§5.1). |
| `figstyle.py` | Shared figure style (§13): 300 dpi, editable SVG text, no rainbow colormap. |
| `gen_remd_mdps.py` | gen_remd_mdps.py <hyb|unhyb> — generate a geometric temperature ladder and one |
| `lib_common.py` | Shared helpers: config loading, provenance, sequence utilities. |
| `merge_cnt_topology.py` | merge_cnt_topology.py — insert the frozen CNT molecule into a pdb2gmx topology |
| `record_versions.sh` | Provenance capture (§6.4). Run from the pipeline root. |
| `unity_modules.sh` | Source this on Unity to load the GPU GROMACS + supporting modules. |

## 02_peak_fitting

Gaussian / free-form spectral peak fitting and fit diagnostics

### Related script variants

<details><summary><code>extract_gaussian/</code> &mdash; 8 variants</summary>

| File | Notes |
|---|---|
| `extract_gaussian.py` | extract_gaussian.py |
| `extract_gaussian_exp2.py` | Per-chirality split-Gaussian+baseline fit for exp2 (reuses exp1 extract_gaussian fit code). |
| `extract_gaussian_exp3.py` | Per-chirality split-Gaussian+baseline fit for exp3 (reuses exp1 extract_gaussian fit code). |
| `extract_gaussian_exp4_serum.py` | Per-chirality split-Gaussian fit for exp4 SERUM (STMN_22 = 22nt GT15-STMN2-CE sensor). |
| `extract_gaussian_exp4_water.py` | Per-chirality split-Gaussian fit for exp4 SERUM (STMN_22 = 22nt GT15-STMN2-CE sensor). |
| `extract_gaussian_exp5_20.py` | Per-chirality split-Gaussian fit for exp5 / 20nt GT15-STMN2-CE sensor in defined background |
| `extract_gaussian_exp5_22.py` | Per-chirality split-Gaussian fit for exp5 / 22nt GT15-STMN2-CE sensor in defined background |
| `extract_gaussian_serum.py` | Per-chirality split-Gaussian fit for exp3 SERUM plate (reuses exp1 fit code). |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `diagnose_fits.py` | Diagnose whether low R2 = genuine dim peaks or a fitting bug. |
| `fit_free.py` | Fit-FREE spectral descriptors (no Gaussian, no curve_fit, no R2 gate). |
| `test_is_it_fitting.py` | Is the brightness->center coupling a FITTING problem (estimator artifact) or a |

## 03_batch_correction

EEM batch-effect correction, reference calibration and detrending

### Standalone scripts

| File | Notes |
|---|---|
| `detrend_check.py` | Left-right detrend then re-test (1) dose null and (2) random-RNA suppression, since the two 0-types |

## 05_feature_selection_importance

Recursive feature selection, saliency, permutation and latent-dimension importance

### Related script variants

<details><summary><code>analyze_feature_tracking/</code> &mdash; 3 variants</summary>

| File | Notes |
|---|---|
| `analyze_feature_tracking.py` | Which features move with total intensity? |
| `analyze_feature_tracking_3tube.py` | Feature-vs-total analysis restricted to the 3 reliably-fit tubes: |
| `analyze_feature_tracking_r2.py` | Feature-vs-total analysis, but using ONLY peaks with fit R2 > 0.95. |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `analyze_brightness_specificity.py` | Is the global brightening SPECIFIC to STMN2-CE on GT15-STMN2 vs the GT15 control? |
| `rank_specificity.py` | Rank every (chirality, feature) by SPECIFIC STMN2-CE dose response: |

## 07_regression

ALSFRS score / slope regression and PLS regression models

### Standalone scripts

| File | Notes |
|---|---|
| `hill_cooperativity.py` | Hill-coefficient cooperativity test for the GT15-STMN2 water dose-response (§14). |

## 08_statistics

PERMANOVA, PERMDISP, linear mixed models, ANOVA, confounder and dependence testing

### Related script variants

<details><summary><code>test_row_dependence/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `test_row_dependence.py` | Test user's observation: STMN2-CE concentration trend in WATER is more evident in LOWER rows |
| `test_row_dependence_serum.py` | Same UPPER vs LOWER row split on the SERUM plate (where the high-copy response was robust). |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `analyze_interaction.py` | analyze_interaction.py |
| `analyze_position_ratio.py` | analyze_position_ratio.py |
| `analyze_shift_coherence.py` | Deeper look at the FITTED peak centers: are the shifts COHERENT (all tubes move |
| `test_neighbor_artifact.py` | Test the NEIGHBOR / cross-talk hypothesis in exp4 serum: |
| `test_position_detrend.py` | Estimate the left-right SPATIAL gradient from the BELOW-THRESHOLD (near-baseline) columns only, |
| `within_row_trend.py` | Within-EACH-ROW concentration trend, and whether it is STABLE (reproducible) across the 8 rows. |

## 09_dimensionality_reduction

PCA, UMAP and latent-dimension convergence analyses

### Related script variants

<details><summary><code>analyze_chirality/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `analyze_chirality.py` | Chirality docking analysis (§11.6, Gate 4). Reads chirality pose_metrics, |
| `analyze_chirality_exp2.py` | Per-chirality dose analysis for exp2 (chirality_gaussian_descriptors_exp2.csv). |

</details>


## 10_qc_validation

Quality control, technical controls, surrogate generation and pipeline smoke tests

### Standalone scripts

| File | Notes |
|---|---|
| `aggregate_smoke.py` | Aggregate smoke-test status (Gate 0). Reads all per-tool JSONs and writes |
| `benchmark_gate.py` | Benchmark gate (Gate 3, §11.3) — correct bootstrap median-difference logic. |
| `gate2_thermo.py` | Gate 2 (thermodynamic model validity) status, from hybridization outputs. |
| `smoke_docking.py` | Docking-env smoke tests: Meeko ligand prep, Vina receptor parsing, base-vs- |
| `smoke_seq.py` | Seq-env smoke tests: MELTING, NUPACK (blocker-aware), Biopython cross-check. |
| `smoke_struct.py` | Struct-env smoke tests: ASE SWCNT generator, Open Babel, AmberTools nucleic |
| `test_toy_wiring.py` | Toy test (§16): fast checks of schema wiring + the integrity-critical rule that |
| `validate_outputs.py` | Output validation (§16, Gate 6 component). Verifies required outputs/gates |
| `validate_sequences.py` | Sequence validation gate (§5.3, §5.4, Gate 1). |

## 11_response_analysis

Dose-response, serum/water assay response and blinded/unblinded readouts

### Versioned script families

<details><summary><code>analyze_2H_water/</code> &mdash; 3 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_analyze_2H_water.py` | exp2 / 2H_WATER  --- BLIND STMN2-CE titration in neuronal-mRNA-enriched water. |
| v2 | `v2_analyze_2H_water_v2.py` | exp2 / 2H_WATER  -- CORRECTED layout (per user). |
| v3 | `v3_analyze_2H_water_v3.py` | exp2/2H_WATER v3 -- rows=8 technical replicates; between-row gradient = acquisition |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `analyze_all_dose.py` | DECISIVE dose-vs-edge test across all 4 exp2 plates. |
| `analyze_ch65_only.py` | ch6_5-only analysis. (6,5) is the single tube with R2>=0.95 in ALL 95 wells |
| `analyze_ch75_dlambda.py` | (7,5)-specific Delta-lambda analysis -- the paper's designated sensor readout. |
| `analyze_exp3_blind.py` | exp3 BLIND per-condition analysis (same analysis types as exp2). |
| `analyze_exp3_nofit_blind.py` | exp3 BLIND per-condition analysis WITHOUT peak fitting (model-free cross-check). |
| `analyze_exp3_unblind.py` | exp3 UNBLINDED (v2 - no bogus position gradient; only legitimate row/heating drift removed). |
| `analyze_exp4_serum_blind.py` | exp4 SERUM (STMN_22 = 22nt GT15-STMN2-CE sensor) -- BLIND model-free per-condition analysis. |
| `analyze_exp4_serum_fit_blind.py` | exp4 SERUM (STMN_22) -- BLIND FITTED per-condition analysis (mean gauss_max over clean chiralities). |
| `analyze_exp4_serum_unblind.py` | exp4 SERUM UNBLINDED (STMN_22 = 22nt GT15-STMN2-CE sensor, in serum). |
| `analyze_exp4_water.py` | exp4 WATER (STMN_22 = 22nt GT15-STMN2-CE sensor) full pipeline: blind ranking + gradients |
| `analyze_exp5_20.py` | exp5 / 20nt GT15-STMN2-CE sensor in defined background (SDS1%+BSA0.5%+randomRNA 30ng/mL). |
| `analyze_exp5_22.py` | exp5 / 22nt GT15-STMN2-CE sensor in DEFINED BACKGROUND (SDS 1% + BSA 0.5% + random RNA 30 ng/mL). |
| `analyze_response.py` | analyze_response.py |
| `analyze_serum_blind.py` | exp3 SERUM — BLIND per-condition analysis (fitted + model-free), same pipeline as water. |
| `analyze_serum_generic.py` | SERUM plate analyzer. Layout: 8 rows x 5 columns, ROW-MAJOR (5 cols/row). Rows = technical |
| `analyze_serum_nofit_blind.py` | exp3 SERUM — BLIND model-free (NO fitting) per-condition analysis (mirrors water no-fit). |
| `analyze_serum_unblind.py` | exp3 SERUM UNBLINDED. 8 rows x 12 cols row-major (col11=7 reps). In serum matrix: |
| `analyze_water_generic.py` | Generic 'water' plate analyzer (corrected layout): 6 conc-COLUMNS x 8 replicate ROWS, |
| `ratio_readouts.py` | Intensity-INDEPENDENT (ratiometric / wavelength) readouts for exp5/22, which cancel the |

## 12_molecular_modeling

Docking, hybridization thermodynamics, conformer generation and MD-derived descriptors

### Standalone scripts

| File | Notes |
|---|---|
| `base_geometry.py` | base_geometry.py — per-nucleobase radial distance from the tube surface and base |
| `blueshift_predict.py` | blueshift_predict.py — combine Hyb vs Unhyb radial densities into a predicted |
| `free_energy_cycle.py` | free_energy_cycle.py — Harvey Suppl. Fig S11 thermodynamic cycle, adapted to the |
| `generate_conformers.py` | Anchor conformer ensembles (§10) — nucleic-acid-aware, NOT RDKit/MMFF. |
| `integrate_experimental.py` | Experimental NIR integration (§14) — REAL data from ../../exp1/. |
| `lib_dock.py` | Controlled Vina docking helpers (§11): sidewall-offset box, map reuse, |
| `lib_ligprep.py` | Controlled rigid-ligand PDBQT preparation for ssDNA conformers (§10.3). |
| `lib_struct.py` | SWCNT geometry + controlled receptor-PDBQT writing (§9). |
| `radial_density.py` | radial_density.py — radial number-density profiles of WATER (OW) and PHOSPHATE (P) |
| `run_docking.py` | Controlled Vina docking driver (§11) — benchmark + chirality stages. |
| `run_hybridization.py` | Hybridization thermodynamics (§8, Gate 2) — MELTING + Biopython + descriptors. |
| `run_nupack.py` | NUPACK 4.1 analyses (§8.2, §8.4) — used only where legitimate. |

## 13_figures

Publication and diagnostic figure generation

### Related script variants

<details><summary><code>plot_curves/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `plot_curves.py` | Separated GT15-STMN2 vs GT15 dose curves (absolute + normalized-to-0-conc). |
| `plot_curves_clean.py` | Dose curves on the clean tube set: total3 (ch6_5+ch7_5+ch8_3) and ch6_5 alone. |

</details>

<details><summary><code>render_reports/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `render_reports__from-code.py` | Render METHODS/RESULTS/LIMITATIONS/FINAL_ANALYSIS_REPORT + CLAIM_EVIDENCE_MATRIX |
| `render_reports__from-scripts.py` | Render METHODS/RESULTS/LIMITATIONS/FINAL_ANALYSIS_REPORT + CLAIM_EVIDENCE_MATRIX |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `make_figures.py` | Generate publication figures (§13) as SVG + 300 dpi PNG. Each function is |
| `plot_centroids.py` | Fit-free centroid (Delta-lambda) dose curves, GT15-STMN2 vs control, water. |
| `plot_chirality_byplate_exp2.py` | Per-chirality fitted dose-response, broken out by time point x matrix (4 plates). |
| `plot_global_spectra.py` | Global emission spectrum (intensity vs emission wavelength) vs STMN2-CE concentration. |
| `plot_serum_dose.py` | Serum dose-response figures, columns mapped to concentration and ordered by dose. |
