# targetALS

Target ALS biofluid SWCNT spectroscopy & ML biomarker discovery

**180 scripts** across **10 analysis types**.

## Analysis types

| Folder | Scripts | What it covers |
|---|---|---|
| [`01_preprocessing/`](./01_preprocessing/) | 13 | Data ingestion, renaming, cropping, cache building and system/structure setup |
| [`03_batch_correction/`](./03_batch_correction/) | 9 | EEM batch-effect correction, reference calibration and detrending |
| [`04_autoencoder_latents/`](./04_autoencoder_latents/) | 38 | Convolutional autoencoders, latent extraction and deterministic proxy-latent pipelines |
| [`05_feature_selection_importance/`](./05_feature_selection_importance/) | 19 | Recursive feature selection, saliency, permutation and latent-dimension importance |
| [`06_classification/`](./06_classification/) | 20 | Supervised ALS-vs-control classifiers (SVM, PLS-DA, RF), thresholds and deployment |
| [`07_regression/`](./07_regression/) | 13 | ALSFRS score / slope regression and PLS regression models |
| [`08_statistics/`](./08_statistics/) | 28 | PERMANOVA, PERMDISP, linear mixed models, ANOVA, confounder and dependence testing |
| [`09_dimensionality_reduction/`](./09_dimensionality_reduction/) | 9 | PCA, UMAP and latent-dimension convergence analyses |
| [`10_qc_validation/`](./10_qc_validation/) | 15 | Quality control, technical controls, surrogate generation and pipeline smoke tests |
| [`13_figures/`](./13_figures/) | 16 | Publication and diagnostic figure generation |

## 01_preprocessing

Data ingestion, renaming, cropping, cache building and system/structure setup

### Related script variants

<details><summary><code>add_emission_column/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `add_emission_column.py` | Add an 'Emission' column (as the FIRST column) to one or many CSV/XLSX files. |
| `add_emission_column_py39.py` | Add an 'Emission' column (as the FIRST column) to one or many CSV/XLSX files. |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `build_cache.py` | Cache the EEM tensor used by the conv-autoencoder into a single npz so the |
| `combine_renamed.py` | 1. Search for renamed files |
| `crop_eem_wavelengths_to_new_root.py` | Crop NS Spectralyzer-style EEM Excel exports by wavelength and SAVE to a separate output root, |
| `debug_mapping.py` |  |
| `delate1400.py` | === 1. Folder with your Excel files === |
| `extract_controls.py` | Extract 12-chirality peak features from the PBS (P) and FBS (F) technical |
| `extract_kinetics.py` | Extract 12-chirality features (max5x5 + gauss_max) from the dense corona-kinetics |
| `fetch_gene_symbols.py` | Map every UniProt accession in the three Welch result files to a gene symbol. |
| `identify_chirality.py` | --- CONFIG --- |
| `organize.py` | Define the target directory |
| `z_h_rename.py` | Ensure the folder path exists |

## 03_batch_correction

EEM batch-effect correction, reference calibration and detrending

### Versioned script families

<details><summary><code>calibrate_eem_to_reference/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_calibrate_eem_to_reference.py` | calibrate_eem_to_reference.py |
| v2 | `v2_calibrate_eem_to_reference_v2.py` | calibrate_eem_to_reference_v2.py |

</details>

<details><summary><code>force_correction/</code> &mdash; 3 versions</summary>

| Version | File | Notes |
|---|---|---|
| v4 | `v4_force_correction_v4.py` | ================= HELPER FUNCTIONS ================= |
| v5 | `v5_force_correction_v5_take_closest_or_autopeaks.py` | force_correction_v5_take_closest_or_autopeaks.py |
| v6 | `v6_force_correction_v6_auto_invert.py` | force_correction_v6_auto_invert.py |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `diagnose_calibration_failure.py` | diagnose_calibration_failure.py |
| `eem_batch_correct_from_folders_xlsx_FIXED_v6_coords_only_rangeaware.py` | eem_batch_correct_from_folders_xlsx_FIXED_v4_coords_only_rangeaware.py |
| `eem_batch_correct_from_folders_xlsx_FIXED_v7_autopick_closest_peaks.py` | eem_batch_correct_from_folders_xlsx_FIXED_v7_autopick_closest_peaks.py |
| `visualize_batch_effects.py` | Visualize batch effects in synthetic surrogate samples |

## 04_autoencoder_latents

Convolutional autoencoders, latent extraction and deterministic proxy-latent pipelines

### Versioned script families

<details><summary><code>conv_autoencoder_chirality/</code> &mdash; 3 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_conv_autoencoder_chirality.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v3 | `v3_conv_autoencoder_chirality_informed_tp_anyxlsx_v3.py` | Chirality-Informed Convolutional Autoencoder with Temporal Attention |
| v5 | `v5_conv_autoencoder_chirality_informed_tp_anyxlsx_v5_oof_perm_nested.py` | Chirality-Informed Convolutional Autoencoder with Temporal Attention |

</details>

<details><summary><code>conv_autoencoder_detailed/</code> &mdash; 6 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_conv_autoencoder_detailed__from-cropped-em920-ex520-815.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v0 | `v0_conv_autoencoder_detailed__from-exp5.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v0 | `v0_conv_autoencoder_detailed_fixed.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v0 | `v0_conv_autoencoder_detailed_updated.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v2 | `v2_conv_autoencoder_detailed_cropped_paths_only_v2.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v3 | `v3_conv_autoencoder_detailed_cropped_paths_only_v3.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |

</details>

<details><summary><code>multiscale_chirality_pipeline/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_multiscale_chirality_pipeline.py` | MULTI-SCALE CHIRALITY-ANCHORED FEATURE EXTRACTION PIPELINE |
| v3 | `v3_multiscale_chirality_pipeline_v3_no_heights.py` | MULTI-SCALE CHIRALITY-ANCHORED FEATURE EXTRACTION PIPELINE (v3 - No Heights) |

</details>

<details><summary><code>step2_encoder_weight_proxy/</code> &mdash; 4 versions</summary>

| Version | File | Notes |
|---|---|---|
| v3.2 | `v3.2_step2_encoder_weight_proxy_v3.2.py` | STEP 2 v3: Encoder-Weight-Based Proxy Latent Pipeline |
| v3.3 | `v3.3_step2_encoder_weight_proxy_v3.3.py` | STEP 2 v3: Encoder-Weight-Based Proxy Latent Pipeline |
| v3.4 | `v3.4_step2_encoder_weight_proxy_v3.4.py` | STEP 2 v3: Encoder-Weight-Based Proxy Latent Pipeline |
| v3 | `v3_step2_encoder_weight_proxy_v3.py` | STEP 2 v3: Encoder-Weight-Based Proxy Latent Pipeline |

</details>

<details><summary><code>step3_targeted_rois/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_step3_targeted_rois.py` | STEP 3 (TARGETED): Pixel Attribution & ROI Extraction |
| v2 | `v2_step3_targeted_rois_v2.py` | STEP 3 v2 (TARGETED): Pixel Attribution & ROI Extraction |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `apply_proxy_latents_batch_corrected.py` | apply_proxy_latents_batch_corrected.py |
| `apply_proxy_latents_to_new_dataset_v2.py` | apply_proxy_latents_to_new_dataset.py |
| `chirality_autoencoder_feature_selection.py` | CHIRALITY-INFORMED AUTOENCODER FOR FEATURE SELECTION |
| `chirality_direct_pipeline.py` | chirality_direct_pipeline.py |
| `chirality_feature_pipeline.py` | Chirality Descriptor Feature Extraction Pipeline |
| `conv_autoencoder_longitudinal_v2.py` | Convolutional Autoencoder with Longitudinal LSTM (Trajectory Analysis) |
| `conv_autoencoder_regress_alsfrs_slope_v5_stability_heldout.py` | Convolutional Autoencoder + Snapshot Attention + Trajectory Encoder (GRU) for ALSFRS-slope regression |
| `export_fullfit_ridge_proxy_params.py` |  |
| `export_v12_exact_fullfit.py` |  |
| `fused_pipeline_opt_percentile.py` | FUSED PIPELINE: Optimal Percentile Search (Targeted ROIs + Proxy Classification) |
| `hybrid_chirality_autoencoder.py` | HYBRID CHIRALITY-FOCUSED AUTOENCODER + ML PIPELINE |
| `proxy_latent_inference.py` | Proxy Latent Inference Engine (Strategy E) |
| `proxy_latent_pipeline_v13_leakfree.py` | proxy_latent_pipeline_v13_leakfree.py |
| `proxy_latent_pipeline_v14_power_mi.py` | proxy_latent_pipeline_v14_power_mi.py |
| `step1_extract_and_rank_latent_features.py` | STEP 1: Extract Latent Features from 5-Fold Pre-trained Models & Rank by Importance |
| `step2_compute_mse_maps.py` | STEP 2: Compute Per-Pixel Reconstruction MSE (averaged across 5 models) |
| `step3_4_fused_proxy_pipeline.py` | STEP 3+4 FUSED: ROI Extraction + Proxy Reconstruction with Percentile Optimization |
| `step4_final_proxy_v2.py` | STEP 4 v2 (FINAL): Targeted Two-Stage Proxy Reconstruction |
| `step4_final_top5_proxy.py` | STEP 4 (FINAL - CORRECTED): Targeted Two-Stage Proxy Reconstruction (Top 5 Features) |
| `triangulation2_ae.py` | Where does the conv-AE disease signal live? |
| `triangulation_disease.py` | Triangulation: is the ALS-vs-CTRL signal carried by the DRIFT relative-movement |

## 05_feature_selection_importance

Recursive feature selection, saliency, permutation and latent-dimension importance

### Versioned script families

<details><summary><code>step_opt_latent_selection_auc/</code> &mdash; 4 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_step_opt_latent_selection_auc.py` | STEP OPT: Optimal Latent Selection (AUC Ranking + MI Redundancy) |
| v2 | `v2_step_opt_latent_selection_auc_v2.py` | STEP OPT v2: Optimal Latent Selection (Leak-Free AUC + Spearman Redundancy + Binomial Power) |
| v3 | `v3_step_opt_latent_selection_auc_v3.py` | STEP OPT v3: Optimal Latent Selection (Leak-Free AUC + AUC-Monotonicity Stop) |
| v3 | `v3_step_opt_latent_selection_auc_v3_strict.py` | STEP OPT v3 (CORRECTED): Optimal Latent Selection (Strict Nested Leak-Free AUC) |

</details>

### Related script variants

<details><summary><code>chirality_interpolated_extraction/</code> &mdash; 3 variants</summary>

| File | Notes |
|---|---|
| `chirality_interpolated_extraction.py` | chirality_interpolated_extraction.py |
| `chirality_interpolated_extraction_o.py` | chirality_interpolated_extraction.py |
| `chirality_interpolated_extraction_oo.py` | chirality_interpolated_extraction.py |

</details>

<details><summary><code>recursive_loocv_feature_selection/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `recursive_loocv_feature_selection.py` | recursive_loocv_feature_selection.py |
| `recursive_loocv_feature_selection_corrected.py` | recursive_loocv_feature_selection_corrected.py |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `analyze_importance_at_chiralities.py` | Analyze importance maps at chirality positions vs neighboring regions. |
| `chirality_descriptor_extraction.py` | chirality_descriptor_extraction.py |
| `chirality_dim35_convergence.py` | Chirality–Latent Dim35 Convergence Analysis |
| `chirality_importance_test.py` | Test whether chirality positions have significantly higher importance scores |
| `chirality_roi_fusion.py` | chirality_roi_fusion.py |
| `dim35_chirality_saliency.py` | dim35-Specific Gradient Saliency at Chirality Positions |
| `redundancy_check.py` | Choose which extracted feature you want to use: |
| `timepoint_ensemble_discriminative_importance.py` | timepoint_ensemble_discriminative_importance.py |
| `timepoint_importance_pretrained.py` | TIMEPOINT-SPECIFIC FEATURE IMPORTANCE PIPELINE (Using Pre-trained Models) |
| `timepoint_importance_recon_filtered.py` | TIMEPOINT-SPECIFIC FEATURE IMPORTANCE PIPELINE (Pre-trained Models) |

## 06_classification

Supervised ALS-vs-control classifiers (SVM, PLS-DA, RF), thresholds and deployment

### Versioned script families

<details><summary><code>chirality_classification_pipeline/</code> &mdash; 5 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_chirality_classification_pipeline.py` | Chirality Descriptor Classification Pipeline |
| v1 | `v1_chirality_classification_pipeline_v1_fixed.py` | Chirality Descriptor Classification Pipeline |
| v2 | `v2_chirality_classification_pipeline_v2.py` | Chirality Descriptor Classification Pipeline — V2 (Improved) |
| v2 | `v2_chirality_classification_pipeline_v2_nodeltas.py` | Chirality Descriptor Classification Pipeline — V2 (Improved) |
| v3 | `v3_chirality_classification_pipeline_v3_enet.py` | Chirality Descriptor Classification Pipeline — V3 (Elastic Net Selection) |

</details>

### Related script variants

<details><summary><code>rerun_faithful/</code> &mdash; 3 variants</summary>

| File | Notes |
|---|---|
| `rerun_faithful.py` | Faithful re-run of the ORIGINAL training procedure that produced |
| `rerun_faithful_noC9.py` | NO-C9 variant: identical faithful re-run of the ORIGINAL training procedure, |
| `rerun_faithful_noC9_matched.py` | MATCHED-FOLD no-C9 variant. Isolates the effect of EXCLUDING the two C9orf72 |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `age_matched_retrain.py` | AGE-MATCHED RETRAIN  -- the rigorous age-leakage test. |
| `calibrate_threshold.py` | --- LOAD YOUR RESULTS --- |
| `deploy_biomarker_v12_exact.py` |  |
| `eval_5fold_metrics.py` | Evaluate the saved 5-fold autoencoder+classifier models in 5_fold_models_original/ |
| `plsda_baseline.py` | PLS-DA (and companion linear chemometric) baselines for ALS-vs-CTRL from EEM, |
| `predict_exp7_with_exp5_models_v7_lazy_init.py` | predict_exp7_with_exp5_models_v7_lazy_init.py |
| `predict_exp7_with_exp5_models_v7b_posindex_autoflip.py` | predict_exp7_with_exp5_models_v7b_posindex_autoflip.py |
| `rerun_train_5fold.py` | Part 2 — Controlled re-run of the 5-fold autoencoder+classifier training. |
| `roi_classification_pipeline.py` | ROI-BASED CLASSIFICATION PIPELINE |
| `score_new_samples_with_svm_params.py` | Score new samples using exported linear SVM parameters (weights/intercept + thresholds). |
| `train_linear_svm_on_proxy_logo_ALS_CTRL_with_jpegs.py` | LOOCV Linear SVM (LinearSVC) on proxy latents (proxy_latents.csv) with: |
| `umap_rf.py` |  |

## 07_regression

ALSFRS score / slope regression and PLS regression models

### Versioned script families

<details><summary><code>ALSFRS_Score_Analysis/</code> &mdash; 4 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_ALSFRS_Score_Analysis.py` | 1. LOAD DATA |
| v0 | `v0_ALSFRS_Score_Analysis_reg.py` | 1. LOAD DATA |
| v2 | `v2_ALSFRS_Score_Analysis2.py` | 1. LOAD DATA |
| v3 | `v3_ALSFRS_Score_Analysis3.py` | 1. LOAD DATA |

</details>

<details><summary><code>ALSFRS_Slope_Analysis/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_ALSFRS_Slope_Analysis.py` | 1. LOAD DATA |
| v3 | `v3_ALSFRS_Slope_Analysis3.py` | 1. LOAD DATA |

</details>

### Related script variants

<details><summary><code>PLS/</code> &mdash; 3 variants</summary>

| File | Notes |
|---|---|
| `PLS.py` | 1. LOAD DATA |
| `PLS_abs_pred.py` | 1. LOAD DATA |
| `PLS_ca.py` | 1. LOAD DATA |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `ALSFRS_Score_Analysis2_fdr.py` | 1. LOAD DATA |
| `chirality_regression.py` | ALSFRS / ALSFRS-slope Chirality Regression Pipeline (No PyTorch Required) |
| `dim477_nonpair_pls_analysis.py` |  |
| `dim477_ridge_diagnose.py` | Diagnose the sharp red (ALS>CTRL) ridges in the difference surface: |

## 08_statistics

PERMANOVA, PERMDISP, linear mixed models, ANOVA, confounder and dependence testing

### Versioned script families

<details><summary><code>interdependence_pipeline_xlsx/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_interdependence_pipeline_xlsx.py` | interdependence_pipeline_xlsx.py |
| v2 | `v2_interdependence_pipeline_xlsx_v2.py` | interdependence_pipeline_xlsx_v2.py |

</details>

<details><summary><code>lmm_lr_tests/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_lmm_lr_tests.py` | Inputs |
| v2 | `v2_lmm_lr_tests2.py` | Inputs |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `age_sex_confound.py` | Assess age & sex confounding on (a) the AE latent representation broadly and |
| `all_confounders.py` | Systematic confounder screen for the ALS-vs-CTRL corona signal. |
| `c9_alsaxis.py` | Across ALL latent features: do the C9+ controls project toward ALS more than |
| `c9_fold3_why.py` | WHY does fold3 (uniquely) flag C9 control 10 as ALS-like? |
| `c9_phenotype.py` | Do the two C9orf72+ controls (codes 10, 30) show an ALS phenotype in the models? |
| `c9_significance.py` | Are the C9orf72+ controls (10, 30) significantly more ALS-like than other |
| `chirality_descriptor_analysis.py` | chirality_descriptor_analysis.py |
| `chirality_independence_pipeline.py` | USER SETTINGS |
| `dependance_an.py` | 1. CONFIGURATION |
| `dim477_chirality_grouped_enrichment.py` |  |
| `dim477_nfl_misdx.py` | dim477 vs NfL — class separation & misdiagnosis check. |
| `dim477_nfl_oof_crosscheck.py` | Cross-check: where do the exp5 CNN OOF misclassifications land in dim477-NfL space? |
| `dim477_sig_emission.py` | (1) Find where the ALS vs CTRL difference in the raw normalized 24h EEM is |
| `dim_chirality_nfl_bridge.py` | Bridge analysis between latent dimensions, chirality descriptors, and NFL. |
| `intensity_covariance.py` | 1. CONFIGURATION |
| `normality_intensity_supp.py` | Normality assessment of SWCNT chirality peak-intensity features. |
| `permanova_distance_sensitivity.py` | Distance-metric sensitivity analysis for the timepoint PERMANOVA. |
| `permanova_full12_within_subject.py` | SETTINGS |
| `permanova_gaussfit_consistent.py` | Self-consistent PERMANOVA on the Gaussian-fit peak amplitudes + metric-robustness. |
| `permdisp_full12_within_subject.py` | SETTINGS |
| `residual.py` |  |
| `rm_anova_chirality_specific.py` | -------- SETTINGS -------- |
| `swcns_dependence_folders.py` | USER SETTINGS |
| `swcns_dependence_paired.py` | USER SETTINGS |

## 09_dimensionality_reduction

PCA, UMAP and latent-dimension convergence analyses

### Versioned script families

<details><summary><code>chirality_dim_convergence/</code> &mdash; 3 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_chirality_dim_convergence.py` | Chirality–Latent Dimension Convergence Analysis |
| v0 | `v0_chirality_dim_convergence_2_adapted.py` | Chirality–Latent Dimension Convergence Analysis |
| v2 | `v2_chirality_dim_convergence_2.py` | Chirality–Latent Dimension Convergence Analysis |

</details>

<details><summary><code>dim477_region_pilot/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_dim477_region_pilot.py` | PILOT: chirality-region reduction of the 24h dim477 reconstruction map. |
| v2 | `v2_dim477_region_pilot_v2.py` | PILOT v2 — region reduction of the 24h dim477 component, revised criteria. |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `delta_PCA.py` |  |
| `run_latent_clinical_corr_and_pca.py` | run_latent_clinical_corr_and_pca.py |
| `run_modelwise_union_corr_pca.py` | run_modelwise_union_corr_pca.py |
| `time_pca.py` |  |

## 10_qc_validation

Quality control, technical controls, surrogate generation and pipeline smoke tests

### Related script variants

<details><summary><code>generate_synthetic_surrogates/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `generate_synthetic_surrogates__from-targetals.py` | Generate synthetic surrogate samples with batch effects for spectroscopic data augmentation |
| `generate_synthetic_surrogates__from-v1.py` | Generate synthetic surrogate samples with batch effects for spectroscopic data augmentation |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `controls_analysis.py` | Technical-control analysis (A-D) for the reviewer comment: |
| `exp4_controls_analysis.py` | exp4 technical controls (FIT_controls_ex4.xlsx) — the strongest control set: |
| `kinetics_analysis.py` | Dense corona-kinetics analysis (7 tp x 3 conditions x 2 CNT conc, 1 EEM each). |
| `qc_ae_axis_angle.py` | How far off is the PBS->FBS axis from the CTRL->ALS axis in the conv-AE latent? |
| `qc_ae_controls.py` | Tier-3 QC in the autoencoder's own feature space. |
| `qc_ae_globalscale.py` | Does the per-image min-max normalisation hide the PBS/FBS contrast? |
| `qc_ae_pca_controls.py` | PCA of the conv-AE latent space computed on the TECHNICAL CONTROLS ONLY |
| `qc_ae_umap.py` | UMAP of the conv-AE latent space (z_agg), with the technical controls embedded |
| `qc_apply_exp5.py` | Apply the PBS-anchored QC framework (qc_metrics.py) retrospectively to the exp5 |
| `qc_metrics.py` | Assay QC framework anchored on the PBS negative control. |
| `qc_scheme_figure.py` | Schematic summary of the three-tier PBS-anchored QC framework, with the proposed |
| `size_matched_control.py` | SIZE-MATCHED CONTROL for the age-matched retrain. |
| `validate_multiplexing_ple.py` | USER SETTINGS |

## 13_figures

Publication and diagnostic figure generation

### Versioned script families

<details><summary><code>age_sex_confound_figs/</code> &mdash; 2 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_age_sex_confound_figs.py` | (a) age distribution by group |
| v2 | `v2_age_sex_confound_figs2.py` | Two SEPARATE confounding figures: |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `c9_phenotype_fig.py` | latent ALS-composite percentiles (from c9_phenotype.py stdout) |
| `chirality_trajectories_emm.py` | Settings |
| `dim477_eem_trace.py` | Trace dim477 back to EEM features for the two dim477-misclassified controls |
| `dim477_gif_only.py` | Regenerate only the edge-cropped rotating GIF, with downsampled surfaces |
| `dim477_group_maps.py` | Group-average dim477 spectral maps: ALS vs CTRL. |
| `dim477_surface_compare.py` | 3D surface comparison of the mean 24h EEM: ALS vs CTRL. |
| `dim477_surface_cropped.py` | Edge-cropped re-render of the ALS vs CTRL 24h EEM surfaces. |
| `dim477_surface_gif_drape.py` | (1) Rotating GIF of the overlaid ALS vs CTRL mean 24h EEM surfaces. |
| `emm_contrasts_plot.py` | Settings |
| `plot_existing_metrics.py` | Part 1 — Metrics + graphs for the EXISTING saved models in 5_fold_models_original/. |
| `plot_rerun_metrics.py` | Part 2 plotting — graphs for the controlled re-run (model_metrics_rerun/), |
| `residual_heatmap.py` | Folder containing your corr_residual_*.csv files |
| `step5_visualize_top5_rois.py` | STEP 5 (CORRECTED): Visualize Top 5 ROI Masks on Mean EEM Spectrum |
| `volcano_plots.py` | Publication-quality volcano plots for the ALS vs CTRL MS experiments. |
