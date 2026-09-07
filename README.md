# ML-nano — SWCNT nanosensor analysis code (ALS)

Analysis and machine-learning code for single-walled carbon nanotube (SWCNT)
optical nanosensor work on **amyotrophic lateral sclerosis (ALS)**: near-infrared
excitation–emission matrix (EEM) spectroscopy of DNA/protein-corona SWCNT sensors,
applied to patient biofluids and to STMN2 cryptic-exon detection.

Scripts are organised **project → type of analysis → version family**.

## Projects

| Project | Scripts | Focus |
|---|---|---|
| [`targetALS/`](./targetALS/) | 180 | Target ALS biofluid cohort: EEM batch correction, convolutional autoencoder latents, proxy-latent biomarker pipelines, ALS-vs-control classification and ALSFRS regression |
| [`cryptic_mRNA/`](./cryptic_mRNA/) | 97 | STMN2 cryptic-exon sensor assays: Gaussian peak fitting, dose–response and specificity readouts, plus SWCNT/oligo molecular modeling (docking, hybridization thermodynamics, MD) |
| [`paper_SWCNT/`](./paper_SWCNT/) | 11 | Analysis code accompanying the SWCNT ALS biomarker manuscript |

**288 unique scripts** in total.

## How this is organised

### 1. Project
Top-level directory, one per research project.

### 2. Type of analysis
Numbered directories inside each project, roughly following the order of a
processing pipeline:

| Folder | What it covers |
|---|---|
| `01_preprocessing` | Data ingestion, renaming, cropping, cache building, system/structure setup |
| `02_peak_fitting` | Gaussian / free-form spectral peak fitting and fit diagnostics |
| `03_batch_correction` | EEM batch-effect correction, reference calibration, detrending |
| `04_autoencoder_latents` | Convolutional autoencoders, latent extraction, deterministic proxy-latent pipelines |
| `05_feature_selection_importance` | Recursive feature selection, saliency, permutation and latent-dimension importance |
| `06_classification` | Supervised ALS-vs-control classifiers (SVM, PLS-DA, RF), thresholds, deployment |
| `07_regression` | ALSFRS score / slope regression and PLS regression |
| `08_statistics` | PERMANOVA, PERMDISP, linear mixed models, ANOVA, confounder and dependence testing |
| `09_dimensionality_reduction` | PCA, UMAP, latent-dimension convergence |
| `10_qc_validation` | Quality control, technical controls, surrogate generation, pipeline smoke tests |
| `11_response_analysis` | Dose–response, serum/water assay response, blinded/unblinded readouts |
| `12_molecular_modeling` | Docking, hybridization thermodynamics, conformer generation, MD-derived descriptors |
| `13_figures` | Publication and diagnostic figure generation |

Not every project uses every category.

### 3. Versions of the same script
Where a script went through several iterations, the iterations live together in a
folder named after the script, ordered by version:

```
06_classification/
  chirality_classification_pipeline/
    v0_chirality_classification_pipeline.py
    v1_chirality_classification_pipeline_v1_fixed.py
    v2_chirality_classification_pipeline_v2.py
    v2_chirality_classification_pipeline_v2_nodeltas.py
    v3_chirality_classification_pipeline_v3_enet.py
```

The **highest version number is the most recent** one. Original filenames are kept
after the `vN_` prefix so they stay traceable.

Some folders group **variants** rather than versions — scripts that differ by
experiment, sample or configuration rather than by revision (for example
`extract_gaussian/` holds one fitter per experiment). Those keep their original
filenames, with no `vN_` prefix.

Scripts with only one version sit directly in their analysis-type folder.

## Provenance — `MANIFEST.csv`

Every script maps back to where it came from. `MANIFEST.csv` has one row per file:

| Column | Meaning |
|---|---|
| `repo_path` | Location in this repository |
| `project` / `analysis_type` / `version_family` / `version` | Assigned classification |
| `original_path` | Path in the original working folder |
| `duplicate_copies` | Other original paths that held a byte-identical copy |
| `summary` | First descriptive line from the script |

61 byte-identical duplicates (the same script copied into several experiment
folders) were folded into a single copy; every original location is recorded in
`duplicate_copies`, so nothing is lost.

## Scope

Data files, model weights and results are **not** included; these are analysis
scripts only. Paths inside the scripts refer to the original local and HPC
working directories and will need to be updated before reuse.
