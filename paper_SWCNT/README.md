# paper_SWCNT

Analysis code accompanying the SWCNT ALS biomarker manuscript

**11 scripts** across **2 analysis types**.

## Analysis types

| Folder | Scripts | What it covers |
|---|---|---|
| [`01_preprocessing/`](./01_preprocessing/) | 5 | Data ingestion, renaming, cropping, cache building and system/structure setup |
| [`04_autoencoder_latents/`](./04_autoencoder_latents/) | 6 | Convolutional autoencoders, latent extraction and deterministic proxy-latent pipelines |

## 01_preprocessing

Data ingestion, renaming, cropping, cache building and system/structure setup

### Related script variants

<details><summary><code>add_emission_column/</code> &mdash; 2 variants</summary>

| File | Notes |
|---|---|
| `add_emission_column__from-ex123.py` | === 1. Folder with your *_e1.xlsx files === |
| `add_emission_column__from-ex4.py` | === 1. Folder with your *_e1.xlsx files === |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `delate1400.py` | === 1. Folder with your Excel files === |
| `delete805.py` | === 1. Folder with your Excel files === |
| `rename_and_make_labels.py` | === 1. Set your folder path here === |

## 04_autoencoder_latents

Convolutional autoencoders, latent extraction and deterministic proxy-latent pipelines

### Versioned script families

<details><summary><code>conv_autoencoder_detailed/</code> &mdash; 4 versions</summary>

| Version | File | Notes |
|---|---|---|
| v0 | `v0_conv_autoencoder_detailed.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v0 | `v0_conv_autoencoder_detailed_fixed_metrics.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v2 | `v2_conv_autoencoder_detailed_fixed_metrics_v2.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
| v3 | `v3_conv_autoencoder_detailed_fixed_with_metrics_v3.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |

</details>

### Standalone scripts

| File | Notes |
|---|---|
| `consensus_analysis_adapted_CTRL0_ALS1_v2_export_latents.py` | consensus_analysis_adapted.py |
| `conv_autoencoder_ALSFRS_regression.py` | Convolutional Autoencoder with Attention-Based Temporal Aggregation |
