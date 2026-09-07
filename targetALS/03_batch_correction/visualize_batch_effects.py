"""
Visualize batch effects in synthetic surrogate samples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def visualize_batch_effects(input_dir, output_dir, sample_num='1'):
    """
    Visualize original vs synthetic samples with batch effects
    """
    
    # Load original sample
    original_file = f"{sample_num}_3d__570_opj_with_emission.xlsx"
    df_original = pd.read_excel(f"{input_dir}/{original_file}")
    
    # Load surrogates
    surrogates = []
    batch_types = []
    
    # Get batch types from labels
    labels = pd.read_csv(f"{output_dir}/sample_labels_augmented.csv")
    labels['original_sample'] = labels['original_sample'].astype(str)
    sample_surrogates = labels[labels['original_sample'] == str(sample_num)].copy()
    
    for i in range(1, 6):
        surrogate_file = f"{sample_num}_surrogate_{i:02d}_3d__570_opj_with_emission.xlsx"
        df_surrogate = pd.read_excel(f"{output_dir}/{surrogate_file}")
        surrogates.append(df_surrogate)
        
        # Get batch type for this surrogate
        surr_row = sample_surrogates[sample_surrogates['surrogate_number'] == i]
        if len(surr_row) > 0:
            batch_type = surr_row['batch_effect_type'].values[0]
        else:
            batch_type = "unknown"
        batch_types.append(str(batch_type))
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Batch Effect Comparison: Sample {sample_num} (Original vs Synthetic Surrogates)', 
                 fontsize=14, fontweight='bold')
    
    # Get emission wavelengths
    emission = df_original['Emission'].values
    
    # Plot original in first subplot
    ax = axes[0, 0]
    spectral_cols = [col for col in df_original.columns if col != 'Emission']
    original_data = df_original[spectral_cols].values
    
    # Plot a few representative excitation wavelengths
    ex_wavelengths_to_plot = [10, 20, 30, 40, 50]  # Indices
    colors = plt.cm.viridis(np.linspace(0, 1, len(ex_wavelengths_to_plot)))
    
    for idx, ex_idx in enumerate(ex_wavelengths_to_plot):
        ax.plot(emission, original_data[:, ex_idx], 
               color=colors[idx], alpha=0.7, linewidth=1.5,
               label=f'Ex: {spectral_cols[ex_idx].split("_")[1][:3]} nm')
    
    ax.set_title('ORIGINAL', fontweight='bold', fontsize=12)
    ax.set_xlabel('Emission wavelength (nm)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot surrogates
    for idx, (df_surr, batch_type) in enumerate(zip(surrogates, batch_types)):
        ax = axes[(idx+1)//2, (idx+1)%2]
        surrogate_data = df_surr[spectral_cols].values
        
        for ex_idx_plot, ex_idx in enumerate(ex_wavelengths_to_plot):
            ax.plot(emission, surrogate_data[:, ex_idx], 
                   color=colors[ex_idx_plot], alpha=0.7, linewidth=1.5,
                   label=f'Ex: {spectral_cols[ex_idx].split("_")[1][:3]} nm')
        
        ax.set_title(f'SURROGATE {idx+1}: {batch_type}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Emission wavelength (nm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_effects_visualization_sample_{sample_num}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: batch_effects_visualization_sample_{sample_num}.png")
    
    # Create difference plots
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(f'Batch Effect Differences: Sample {sample_num} (Surrogate - Original)', 
                  fontsize=14, fontweight='bold')
    
    for idx, (df_surr, batch_type) in enumerate(zip(surrogates, batch_types)):
        ax = axes2[idx//3, idx%3]
        
        surrogate_data = df_surr[spectral_cols].values
        difference = surrogate_data - original_data
        
        # Create 2D heatmap of differences
        im = ax.imshow(difference.T, aspect='auto', cmap='RdBu_r', 
                      extent=[emission[0], emission[-1], 0, len(spectral_cols)])
        ax.set_title(f'Surrogate {idx+1}: {batch_type}', fontsize=10)
        ax.set_xlabel('Emission wavelength (nm)')
        ax.set_ylabel('Excitation index')
        plt.colorbar(im, ax=ax, label='Intensity difference')
    
    # Hide the 6th subplot
    axes2[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_effects_differences_sample_{sample_num}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved difference plot: batch_effects_differences_sample_{sample_num}.png")
    
    return fig, fig2


def create_summary_statistics(output_dir):
    """Create summary statistics of the augmented dataset"""
    
    labels_aug = pd.read_csv(f"{output_dir}/sample_labels_augmented.csv")
    labels_combined = pd.read_csv(f"{output_dir}/sample_labels_combined.csv")
    
    print("\n" + "="*60)
    print("DATASET AUGMENTATION SUMMARY")
    print("="*60)
    
    print("\nOriginal dataset:")
    print(f"  Total samples: 39")
    print(f"  ALS: 20 samples")
    print(f"  CTRL: 19 samples")
    
    print("\nAugmented dataset (synthetic only):")
    print(f"  Total samples: {len(labels_aug)}")
    print("\nBy group:")
    print(labels_aug['group'].value_counts())
    
    print("\nBy batch effect type:")
    print(labels_aug['batch_effect_type'].value_counts())
    
    print("\nCombined dataset (original + synthetic):")
    print(f"  Total samples: {len(labels_combined)}")
    print("\nBy group:")
    print(labels_combined['group'].value_counts())
    
    print("\n" + "="*60)
    
    # Create distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Group distribution
    labels_combined['group'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#3498db'])
    axes[0].set_title('Group Distribution (Original + Synthetic)', fontweight='bold')
    axes[0].set_xlabel('Group')
    axes[0].set_ylabel('Number of samples')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Plot 2: Batch effect type distribution
    batch_counts = labels_aug['batch_effect_type'].value_counts()
    axes[1].barh(range(len(batch_counts)), batch_counts.values)
    axes[1].set_yticks(range(len(batch_counts)))
    axes[1].set_yticklabels(batch_counts.index, fontsize=9)
    axes[1].set_title('Batch Effect Type Distribution', fontweight='bold')
    axes[1].set_xlabel('Number of samples')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_summary.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved summary plot: dataset_summary.png")
    
    return fig


if __name__ == "__main__":
    INPUT_DIR = "/mnt/project"
    OUTPUT_DIR = "/mnt/user-data/outputs/synthetic_surrogates"
    
    # Visualize batch effects for first ALS and CTRL samples
    print("Generating visualizations...")
    visualize_batch_effects(INPUT_DIR, OUTPUT_DIR, sample_num='1')  # ALS
    visualize_batch_effects(INPUT_DIR, OUTPUT_DIR, sample_num='10')  # CTRL
    
    # Create summary statistics
    create_summary_statistics(OUTPUT_DIR)
    
    print("\n✓ All visualizations complete!")
