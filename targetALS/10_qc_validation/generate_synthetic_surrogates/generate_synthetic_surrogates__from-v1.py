"""
Generate synthetic surrogate samples with batch effects for spectroscopic data augmentation
Author: Postdoc researcher, Fallini Lab - URI Neuroscience
Purpose: Data augmentation for ALS ML classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.ndimage import shift as scipy_shift
from scipy.interpolate import interp1d

class BatchEffectGenerator:
    """Generate synthetic samples with realistic batch effects for fluorescence spectroscopy"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
    def add_baseline_shift(self, data, intensity=None):
        """Add constant baseline offset (vertical shift)"""
        if intensity is None:
            # Random baseline shift between -5% and +5% of median signal
            intensity = np.random.uniform(-0.05, 0.05) * np.median(np.abs(data))
        return data + intensity
    
    def add_intensity_scaling(self, data, scale=None):
        """Apply multiplicative intensity scaling"""
        if scale is None:
            # Random scaling between 0.85 and 1.15
            scale = np.random.uniform(0.85, 1.15)
        return data * scale
    
    def add_wavelength_shift(self, data, emission_wavelengths, shift_nm=None):
        """Apply small wavelength shift (simulates calibration drift)"""
        if shift_nm is None:
            # Random shift between -2 and +2 nm
            shift_nm = np.random.uniform(-2, 2)
        
        # Calculate shift in pixel units
        wavelength_step = np.mean(np.diff(emission_wavelengths))
        shift_pixels = shift_nm / wavelength_step
        
        # Apply shift to each excitation column
        shifted_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            shifted_data[:, i] = scipy_shift(data[:, i], shift_pixels, mode='nearest')
        
        return shifted_data
    
    def add_gaussian_noise(self, data, noise_level=None):
        """Add Gaussian noise"""
        if noise_level is None:
            # Noise level between 0.5% and 2% of signal std
            noise_level = np.random.uniform(0.005, 0.02) * np.std(data)
        
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    def add_baseline_drift(self, data, drift_magnitude=None):
        """Add non-linear baseline drift (wavelength-dependent)"""
        if drift_magnitude is None:
            # Random drift magnitude
            drift_magnitude = np.random.uniform(0.01, 0.05) * np.median(np.abs(data))
        
        # Create wavelength-dependent drift
        n_wavelengths = data.shape[0]
        x = np.linspace(0, 1, n_wavelengths)
        
        # Random polynomial drift (quadratic or cubic)
        if np.random.rand() > 0.5:
            drift = drift_magnitude * (x**2 - 0.5)
        else:
            drift = drift_magnitude * (x**3 - 0.5*x)
        
        # Apply drift to all excitation wavelengths
        drift_matrix = np.tile(drift.reshape(-1, 1), (1, data.shape[1]))
        return data + drift_matrix
    
    def add_excitation_intensity_variation(self, data, variation=None):
        """Simulate excitation source intensity variations across wavelengths"""
        if variation is None:
            variation = np.random.uniform(0.02, 0.08)
        
        n_excitations = data.shape[1]
        # Create smooth variation across excitation wavelengths
        x = np.linspace(0, 2*np.pi, n_excitations)
        intensity_profile = 1 + variation * np.sin(x + np.random.uniform(0, 2*np.pi))
        
        # Apply to data
        return data * intensity_profile
    
    def generate_surrogate(self, original_data, emission_wavelengths, 
                          batch_effect_type='all', intensity='random'):
        """
        Generate synthetic surrogate with specified batch effects
        
        Parameters:
        -----------
        original_data : numpy array
            Original spectroscopic data
        emission_wavelengths : numpy array
            Emission wavelength values
        batch_effect_type : str or list
            'all', 'mild', 'moderate', 'severe', or list of specific effects
        intensity : str
            'random', 'mild', 'moderate', 'severe'
        """
        surrogate = original_data.copy()
        
        # Define effect combinations
        if batch_effect_type == 'all' or batch_effect_type == 'moderate':
            effects = ['baseline_shift', 'intensity_scaling', 'wavelength_shift', 
                      'gaussian_noise', 'baseline_drift']
        elif batch_effect_type == 'mild':
            effects = np.random.choice(['baseline_shift', 'intensity_scaling', 
                                       'gaussian_noise'], size=2, replace=False)
        elif batch_effect_type == 'severe':
            effects = ['baseline_shift', 'intensity_scaling', 'wavelength_shift', 
                      'gaussian_noise', 'baseline_drift', 'excitation_variation']
        elif isinstance(batch_effect_type, list):
            effects = batch_effect_type
        else:
            effects = [batch_effect_type]
        
        # Apply effects
        for effect in effects:
            if effect == 'baseline_shift':
                surrogate = self.add_baseline_shift(surrogate)
            elif effect == 'intensity_scaling':
                surrogate = self.add_intensity_scaling(surrogate)
            elif effect == 'wavelength_shift':
                surrogate = self.add_wavelength_shift(surrogate, emission_wavelengths)
            elif effect == 'gaussian_noise':
                surrogate = self.add_gaussian_noise(surrogate)
            elif effect == 'baseline_drift':
                surrogate = self.add_baseline_drift(surrogate)
            elif effect == 'excitation_variation':
                surrogate = self.add_excitation_intensity_variation(surrogate)
        
        return surrogate


def process_all_samples(input_dir, output_dir, n_surrogates_per_sample=5):
    """
    Process all samples and generate synthetic surrogates
    
    Parameters:
    -----------
    input_dir : str
        Directory containing original samples
    output_dir : str
        Directory to save synthetic surrogates
    n_surrogates_per_sample : int
        Number of synthetic samples to generate per original sample
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample labels
    labels_df = pd.read_csv(os.path.join(input_dir, 'sample_labels.csv'))
    
    # Initialize batch effect generator
    batch_gen = BatchEffectGenerator()
    
    # Storage for new labels
    new_labels = []
    
    # Batch effect types to cycle through
    batch_types = ['mild', 'moderate', 'severe', 'all', 
                   ['baseline_shift', 'gaussian_noise'],
                   ['intensity_scaling', 'wavelength_shift'],
                   ['baseline_drift', 'excitation_variation']]
    
    print(f"Processing {len(labels_df)} original samples...")
    print(f"Generating {n_surrogates_per_sample} surrogates per sample")
    print(f"Total synthetic samples: {len(labels_df) * n_surrogates_per_sample}\n")
    
    for idx, row in labels_df.iterrows():
        # Get filename (fix naming mismatch)
        original_code = row['code']
        # Convert from dot notation to underscore notation
        filename = original_code.replace('.3d__570.opj_with_emission', 
                                        '_3d__570_opj_with_emission.xlsx')
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        # Load original data
        df_original = pd.read_excel(filepath)
        emission_wavelengths = df_original['Emission'].values
        
        # Extract spectral data (all columns except Emission)
        spectral_cols = [col for col in df_original.columns if col != 'Emission']
        spectral_data = df_original[spectral_cols].values
        
        print(f"Processing {original_code} ({row['group']})...")
        
        # Generate surrogates
        for surrogate_idx in range(n_surrogates_per_sample):
            # Select batch effect type
            batch_type = batch_types[surrogate_idx % len(batch_types)]
            
            # Generate surrogate
            surrogate_data = batch_gen.generate_surrogate(
                spectral_data, 
                emission_wavelengths,
                batch_effect_type=batch_type
            )
            
            # Create new DataFrame
            df_surrogate = pd.DataFrame(surrogate_data, columns=spectral_cols)
            df_surrogate.insert(0, 'Emission', emission_wavelengths)
            
            # Save to Excel
            sample_num = original_code.split('.')[0]
            output_filename = f"{sample_num}_surrogate_{surrogate_idx+1:02d}_3d__570_opj_with_emission.xlsx"
            output_path = os.path.join(output_dir, output_filename)
            df_surrogate.to_excel(output_path, index=False)
            
            # Add to labels
            new_code = output_filename.replace('_3d__570_opj_with_emission.xlsx', 
                                              '.3d__570.opj_with_emission')
            new_labels.append({
                'code': new_code,
                'group': row['group'],
                'original_sample': sample_num,
                'surrogate_number': surrogate_idx + 1,
                'batch_effect_type': str(batch_type)
            })
        
        print(f"  → Generated {n_surrogates_per_sample} surrogates")
    
    # Save augmented labels
    labels_augmented = pd.DataFrame(new_labels)
    labels_augmented.to_csv(os.path.join(output_dir, 'sample_labels_augmented.csv'), 
                           index=False)
    
    # Also create combined labels (original + synthetic)
    original_labels_fixed = labels_df.copy()
    original_labels_fixed['original_sample'] = original_labels_fixed['code'].str.split('.').str[0]
    original_labels_fixed['surrogate_number'] = 0
    original_labels_fixed['batch_effect_type'] = 'original'
    
    combined_labels = pd.concat([original_labels_fixed, labels_augmented], 
                                ignore_index=True)
    combined_labels.to_csv(os.path.join(output_dir, 'sample_labels_combined.csv'), 
                          index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  Original samples: {len(labels_df)}")
    print(f"  Synthetic samples: {len(new_labels)}")
    print(f"  Total samples: {len(combined_labels)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Print summary statistics
    print("Group distribution (synthetic samples):")
    print(labels_augmented['group'].value_counts())
    print(f"\n{labels_augmented['batch_effect_type'].value_counts()}")
    
    return labels_augmented, combined_labels


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "/mnt/project"
    OUTPUT_DIR = "/mnt/user-data/outputs/synthetic_surrogates"
    N_SURROGATES = 5  # Generate 5 surrogates per original sample
    
    # Run processing
    labels_aug, labels_combined = process_all_samples(
        INPUT_DIR, 
        OUTPUT_DIR, 
        n_surrogates_per_sample=N_SURROGATES
    )
    
    print("\n✓ All synthetic surrogates generated successfully!")
    print(f"✓ Check output directory: {OUTPUT_DIR}")
