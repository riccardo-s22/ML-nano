import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_PATH = r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5"

TIME_POINTS = {
    '0h': 'out_0h',
    '6h': 'out_6h',
    '24h': 'out_24h'
}

# Chirality Centers
chirality_centers = {
    '6,5':  (573, 975),
    '7,5':  (647, 1024),
    '7,6':  (645, 1115),
    '8,3':  (667, 952),
    '8,4':  (726, 1100),
    '8,6':  (718, 1170),
    '8,7':  (726, 1260),
    '9,4':  (720, 1100),
    '9,5':  (800, 1240),
    '10,2': (740, 1050),
    '10,3': (800, 1100),
    '10,5': (850, 1250)
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_peak_intensity(filepath, centers):
    """
    Loads file, extracts TOTAL INTENSITY (Sum) of 5x5 region for each chirality.
    Returns a dictionary: {'6,5': 1024.5, '7,5': 500.2, ...}
    """
    try:
        df = pd.read_excel(filepath, index_col=0, engine='openpyxl')
        
        # Clean Axes
        ex_axis = df.columns.astype(str).str.replace('Excitation_', '').astype(float).to_numpy()
        em_axis = df.index.to_numpy(dtype=float)
        data = df.values.T # [Excitation x Emission]

        # Fix flipped axes
        if ex_axis[0] > ex_axis[-1]:
            ex_axis = ex_axis[::-1]
            data = data[::-1, :]
        if em_axis[0] > em_axis[-1]:
            em_axis = em_axis[::-1]
            data = data[:, ::-1]

        intensities = {}
        
        for name, (c_ex, c_em) in centers.items():
            # Find center index
            idx_ex = (np.abs(ex_axis - c_ex)).argmin()
            idx_em = (np.abs(em_axis - c_em)).argmin()
            
            # Extract 5x5 and SUM the intensity
            r_start, r_end = idx_ex - 2, idx_ex + 3
            c_start, c_end = idx_em - 2, idx_em + 3
            
            # Bounds check
            if r_start >= 0 and c_start >= 0 and r_end <= len(ex_axis) and c_end <= len(em_axis):
                roi = data[r_start:r_end, c_start:c_end]
                # Use SUM to get total brightness of the species
                intensities[name] = np.sum(roi) 
            else:
                intensities[name] = 0.0
                
        return intensities

    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

# ==========================================
# 3. MAIN ANALYSIS
# ==========================================

intensity_data = {t: [] for t in TIME_POINTS}

# A. Extract Intensities from all files
for t_label, folder in TIME_POINTS.items():
    full_path = os.path.join(BASE_PATH, folder)
    files = [f for f in glob.glob(os.path.join(full_path, "*.xls*")) if not os.path.basename(f).startswith('~$')]
    
    print(f"Extracting intensities from {t_label} ({len(files)} files)...")
    
    for f in files:
        ints = get_peak_intensity(f, chirality_centers)
        if ints:
            ints['filename'] = os.path.basename(f) # Keep track of sample name
            intensity_data[t_label].append(ints)

# B. Calculate Correlation Matrices
correlation_results = {}

for t_label in TIME_POINTS:
    if not intensity_data[t_label]:
        continue
        
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(intensity_data[t_label])
    
    # Drop filename column for correlation calculation
    df_numeric = df.drop(columns=['filename'], errors='ignore')
    
    # Calculate Pearson Correlation
    # 1.0 = Perfect positive linear relationship
    # -1.0 = Perfect negative linear relationship
    corr_matrix = df_numeric.corr(method='pearson')
    
    correlation_results[t_label] = corr_matrix

# ==========================================
# 4. VISUALIZATION
# ==========================================

if correlation_results:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    order = ['0h', '6h', '24h']
    
    for i, t in enumerate(order):
        if t in correlation_results:
            ax = axes[i]
            # Use 'coolwarm' map: Red=Positive, Blue=Negative, White=Zero
            sns.heatmap(correlation_results[t], annot=False, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax, square=True)
            ax.set_title(f"Intensity Covariance @ {t}\n(Red=Bundled/Conc, Blue=Energy Transfer?)")
            ax.set_xlabel("Chirality")
    
    axes[0].set_ylabel("Chirality")
    plt.tight_layout()
    plt.show()

    # Optional: Plot the "Total Drift"
    # If correlations get stronger over time, the system is bundling.
    if '0h' in correlation_results and '24h' in correlation_results:
        delta = correlation_results['24h'] - correlation_results['0h']
        plt.figure(figsize=(10, 8))
        sns.heatmap(delta, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
        plt.title("Change in Correlation (24h - 0h)\nRed = Relationships becoming tighter")
        plt.show()
else:
    print("No data extracted.")