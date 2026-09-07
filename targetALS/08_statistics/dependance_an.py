import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
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

# Chirality Centers (Excitation, Emission)
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
# 2. DATA LOADING FUNCTIONS
# ==========================================

def clean_axis(axis_values):
    """
    Cleans axis labels by removing text like 'Excitation_' and converting to float.
    """
    clean = []
    for val in axis_values:
        # Convert to string, remove 'Excitation_' or 'Emission_', then to float
        s = str(val).replace('Excitation_', '').replace('Emission_', '')
        try:
            clean.append(float(s))
        except ValueError:
            # If conversion fails (e.g. label is 'Emission'), assume it's metadata and skip/handle
            # But for axis array, we usually need pure numbers.
            # If it's the index name, pandas might have handled it.
            pass
    return np.array(clean)

def load_excel_matrix(folder_name):
    full_path = os.path.join(BASE_PATH, folder_name)
    # Search for .xlsx or .xls, ignoring temporary open files (~$)
    files = [f for f in glob.glob(os.path.join(full_path, "*.xls*")) if not os.path.basename(f).startswith('~$')]
    
    if not files:
        print(f"WARNING: No Excel file found in {folder_name}")
        return None, None, None
    
    target_file = max(files, key=os.path.getsize)
    print(f"Loading: {os.path.basename(target_file)} from {folder_name}")

    try:
        # Load Excel. index_col=0 implies the first column (Emission) is the Index.
        df = pd.read_excel(target_file, index_col=0, engine='openpyxl')
        
        # 1. CLEAN THE COLUMNS (Excitation Axis)
        # The columns are likely ['Excitation_500', 'Excitation_505'...]
        # We strip the text to get just the numbers.
        ex_axis = df.columns.astype(str).str.replace('Excitation_', '').astype(float).to_numpy()
        
        # 2. CLEAN THE INDEX (Emission Axis)
        # The index is likely pure numbers (Emission wavelengths), but we ensure float type.
        em_axis = df.index.to_numpy(dtype=float)

        # 3. ORIENTATION CHECK
        # We need the matrix to be [Row=Excitation, Col=Emission] for consistency with the logic below.
        # Currently: DataFrame Index = Emission, Columns = Excitation.
        # So df.values is [Emission x Excitation].
        # We must TRANSPOSE it to be [Excitation x Emission].
        data_matrix = df.values.T 
        
        # Now: Rows = Excitation, Columns = Emission.
        
        # 4. SORTING CHECK (If wavelengths are in descending order, flip them)
        if ex_axis[0] > ex_axis[-1]:
            ex_axis = ex_axis[::-1]
            data_matrix = data_matrix[::-1, :] # Flip rows
            
        if em_axis[0] > em_axis[-1]:
            em_axis = em_axis[::-1]
            data_matrix = data_matrix[:, ::-1] # Flip cols

        return data_matrix, ex_axis, em_axis

    except Exception as e:
        print(f"Error reading {target_file}: {e}")
        return None, None, None

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

def get_5x5_matrix(data_map, ex_axis, em_axis, center_ex, center_em):
    # Find index closest to center
    idx_ex = (np.abs(ex_axis - center_ex)).argmin()
    idx_em = (np.abs(em_axis - center_em)).argmin()
    
    # Extract 5x5 window
    r_start, r_end = idx_ex - 2, idx_ex + 3
    c_start, c_end = idx_em - 2, idx_em + 3
    
    # Boundary check
    if r_start < 0 or c_start < 0 or r_end > len(ex_axis) or c_end > len(em_axis):
        return np.zeros((5,5))
        
    matrix = data_map[r_start:r_end, c_start:c_end]
    
    if matrix.shape != (5, 5):
        return np.zeros((5,5))
        
    return matrix

# --- MAIN EXECUTION ---
time_results = {}

for t_label, folder in TIME_POINTS.items():
    print(f"--- Processing {t_label} ---")
    data_map, ex_ax, em_ax = load_excel_matrix(folder)
    
    if data_map is not None:
        chirality_vectors = {}
        
        for name, (cen_ex, cen_em) in chirality_centers.items():
            mat = get_5x5_matrix(data_map, ex_ax, em_ax, cen_ex, cen_em)
            vec = mat.flatten()
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            chirality_vectors[name] = vec

        # Similarity Matrix
        keys = list(chirality_centers.keys())
        n = len(keys)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                vec_i = chirality_vectors[keys[i]]
                vec_j = chirality_vectors[keys[j]]
                
                if np.linalg.norm(vec_i) == 0 or np.linalg.norm(vec_j) == 0:
                    sim_matrix[i, j] = 0
                else:
                    sim_matrix[i, j] = 1 - cosine(vec_i, vec_j)
                
        time_results[t_label] = pd.DataFrame(sim_matrix, index=keys, columns=keys)

# ==========================================
# 4. VISUALIZATION
# ==========================================

if time_results:
    # A. Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    order = ['0h', '6h', '24h']
    
    for i, t in enumerate(order):
        if t in time_results:
            ax = axes[i]
            sns.heatmap(time_results[t], annot=False, cmap="viridis", vmin=0, vmax=1, ax=ax, square=True)
            ax.set_title(f"Interdependence @ {t}")
            ax.set_xlabel("Chirality")
    
    axes[0].set_ylabel("Chirality")
    plt.tight_layout()
    plt.show()

    # B. Time Evolution
    if '0h' in time_results and '24h' in time_results:
        delta = time_results['24h'] - time_results['0h']
        plt.figure(figsize=(10, 8))
        sns.heatmap(delta, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
        plt.title("Change in Spectral Overlap (24h - 0h)")
        plt.show()
else:
    print("Analysis failed: No data loaded.")