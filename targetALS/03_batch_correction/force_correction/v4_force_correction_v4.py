import argparse
import pandas as pd
import numpy as np
import re
import os
import sys
from scipy.interpolate import griddata

# ================= HELPER FUNCTIONS =================

def parse_args():
    parser = argparse.ArgumentParser(description="Force Correction Surface Generation & Application (XLSX Support)")
    
    # Input/Output paths
    parser.add_argument("--batch1_pbs_folder", required=True, help="Folder containing Batch 1 PBS controls")
    parser.add_argument("--batch2_pbs_folder", required=True, help="Folder containing Batch 2 PBS controls")
    parser.add_argument("--coords_file", required=True, help="Path to Coordinates_DNA.txt")
    
    # Prefixes
    parser.add_argument("--batch1_prefix", default="", help="Prefix for Batch 1 PBS files (e.g. 'P'). Use '' for all.")
    parser.add_argument("--batch2_prefix", default="", help="Prefix for Batch 2 PBS files (e.g. 'P'). Use '' for all.")
    
    # Processing targets
    parser.add_argument("--input_dir", required=True, help="Root folder of Batch 2 data to correct")
    parser.add_argument("--output_dir", required=True, help="Where to save corrected files")
    
    return parser.parse_args()

def load_eem(file_path):
    """Loads EEM from CSV or XLSX, returns axes and data matrix."""
    try:
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            return None, None, None, None

        # Load based on extension
        if file_path.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)

        # Validate Headers
        ex_cols = [c for c in df.columns if 'Excitation' in c]
        if not ex_cols: 
            return None, None, None, None
            
        ex_axis = np.array([float(re.findall(r"[\d\.]+", c)[0]) for c in ex_cols])
        em_axis = df['Emission'].values
        data = df[ex_cols].values
        return em_axis, ex_axis, data, df
    except Exception as e:
        print(f"Warning: Could not load {os.path.basename(file_path)}: {e}")
        return None, None, None, None

def get_mean_pbs_matrix(folder_path, prefix):
    """Finds .xlsx OR .csv files starting with prefix, loads them, and returns AVERAGE matrix."""
    
    all_files = os.listdir(folder_path)
    # Filter for .csv or .xlsx
    valid_extensions = ('.csv', '.xlsx')
    data_files = [f for f in all_files if f.lower().endswith(valid_extensions) and not f.startswith('~$')]
    
    # Filter by prefix
    files = [f for f in data_files if f.upper().startswith(prefix.upper())]
    
    if not files:
        print(f"\n[ERROR] No .xlsx or .csv files starting with '{prefix}' found in: {folder_path}")
        print(f"Diagnostic: Found {len(data_files)} other data files:")
        for f in data_files[:5]: print(f"  - {f}")
        raise ValueError(f"No matching files found in {os.path.basename(folder_path)}")
        
    print(f"Found {len(files)} replicates in {os.path.basename(folder_path)} (Prefix='{prefix}'): {files}")
    
    stack = []
    em_ref, ex_ref = None, None
    
    for fname in files:
        path = os.path.join(folder_path, fname)
        em, ex, data, _ = load_eem(path)
        if data is not None:
            stack.append(data)
            if em_ref is None: 
                em_ref, ex_ref = em, ex
    
    if not stack:
        raise ValueError("Could not load any valid data from PBS files.")
        
    mean_matrix = np.mean(np.array(stack), axis=0)
    return em_ref, ex_ref, mean_matrix

def parse_coords(txt_path):
    with open(txt_path, 'r') as f:
        text = f.read()
    matches = re.findall(r'Emission\s+=\s+([\d\.]+)\s+Excitation\s+=\s+([\d\.]+)', text)
    return [{'Em': float(m[0]), 'Ex': float(m[1])} for m in matches]

# ================= MAIN EXECUTION =================

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("\n--- 1. Generating Correction Surface ---")
    
    try:
        print(f"Processing Batch 1 (Ref)...")
        em_ref, ex_ref, data_ref_mean = get_mean_pbs_matrix(args.batch1_pbs_folder, args.batch1_prefix)
        
        print(f"Processing Batch 2 (Target)...")
        em_tgt, ex_tgt, data_tgt_mean = get_mean_pbs_matrix(args.batch2_pbs_folder, args.batch2_prefix)
    except ValueError as e:
        print(f"\nSTOPPING: {e}")
        sys.exit(1)

    # 2. Calculate Ratios
    peaks = parse_coords(args.coords_file)
    points = []
    ratios = []

    print(f"\nCalculating ratios for {len(peaks)} chirality peaks...")
    for p in peaks:
        r_idx = (np.abs(em_ref - p['Em'])).argmin()
        c_idx = (np.abs(ex_ref - p['Ex'])).argmin()
        
        val_ref = data_ref_mean[r_idx, c_idx]
        val_tgt = data_tgt_mean[r_idx, c_idx]
        
        # FORCE CALCULATION
        if abs(val_tgt) > 1e-20:
            ratio = val_ref / val_tgt
        else:
            ratio = 1.0
            
        points.append([p['Ex'], p['Em']])
        ratios.append(ratio)

    # 3. Interpolate Surface
    grid_x, grid_y = np.meshgrid(ex_ref, em_ref)
    surface = griddata(np.array(points), np.array(ratios), (grid_x, grid_y), method='linear')
    mask = np.isnan(surface)
    surface[mask] = griddata(np.array(points), np.array(ratios), (grid_x[mask], grid_y[mask]), method='nearest')
    
    print(f"Surface Generated. Global Mean Correction Factor: {np.mean(surface):.4f}")
    pd.DataFrame(surface).to_csv(os.path.join(args.output_dir, "correction_surface_used.csv"))

    print("\n--- 2. Applying Correction to Batch 2 Data ---")
    
    count = 0
    # Process both .xlsx and .csv files in input dir
    valid_exts = ('.xlsx', '.csv')
    
    for root, dirs, files in os.walk(args.input_dir):
        for fname in files:
            # Skip hidden/temp excel files (~$) and non-data files
            if fname.lower().endswith(valid_exts) and not fname.startswith('~$') and "correction" not in fname.lower():
                full_path = os.path.join(root, fname)
                
                _, _, data_raw, df_orig = load_eem(full_path)
                
                if data_raw is not None:
                    data_corrected = data_raw * surface
                    
                    ex_cols = [c for c in df_orig.columns if 'Excitation' in c]
                    df_orig[ex_cols] = data_corrected
                    
                    # Determine Output Path
                    rel_path = os.path.relpath(full_path, args.input_dir)
                    save_path = os.path.join(args.output_dir, rel_path)
                    
                    # Ensure output is always .xlsx for consistency
                    save_path = os.path.splitext(save_path)[0] + ".xlsx"
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    df_orig.to_excel(save_path, index=False)
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"Corrected {count} files...", end='\r')

    print(f"\n\nDone! {count} files corrected.")
    print(f"Output Directory: {args.output_dir}")

if __name__ == "__main__":
    main()