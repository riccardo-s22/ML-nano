import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
# Make sure this matches your actual filename!
# If your file is "example.xlsx", keep it. If it is "example.xlsx - Sheet1.csv", change it.
EEM_AXES_FILE = "example.xlsx"  
CHIRALITY_FILE = "Coordinates_DNA.txt"

def load_eem_axes(filepath):
    """
    Smart loader: Detects .xlsx vs .csv and reads the axes.
    """
    print(f"Loading Axes from: {filepath}")
    
    # 1. Check if file exists
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: File not found: {filepath}")
        return None, None
        
    try:
        # 2. Choose Loader based on extension
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            print("  -> Detected Excel format. Using read_excel...")
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            print("  -> Detected CSV/Text format. Using read_csv...")
            # Try likely encodings for CSV
            try:
                df = pd.read_csv(filepath, encoding='utf-8', engine='python')
            except:
                df = pd.read_csv(filepath, encoding='cp1252', engine='python')

        # 3. Parse Excitation (Columns)
        # We look for columns that contain "Excitation" or look like numbers
        ex_vals = []
        for col in df.columns:
            col_str = str(col)
            if "Excitation" in col_str:
                try:
                    # Clean up "Excitation_00500.000" -> 500.0
                    val = float(col_str.replace("Excitation_", "").replace("_", ""))
                    ex_vals.append(val)
                except: pass
            else:
                # Fallback: check if the column header ITSELF is a number (common in some exports)
                try:
                    val = float(col_str)
                    if 200 < val < 1500: # Sanity check for nm
                        ex_vals.append(val)
                except: pass
        
        ex_axis = np.array(sorted(list(set(ex_vals))), dtype=np.float32)
        
        # 4. Parse Emission (Rows - First Column)
        # We assume the first column holds the Y-axis
        try:
            em_vals = df.iloc[:, 0].astype(float).values
            # Filter NaNs if any
            em_vals = em_vals[~np.isnan(em_vals)]
            em_axis = np.array(sorted(em_vals), dtype=np.float32)
        except:
             # Fallback: try index if column 0 failed
             em_vals = df.index.astype(float).values
             em_axis = np.array(sorted(em_vals), dtype=np.float32)

        if len(ex_axis) == 0 or len(em_axis) == 0:
            print("  ERROR: Could not find numeric axes. Check column headers.")
            print(f"  Columns found: {list(df.columns)[:5]}...")
            return None, None

        print(f"  Excitation: {len(ex_axis)} pts ({ex_axis.min():.1f} - {ex_axis.max():.1f} nm)")
        print(f"  Emission:   {len(em_axis)} pts ({em_axis.min():.1f} - {em_axis.max():.1f} nm)")
        
        return ex_axis, em_axis
        
    except Exception as e:
        print(f"CRITICAL ERROR reading axes: {e}")
        return None, None

def load_chirality_ref(filepath):
    """
    Parses Coordinates_DNA.txt (Block format: [n,m] \n Emission=... \n Excitation=...)
    """
    print(f"Loading Chiralities from: {filepath}")
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: File not found: {filepath}")
        return None
        
    chirality_list = []
    current_name, current_em, current_ex = None, None, None
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Case 1: Name [n,m]
            if line.startswith('[') and ']' in line:
                if current_name and current_em and current_ex:
                    chirality_list.append({'name': current_name, 'ex': current_ex, 'em': current_em})
                current_name = line.replace('[', '(').replace(']', ')')
                current_em, current_ex = None, None
                
            # Case 2: Emission
            elif "Emission" in line and "=" in line:
                try: current_em = float(line.split('=')[1].strip())
                except: pass
                
            # Case 3: Excitation
            elif "Excitation" in line and "=" in line:
                try: current_ex = float(line.split('=')[1].strip())
                except: pass

        # Add last one
        if current_name and current_em and current_ex:
            chirality_list.append({'name': current_name, 'ex': current_ex, 'em': current_em})
            
        df = pd.DataFrame(chirality_list)
        print(f"  Loaded {len(df)} chiralities.")
        return df
        
    except Exception as e:
        print(f"CRITICAL ERROR reading chiralities: {e}")
        return None

def nm_to_pixel(val_nm, axis_nm):
    """Finds the closest pixel index"""
    if axis_nm is None or len(axis_nm) == 0: return 0
    return (np.abs(axis_nm - val_nm)).argmin()

def main():
    # 1. Get Axes
    ex_axis, em_axis = load_eem_axes(EEM_AXES_FILE)
    if ex_axis is None: return

    # 2. Get Chirality Ref
    df_chir = load_chirality_ref(CHIRALITY_FILE)
    if df_chir is None or df_chir.empty: return

    # 3. Create Plot
    plt.figure(figsize=(10, 10))
    plt.xlim(0, len(ex_axis))
    plt.ylim(0, len(em_axis))
    
    print("\n--- MAPPING RESULTS (Pixel Coords) ---")
    
    for _, row in df_chir.iterrows():
        px_x = nm_to_pixel(row['ex'], ex_axis)
        px_y = nm_to_pixel(row['em'], em_axis)
        
        if 0 <= px_x < len(ex_axis) and 0 <= px_y < len(em_axis):
            plt.scatter(px_x, px_y, c='red', marker='x', s=100)
            plt.text(px_x + 1, px_y + 1, row['name'], fontsize=11, color='blue', fontweight='bold')
            print(f"{row['name']}: Ex {row['ex']}nm -> Px {px_x} | Em {row['em']}nm -> Px {px_y}")

    # Standard EEM: Invert Y so Row 0 is at the top
    plt.gca().invert_yaxis() 
    
    plt.title("Chirality Map (Pixel Coordinates)", fontsize=16)
    plt.xlabel(f"Excitation Pixel (0-{len(ex_axis)-1})", fontsize=12)
    plt.ylabel(f"Emission Pixel (0-{len(em_axis)-1})", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("Chirality_Map_Overlay.png", dpi=300)
    print("\nSUCCESS: Saved 'Chirality_Map_Overlay.png'. Open it and compare with ROI plots!")

if __name__ == "__main__":
    main()