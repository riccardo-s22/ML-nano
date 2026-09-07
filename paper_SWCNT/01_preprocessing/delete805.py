from pathlib import Path
import pandas as pd

# === 1. Folder with your Excel files ===
folder = Path(r"C:\Users\riccardo-s\Documents\CNT\paper\ex123\ALS_excel_raw_data_Roy-20251122T180413Z-1-001\ALS_excel_raw_data_Roy\ex3\out_24h")

# Column after which everything should be removed
cut_col = "Excitation_00805.000"

# Emission cutoff: keep only rows with Emission <= this value
emission_cutoff = 1400.8958

for path in folder.glob("*.xlsx"):
    if path.name == "sample_label.xlsx":
        continue  # don't touch the label file

    print(f"Processing {path.name} ...")
    df = pd.read_excel(path)

    # --- 1) Keep columns up to cut_col ---
    if cut_col not in df.columns:
        print(f"  WARNING: {cut_col} not found in {path.name}, skipping column trimming.")
        df_kept = df.copy()
    else:
        idx = df.columns.get_loc(cut_col)
        df_kept = df.iloc[:, : idx + 1]

    # --- 2) Delete rows below emission 1400.8958 (i.e. Emission > 1400.8958) ---
    if "Emission" not in df_kept.columns:
        print(f"  WARNING: 'Emission' column not found in {path.name}, cannot filter rows.")
    else:
        before_rows = len(df_kept)
        df_kept = df_kept[df_kept["Emission"] <= emission_cutoff].reset_index(drop=True)
        after_rows = len(df_kept)
        print(f"  Emission filter: kept {after_rows}/{before_rows} rows (≤ {emission_cutoff}).")

    # Overwrite the file
    df_kept.to_excel(path, index=False)

print(
    "Done: all columns after Excitation_00805.000 have been removed "
    "and rows with Emission > 1400.8958 have been deleted."
)

