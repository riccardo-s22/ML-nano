from pathlib import Path
import pandas as pd

# === 1. Folder with your Excel files ===
folder = Path("/home/riccardo-s_uri_edu/CNT/exp5/out_24h")  # <-- change this if needed

# Emission cutoff: keep only rows with Emission <= this value
emission_cutoff = 1400.8958

for path in folder.glob("*.xlsx"):
    if path.name == "sample_label.xlsx":
        continue  # don't touch the label file

    print(f"Processing {path.name} ...")
    df = pd.read_excel(path)

    # --- Keep all columns, only filter rows by Emission ---
    if "Emission" not in df.columns:
        print(f"  WARNING: 'Emission' column not found in {path.name}, cannot filter rows.")
        df_kept = df.copy()
    else:
        before_rows = len(df)
        df_kept = df[df["Emission"] <= emission_cutoff].reset_index(drop=True)
        after_rows = len(df_kept)
        print(f"  Emission filter: kept {after_rows}/{before_rows} rows (≤ {emission_cutoff}).")

    # Overwrite the file
    df_kept.to_excel(path, index=False)

print("Done: rows with Emission > 1400.8958 have been deleted (all columns kept).")
