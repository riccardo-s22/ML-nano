from pathlib import Path
import pandas as pd

# === 1. Set your folder path here ===
folder = Path("C:/Users/riccardo-s/Documents/CNT/paper/ex123/ALS_excel_raw_data_Roy-20251122T180413Z-1-001/ALS_excel_raw_data_Roy/ex3/out_24h")  # <-- change this

# === 2. Collect files to rename (skip hidden files & existing sample_label) ===
files = [
    p for p in folder.iterdir()
    if p.is_file()
    and not p.name.startswith(".")
    and p.name != "sample_label.xlsx"
]

new_ids = []

for p in files:
    # remove existing extension, then add _e1.xlsx
    stem = p.stem                       # e.g. "sample1" from "sample1.xlsx"
    new_name = f"{stem}_e3.xlsx"        # -> "sample1_e3.xlsx"
    new_path = p.with_name(new_name)

    p.rename(new_path)
    new_ids.append(stem + "_e3")        # ID without extension

# === 3. Create sample_label.xlsx with all file IDs ===
df = pd.DataFrame({"sample_id": new_ids})
df.to_excel(folder / "sample_label.xlsx", index=False)

print("Done! Renamed files and created sample_label.xlsx.")
