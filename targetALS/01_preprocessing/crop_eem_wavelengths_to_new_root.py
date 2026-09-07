#!/usr/bin/env python3
"""
Crop NS Spectralyzer-style EEM Excel exports by wavelength and SAVE to a separate output root,
keeping the SAME filenames.

Cuts out:
  - Emission < 920 nm            -> drops rows
  - Excitation < 520 nm          -> drops columns
  - Excitation > 815 nm          -> drops columns

Input format expected:
  - Column named 'Emission' (preferred) OR first column is emission (nm)
  - Excitation columns named like: 'Excitation_00520.000', etc.

Output:
  OUT_ROOT/
    out_0h/<same filenames>.xlsx
    out_6h/<same filenames>.xlsx
    out_24h/<same filenames>.xlsx
"""

from pathlib import Path
import re
import pandas as pd

# ==========================
# USER SETTINGS
# ==========================
TIMEPOINT_DIRS = [
    Path(r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\out_0h"),
    Path(r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\out_6h"),
    Path(r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\out_24h"),
]

# Separate output folder root (CHANGE THIS)
OUT_ROOT = Path(r"C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\cropped_em920_ex520-815")

EM_MIN = 920.0
EX_MIN = 520.0
EX_MAX = 815.0

# Optional: skip specific filenames (case-insensitive)
SKIP_NAMES = {
    # "sample_label.xlsx",
}

# ==========================
# HELPERS
# ==========================
_EX_RE = re.compile(r"Excitation_\s*0*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_excitation_nm(colname: str):
    """Return excitation nm as float if colname matches expected pattern; else None."""
    if colname is None:
        return None
    s = str(colname).strip()
    m = _EX_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def crop_one_file(xlsx_path: Path, out_path: Path) -> tuple[int, int, int, int]:
    """Return (rows_before, rows_after, cols_before, cols_after)."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Detect emission column (prefer explicit 'Emission', else first column)
    em_col = "Emission" if "Emission" in df.columns else df.columns[0]

    rows_before = len(df)
    cols_before = len(df.columns)

    # Filter emission rows
    em = pd.to_numeric(df[em_col], errors="coerce")
    df = df.loc[em >= EM_MIN].copy()

    # Keep excitation columns in range, ordered by excitation numeric value
    exc_cols = []
    exc_vals = []
    for c in df.columns:
        if c == em_col:
            continue
        ex_nm = parse_excitation_nm(c)
        if ex_nm is None:
            continue
        if EX_MIN <= ex_nm <= EX_MAX:
            exc_cols.append(c)
            exc_vals.append(ex_nm)

    if not exc_cols:
        raise ValueError(f"No excitation columns found in [{EX_MIN}, {EX_MAX}] in {xlsx_path.name}")

    # sort excitation columns by numeric nm
    exc_cols_sorted = [c for _, c in sorted(zip(exc_vals, exc_cols))]

    df_out = pd.concat([df[[em_col]], df[exc_cols_sorted]], axis=1).rename(columns={em_col: "Emission"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False, engine="openpyxl")

    rows_after = len(df_out)
    cols_after = len(df_out.columns)
    return rows_before, rows_after, cols_before, cols_after

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    skip_lower = {s.lower() for s in SKIP_NAMES}

    for tp_dir in TIMEPOINT_DIRS:
        if not tp_dir.exists():
            print(f"[SKIP] Folder not found: {tp_dir}")
            continue

        # Mirror each timepoint folder name under OUT_ROOT
        out_dir = OUT_ROOT / tp_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Timepoint folder: {tp_dir} ===")
        print(f"Output folder: {out_dir}")

        xlsx_files = sorted([p for p in tp_dir.glob("*.xlsx") if not p.name.startswith("~$")])
        if not xlsx_files:
            print("  (No .xlsx files found)")
            continue

        for xlsx_path in xlsx_files:
            if xlsx_path.name.lower() in skip_lower:
                print(f"  [SKIP] {xlsx_path.name}")
                continue

            out_path = out_dir / xlsx_path.name  # SAME filename
            try:
                rb, ra, cb, ca = crop_one_file(xlsx_path, out_path)
                print(f"  {xlsx_path.name}: rows {ra}/{rb} kept (EM>={EM_MIN}); cols {ca}/{cb} kept (EX {EX_MIN}-{EX_MAX})")
            except Exception as e:
                print(f"  [ERROR] {xlsx_path.name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
