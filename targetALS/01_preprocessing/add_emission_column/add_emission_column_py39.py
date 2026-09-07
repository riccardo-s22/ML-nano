#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Add an 'Emission' column (as the FIRST column) to one or many CSV/XLSX files.
# Compatible with Python 3.7+ (uses typing.Optional instead of the "|" union operator).
#
# Usage examples:
#   python add_emission_column_py39.py "C:\path\to\file.xlsx"
#   python add_emission_column_py39.py "C:\path\to\folder"
#   python add_emission_column_py39.py "C:\data" --emission-file "C:\data\emission_values.txt" --outdir "C:\data\out"
#
# Requires: pandas, openpyxl (for Excel), numpy
# Install (if needed): python -m pip install pandas openpyxl numpy

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

SUPPORTED_EXTS = {'.csv', '.tsv', '.xlsx', '.xls'}

def read_emission_values(path: Path) -> List[float]:
    vals = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.lower().startswith('emission'):
                continue
            try:
                vals.append(float(s))
            except ValueError:
                # ignore non-numeric lines
                pass
    if not vals:
        raise ValueError(f'No numeric emission values found in {path}')
    return vals

def insert_emission_column(df: pd.DataFrame, emission_vals: List[float], col_name: str='Emission') -> pd.DataFrame:
    n = len(df)
    # match length
    if len(emission_vals) < n:
        v = emission_vals + [np.nan] * (n - len(emission_vals))
    else:
        v = emission_vals[:n]
    out = df.copy()
    out.insert(0, col_name, v)
    return out

def process_csv(path: Path, emission_vals: List[float], inplace: bool=False, outdir: Optional[Path]=None) -> Path:
    # Decide delimiter by extension
    sep = ',' if path.suffix.lower() == '.csv' else '\t'
    df = pd.read_csv(path, sep=sep, header=0)
    out_df = insert_emission_column(df, emission_vals)

    if inplace:
        out_path = path
    else:
        out_path = (outdir or path.parent) / f'{path.stem}_with_emission{path.suffix}'
    out_df.to_csv(out_path, index=False, sep=sep)
    return out_path

def process_excel(path: Path, emission_vals: List[float], inplace: bool=False, outdir: Optional[Path]=None) -> Path:
    xls = pd.ExcelFile(path)
    out_sheets = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet, header=0)
        out_sheets[sheet] = insert_emission_column(df, emission_vals)

    if inplace:
        out_path = path
    else:
        out_path = (outdir or path.parent) / f'{path.stem}_with_emission{path.suffix}'

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for sheet, sdf in out_sheets.items():
            sdf.to_excel(writer, sheet_name=sheet, index=False)
    return out_path

def collect_targets(arg_paths) -> list:
    targets = []
    for p in arg_paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            targets.append(p)
        elif p.is_dir():
            for ext in SUPPORTED_EXTS:
                targets.extend(sorted(p.rglob(f'*{ext}')))
        else:
            print(f'WARN: Skipping {p} (not a supported file or directory)')
    # de-duplicate while keeping order
    seen = set()
    unique = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Insert 'Emission' as first column in CSV/XLSX files.")
    ap.add_argument('paths', nargs='+', help='Input files and/or directories')
    ap.add_argument('--emission-file', default='emission_values.txt',
                    help='Text file with one emission value per line (default: emission_values.txt)')
    ap.add_argument('--col-name', default='Emission', help="Column name to use (default: 'Emission')")
    ap.add_argument('--inplace', action='store_true', help='Overwrite input files (default: write _with_emission copies)')
    ap.add_argument('--outdir', default=None, help='Optional output directory (created if missing)')
    args = ap.parse_args(argv)

    emission_path = Path(args.emission_file)
    if not emission_path.exists():
        print(f'ERROR: Emission file not found: {emission_path}', file=sys.stderr)
        return 2
    emission_vals = read_emission_values(emission_path)

    targets = collect_targets(args.paths)
    if not targets:
        print('No target files found.', file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    for t in targets:
        try:
            if t.suffix.lower() in {'.csv', '.tsv'}:
                out_path = process_csv(t, emission_vals, inplace=args.inplace, outdir=outdir)
            else:
                out_path = process_excel(t, emission_vals, inplace=args.inplace, outdir=outdir)
            print(f'OK: {t} -> {out_path}')
        except Exception as e:
            print(f'ERROR processing {t}: {e}', file=sys.stderr)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
