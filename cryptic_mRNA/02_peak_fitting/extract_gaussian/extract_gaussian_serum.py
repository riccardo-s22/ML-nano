#!/usr/bin/env python3
"""Per-chirality split-Gaussian fit for exp3 SERUM plate (reuses exp1 fit code).
Format: col0=emission, col1=clipped ref ramp (IGNORE), cols2+=wells row-major (tiny units, real).
Grid 8 rows x 12 columns, row-major, last well (row7 col11) missing -> 95 wells (col11 has 7 reps).
Conditions BLIND: labelled by column index C00..C11. Output chirality_gaussian_descriptors_serum.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1")
from extract_gaussian import CHIRALITY, DESC, EXCITATIONS, assign_excitation, fit_chirality
BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3/serum"; PREFIX="serum__"; NCOL=12
ex_map=assign_excitation()
em=None; spec={}
for e in EXCITATIONS:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]
    spec[e]=a[:,2:]
nwell=spec[EXCITATIONS[0]].shape[1]
rows=[]
for w in range(nwell):
    p=w+1; r=p//NCOL; c=p%NCOL   # first grid cell (row0,col0) is MISSING -> file well w = grid pos w+1
    meta=dict(plate="serum",matrix="serum",hours=2,w=w,row=r,col=c,well=f"r{r}c{c}",cond=f"C{c:02d}")
    for ch,cc in CHIRALITY.items():
        d=fit_chirality(em, spec[ex_map[ch]][:,w], cc["em"]); d["ex_used"]=ex_map[ch]
        for k in DESC: meta[f"{ch}_{k}"]=d[k]
    rows.append(meta)
df=pd.DataFrame(rows); out=f"{BASE}/chirality_gaussian_descriptors_serum.csv"; df.to_csv(out,index=False)
r2=df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel(); r2=r2[~np.isnan(r2)]
print(f"Saved {out} shape={df.shape} ({nwell} wells); cols/well:",df.groupby('col').size().to_dict())
print(f"R2 median={np.median(r2):.3f} frac>0.95={np.mean(r2>0.95):.2f}")
for ch in CHIRALITY:
    print(f"  {ch:7s} frac_r2>0.95={np.mean(df[f'{ch}_gauss_r2']>0.95):.2f}")
