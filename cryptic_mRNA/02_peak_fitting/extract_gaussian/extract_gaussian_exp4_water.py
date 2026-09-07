#!/usr/bin/env python3
"""Per-chirality split-Gaussian fit for exp4 SERUM (STMN_22 = 22nt GT15-STMN2-CE sensor).
Grid 8x12 row-major, missing grid cell = row7,col7 (pos 91) -> 95 wells (col7=7 reps).
File well w -> grid p = w if w<91 else w+1. BLIND: label by column C00..C11.
Output chirality_gaussian_descriptors_exp4_water.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1")
from extract_gaussian import CHIRALITY, DESC, EXCITATIONS, assign_excitation, fit_chirality
BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water"; PREFIX="STMN_22_WATER__"; NCOL=12; MISS=95
ex_map=assign_excitation()
em=None; spec={}
for e in EXCITATIONS:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]
    spec[e]=a[:,2:]
nwell=spec[EXCITATIONS[0]].shape[1]
rows=[]
for w in range(nwell):
    p=w if w<MISS else w+1; r=p//NCOL; c=p%NCOL
    meta=dict(plate="exp4_water",matrix="water",hours=2,w=w,gp=p,row=r,col=c,well=f"r{r}c{c}",cond=f"C{c:02d}")
    for ch,cc in CHIRALITY.items():
        d=fit_chirality(em, spec[ex_map[ch]][:,w], cc["em"]); d["ex_used"]=ex_map[ch]
        for k in DESC: meta[f"{ch}_{k}"]=d[k]
    rows.append(meta)
df=pd.DataFrame(rows); out=f"{BASE}/chirality_gaussian_descriptors_exp4_water.csv"; df.to_csv(out,index=False)
r2=df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel(); r2=r2[~np.isnan(r2)]
print(f"Saved {out} shape={df.shape} ({nwell} wells); reps/col:",df.groupby('col').size().to_dict())
print(f"R2 median={np.median(r2):.3f} frac>0.95={np.mean(r2>0.95):.2f}")
for ch in CHIRALITY:
    print(f"  {ch:7s} frac_r2>0.95={np.mean(df[f'{ch}_gauss_r2']>0.95):.2f}")
