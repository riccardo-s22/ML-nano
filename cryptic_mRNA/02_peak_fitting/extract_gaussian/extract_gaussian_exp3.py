#!/usr/bin/env python3
"""
Per-chirality split-Gaussian+baseline fit for exp3 (reuses exp1 extract_gaussian fit code).
exp3 format (same as exp2): each file col0=emission wl, col1=clipped reference ramp (IGNORE),
cols2+=wells (ROW-MAJOR). One plate: water_2h, 88 wells = 8 replicate rows x 11 condition-columns.
Conditions are kept BLIND: labelled only by column index 0..10.
Outputs chirality_gaussian_descriptors_exp3.csv (one row/well).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1")
from extract_gaussian import CHIRALITY, DESC, EXCITATIONS, assign_excitation, fit_chirality

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3"
PREFIX="water_2h__"
NCOL=11; NROW=8
ex_map=assign_excitation()

def load_plate():
    em=None; spec={}
    for e in EXCITATIONS:
        a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
        if em is None: em=a[:,0]
        spec[e]=a[:,2:]          # emission x wells (row-major well order); col1 ignored
    return em, spec

em,spec=load_plate()
nwell=spec[EXCITATIONS[0]].shape[1]
assert nwell==NCOL*NROW, f"expected {NCOL*NROW} wells, got {nwell}"
rows=[]
for w in range(nwell):
    r=w//NCOL; c=w%NCOL
    meta=dict(plate="water_2h", matrix="water", hours=2,
              row=r, col=c, well=f"r{r}c{c}",
              cond=f"C{c:02d}")          # BLIND condition label = column index
    for ch,cc in CHIRALITY.items():
        ex=ex_map[ch]
        d=fit_chirality(em, spec[ex][:,w], cc["em"])
        d["ex_used"]=ex
        for k in DESC: meta[f"{ch}_{k}"]=d[k]
    rows.append(meta)

df=pd.DataFrame(rows)
out=f"{BASE}/chirality_gaussian_descriptors_exp3.csv"; df.to_csv(out,index=False)
r2=df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel(); r2=r2[~np.isnan(r2)]
print(f"Saved {out}  shape={df.shape}  ({nwell} wells: {NROW} rows x {NCOL} cols)")
print(f"R2 overall: median={np.median(r2):.3f} frac>0.95={np.mean(r2>0.95):.2f} frac>0.9={np.mean(r2>0.9):.2f}")
print("\nper-chirality R2>0.95 fraction & fitted-vs-nominal center:")
for ch,cc in CHIRALITY.items():
    rr=df[f"{ch}_gauss_r2"]; fc=df[f"{ch}_gauss_em_center"].mean()
    print(f"  {ch:7s} ex{ex_map[ch]} frac_r2>0.95={np.mean(rr>0.95):.2f}  nominal={cc['em']:7.1f} fitted={fc:7.1f} d={fc-cc['em']:+5.1f}")
