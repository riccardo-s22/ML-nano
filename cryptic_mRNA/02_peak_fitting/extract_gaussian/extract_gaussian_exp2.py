#!/usr/bin/env python3
"""
Per-chirality split-Gaussian+baseline fit for exp2 (reuses exp1 extract_gaussian fit code).
exp2 format: each file col0=emission wl, col1=ramp(ignore), cols2+=wells (ROW-MAJOR).
4 plates (water/serum x 2h/3h). Outputs chirality_gaussian_descriptors_exp2.csv (one row/well).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1")
from extract_gaussian import CHIRALITY, DESC, EXCITATIONS, assign_excitation, fit_chirality

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2"
# folder, prefix, ncol, nrow, drop_last_row, conc map by column index
PLATES=[("2H_WATER","STMN_2_1H__",6,8,False,[0,100,0.1,10,1,"water"]),
        ("3H_WATER","3H_WATER__", 6,8,False,[0,100,0.1,10,1,"water"]),
        ("2H_SERUM","SERUM__",    5,8,True ,[1000,10,0,1,100]),
        ("3H_SERUM","3HSERUM__",  5,8,True ,[1000,10,0,1,100])]
ex_map=assign_excitation()

def load_plate(folder,prefix):
    em=None; spec={}
    for e in EXCITATIONS:
        a=np.loadtxt(f"{BASE}/{folder}/{prefix}{e}.txt")
        if em is None: em=a[:,0]
        spec[e]=a[:,2:]          # emission x wells (row-major well order)
    return em, spec

rows=[]
for folder,prefix,ncol,nrow,drop,cmap in PLATES:
    em,spec=load_plate(folder,prefix)
    nwell=spec[EXCITATIONS[0]].shape[1]
    for w in range(nwell):
        r=w//ncol; c=w%ncol
        if drop and r==nrow-1: continue           # exclude bad last row (serum)
        conc=cmap[c]
        meta=dict(plate=folder, matrix=folder[3:], hours=int(folder[0]),
                  row=r, col=c, well=f"{folder}_r{r}c{c}",
                  conc=str(conc), is_water=(conc=="water"),
                  dose=(np.nan if conc=="water" else float(conc)))
        for ch,cc in CHIRALITY.items():
            ex=ex_map[ch]
            d=fit_chirality(em, spec[ex][:,w], cc["em"])
            d["ex_used"]=ex
            for k in DESC: meta[f"{ch}_{k}"]=d[k]
        rows.append(meta)

df=pd.DataFrame(rows)
out=f"{BASE}/chirality_gaussian_descriptors_exp2.csv"; df.to_csv(out,index=False)
r2=df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel(); r2=r2[~np.isnan(r2)]
print(f"Saved {out}  shape={df.shape}")
print(f"wells/plate:", df.groupby('plate').size().to_dict())
print(f"R2 overall: median={np.median(r2):.3f} frac>0.95={np.mean(r2>0.95):.2f} frac>0.9={np.mean(r2>0.9):.2f}")
print("\nper-chirality R2>0.95 fraction & fitted-vs-nominal center drift:")
for ch,cc in CHIRALITY.items():
    rr=df[f"{ch}_gauss_r2"]; fc=df[f"{ch}_gauss_em_center"].mean()
    print(f"  {ch:7s} ex{ex_map[ch]} frac_r2>0.95={np.mean(rr>0.95):.2f}  nominal={cc['em']:7.1f} fitted={fc:7.1f} d={fc-cc['em']:+5.1f}")
