#!/usr/bin/env python3
"""Per-chirality split-Gaussian fit for exp5 / 20nt GT15-STMN2-CE sensor in defined background
(SDS1%+BSA0.5%+randomRNA 30ng/mL). OLD FILE FORMAT: 512 x 98 = emission(col0)+ref ramp(col1 IGNORE)+
96 wells(cols2+). FULL 8x12 grid, NO missing well -> p=w, row=w//12, col=w%12.
Randomized split layout: TOP rows 0-3 vs BOTTOM rows 4-7 different column->concentration maps.
NO wrong-sensor well reported for the 20 plate (that error was specific to the 22 plate) -> keep all wells.
Output chirality_gaussian_descriptors_exp5_20.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1")
from extract_gaussian import CHIRALITY, DESC, EXCITATIONS, assign_excitation, fit_chirality

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp5/20"; PREFIX="STMN_20_WATER_2H__"; NCOL=12
TOP={0:(1e5,'d'),1:(0,'b'),2:(5000,'d'),3:(1,'d'),4:(1000,'d'),5:(10,'d'),6:(0,'z'),7:(50,'d'),8:(500,'d'),9:(0.1,'d'),10:(100,'d'),11:(1e4,'d')}
BOT={0:(0.1,'d'),1:(1e4,'d'),2:(50,'d'),3:(500,'d'),4:(0,'b'),5:(1000,'d'),6:(5000,'d'),7:(100,'d'),8:(10,'d'),9:(1e5,'d'),10:(1,'d'),11:(0,'z')}

ex_map=assign_excitation()
em=None; spec={}
for e in EXCITATIONS:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]
    spec[e]=a[:,2:]                     # cols2+ = 96 wells
nwell=spec[EXCITATIONS[0]].shape[1]
rows=[]
for w in range(nwell):
    p=w; r=p//NCOL; c=p%NCOL
    cp,lab=(TOP if r<=3 else BOT)[c]
    meta=dict(plate="exp5_20",matrix="background_SDS_BSA_randomRNA",hours=2,w=w,row=r,col=c,
              half=("top" if r<=3 else "bot"),well=f"r{r}c{c}",copies=cp,label=lab)
    for ch,cc in CHIRALITY.items():
        d=fit_chirality(em, spec[ex_map[ch]][:,w], cc["em"]); d["ex_used"]=ex_map[ch]
        for k in DESC: meta[f"{ch}_{k}"]=d[k]
    rows.append(meta)
df=pd.DataFrame(rows); out=f"{BASE}/chirality_gaussian_descriptors_exp5_20.csv"; df.to_csv(out,index=False)
r2=df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel(); r2=r2[~np.isnan(r2)]
print(f"Saved {out} shape={df.shape} ({nwell} wells); reps/col:",df.groupby('col').size().to_dict())
print(f"R2 median={np.median(r2):.3f} frac>0.95={np.mean(r2>0.95):.2f}")
for ch in CHIRALITY:
    print(f"  {ch:7s} frac_r2>0.95={np.mean(df[f'{ch}_gauss_r2']>0.95):.2f}")
