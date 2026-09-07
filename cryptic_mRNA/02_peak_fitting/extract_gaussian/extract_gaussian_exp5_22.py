#!/usr/bin/env python3
"""Per-chirality split-Gaussian fit for exp5 / 22nt GT15-STMN2-CE sensor in defined background
(SDS1%+BSA0.5%+randomRNA 30ng/mL). NEW FILE FORMAT: 512 emission-pts x 95 columns = 95 wells DIRECTLY
(no wavelength col, no ref col). Inject the exp4 instrument emission axis. Geometry 8x12 row-major,
missing = last grid cell (col11,row7) -> p=w, row=w//12, col=w%12 (no shift).
Randomized split layout: TOP rows 0-3 vs BOTTOM rows 4-7 have different column->concentration maps.
TOP col0 (=1e5) is the WRONG sensor (20nt) -> flagged label='wrong'. Output CSV includes copies/label/half.
Output chirality_gaussian_descriptors_exp5_22.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1")
from extract_gaussian import CHIRALITY, DESC, EXCITATIONS, assign_excitation, fit_chirality

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp5/22"; PREFIX="STMN_22_WATER_2H_"; NCOL=12
AX=np.loadtxt("/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water/STMN_22_WATER__570.txt")[:,0]
TOP={0:(1e5,'wrong'),1:(0,'b'),2:(5000,'d'),3:(1,'d'),4:(1000,'d'),5:(10,'d'),6:(0,'z'),7:(50,'d'),8:(500,'d'),9:(0.1,'d'),10:(100,'d'),11:(1e4,'d')}
BOT={0:(0.1,'d'),1:(1e4,'d'),2:(50,'d'),3:(500,'d'),4:(0,'b'),5:(1000,'d'),6:(5000,'d'),7:(100,'d'),8:(10,'d'),9:(1e5,'d'),10:(1,'d'),11:(0,'z')}

ex_map=assign_excitation()
em=AX; spec={}
for e in EXCITATIONS:
    W=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")     # 512 x 95, ALL columns are wells
    assert W.shape[0]==em.shape[0], f"row mismatch {W.shape} vs axis {em.shape}"
    spec[e]=W
nwell=spec[EXCITATIONS[0]].shape[1]
rows=[]
for w in range(nwell):
    p=w; r=p//NCOL; c=p%NCOL
    cp,lab=(TOP if r<=3 else BOT)[c]
    meta=dict(plate="exp5_22",matrix="background_SDS_BSA_randomRNA",hours=2,w=w,row=r,col=c,
              half=("top" if r<=3 else "bot"),well=f"r{r}c{c}",copies=cp,label=lab)
    for ch,cc in CHIRALITY.items():
        d=fit_chirality(em, spec[ex_map[ch]][:,w], cc["em"]); d["ex_used"]=ex_map[ch]
        for k in DESC: meta[f"{ch}_{k}"]=d[k]
    rows.append(meta)
df=pd.DataFrame(rows); out=f"{BASE}/chirality_gaussian_descriptors_exp5_22.csv"; df.to_csv(out,index=False)
r2=df[[f"{ch}_gauss_r2" for ch in CHIRALITY]].to_numpy().ravel(); r2=r2[~np.isnan(r2)]
print(f"Saved {out} shape={df.shape} ({nwell} wells); reps/col:",df.groupby('col').size().to_dict())
print(f"R2 median={np.median(r2):.3f} frac>0.95={np.mean(r2>0.95):.2f}")
for ch in CHIRALITY:
    print(f"  {ch:7s} frac_r2>0.95={np.mean(df[f'{ch}_gauss_r2']>0.95):.2f}")
