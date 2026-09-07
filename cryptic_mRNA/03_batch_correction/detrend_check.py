#!/usr/bin/env python3
"""Left-right detrend then re-test (1) dose null and (2) random-RNA suppression, since the two 0-types
sit at different columns and the plate has a strong left-right gradient. Dose is randomized vs column,
so fitting int_rn ~ col globally estimates the pure spatial gradient (dose adds ~0 variance)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp5/22"; PREFIX="STMN_22_WATER_2H_"
EXC=[570,640,670,750,780]; NCOL=12
AX=np.loadtxt("/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water/STMN_22_WATER__570.txt")[:,0]
TOP={0:(1e5,'wrong'),1:(0,'b'),2:(5000,'d'),3:(1,'d'),4:(1000,'d'),5:(10,'d'),6:(0,'z'),7:(50,'d'),8:(500,'d'),9:(0.1,'d'),10:(100,'d'),11:(1e4,'d')}
BOT={0:(0.1,'d'),1:(1e4,'d'),2:(50,'d'),3:(500,'d'),4:(0,'b'),5:(1000,'d'),6:(5000,'d'),7:(100,'d'),8:(10,'d'),9:(1e5,'d'),10:(1,'d'),11:(0,'z')}
pk=(AX>=900)&(AX<=1350); bw=(AX>=1400)&(AX<=1600)
int_all=sum((lambda W:(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0))(np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")) for e in EXC)
rows=[]
for w in range(len(int_all)):
    r=w//NCOL; c=w%NCOL; cp,lab=(TOP if r<=3 else BOT)[c]
    rows.append(dict(row=r,col=c,half=("top" if r<=3 else "bot"),copies=cp,label=lab,int_all=int_all[w]))
df=pd.DataFrame(rows); df=df[df.int_all>0].reset_index(drop=True)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")
dd=df[df.label!="wrong"].copy()

# global left-right baseline (dose randomized => this is the spatial gradient)
b1,b0=np.polyfit(dd["col"],dd["int_rn"],1)
print(f"left-right spatial baseline: int_rn ~= {b0:.4f} {b1:+.4f}*col")
dd["detr"]=dd["int_rn"]-(b0+b1*dd["col"])+1.0

print("\n--- DOSE after left-right detrend ---")
dd["lc"]=np.log10(dd["copies"]+1)
order=[0,0.1,1,10,50,100,500,1000,5000,1e4,1e5]
for cp in order:
    s=dd[dd.copies==cp]; print(f"  {cp:>8g}: raw={s.int_rn.mean():.3f}  detr={s.detr.mean():.3f}  n={len(s)}")
rho,p=spearmanr(dd["lc"],dd["detr"]); print(f"Spearman(copies, detrended) = {rho:+.2f} p={p:.2g}")
# top vs bottom agreement on detrended
prof=dd.groupby(["copies","half"])["detr"].mean().unstack().dropna()
rt,pt=spearmanr(prof["top"],prof["bot"]); print(f"top-vs-bottom detrended profile Spearman = {rt:+.2f} p={pt:.2g}")

print("\n--- SPECIFICITY after left-right detrend ---")
for lab,name,cols in [('b','blank_noRNA (no RNA)   cols1,4','') ,('z','zero_bg (0+randomRNA) cols6,11','')]:
    s=dd[dd.label==lab]; print(f"  {name}: raw={s.int_rn.mean():.3f}  detr={s.detr.mean():.3f}  n={len(s)} (cols {sorted(s.col.unique())})")
b=dd[dd.label=='b']["detr"]; z=dd[dd.label=='z']["detr"]
print(f"  zero_bg vs blank_noRNA, DETRENDED: MWU p={mannwhitneyu(z,b).pvalue:.2g}  (delta={z.mean()-b.mean():+.3f})")
# also compare within same column-neighbourhood: paired by half
for h in ("top","bot"):
    bb=dd[(dd.label=='b')&(dd.half==h)]; zz=dd[(dd.label=='z')&(dd.half==h)]
    print(f"    {h}: blank(col{bb.col.iloc[0]})={bb.int_rn.mean():.3f} vs zero_bg(col{zz.col.iloc[0]})={zz.int_rn.mean():.3f} raw | detr {bb.detr.mean():.3f} vs {zz.detr.mean():.3f}")
