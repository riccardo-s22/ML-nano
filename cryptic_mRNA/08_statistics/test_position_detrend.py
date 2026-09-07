#!/usr/bin/env python3
"""Estimate the left-right SPATIAL gradient from the BELOW-THRESHOLD (near-baseline) columns only,
then subtract it from ALL columns to get a position-corrected dose-response. Key question: does the
high-copy (>=1e3) response survive after removing the left-right spatial trend? Decisive check =
c11 (1e5) which sits at the far RIGHT (dim-position zone) with only low neighbors."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/serum"; PREFIX="STMN_22_Serum__"
EXC=[570,640,670,750,780]; NCOL=12; MISS=90
COPIES={0:1e6,1:0.1,2:1e4,3:1.,4:1e3,5:10.,6:0.,7:20.,8:50.,9:100.,10:500.,11:1e5}
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
rows=[]
for w in range(len(int_all)):
    p=w if w<MISS else w+1; r=p//NCOL; c=p%NCOL
    rows.append(dict(row=r,col=c,copies=COPIES[c],int_all=int_all[w]))
df=pd.DataFrame(rows); df=df[df.int_all>0].reset_index(drop=True)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

# fit linear position baseline from below-threshold wells (own signal flat there)
low=df[df.copies<=500]
b1,b0=np.polyfit(low["col"],low["int_rn"],1)
print(f"spatial baseline (from <=500cp cols):  int_rn ~= {b0:.4f} {b1:+.4f}*col")
df["pos_base"]=b0+b1*df["col"]
df["int_detr"]=df["int_rn"]-df["pos_base"]+1.0   # position-corrected, recentred to 1

order=[0,0.1,1,10,20,50,100,500,1e3,1e4,1e5,1e6]
print(f"\n{'copies':>8} {'col':>3} {'raw_rn':>7} {'posbase':>7} {'detrended':>9}")
for cp in order:
    s=df[df.copies==cp]; c=int(s['col'].iloc[0])
    print(f"{cp:>8g} {c:>3} {s.int_rn.mean():>7.3f} {s.pos_base.mean():>7.3f} {s.int_detr.mean():>9.3f}")

print("\n--- after position detrend ---")
hi=df[df.copies>=1e3]["int_detr"]; bl=df[df.copies==0]["int_detr"]; mid=df[(df.copies>=10)&(df.copies<=500)]["int_detr"]
print(f"high>=1e3  mean={hi.mean():.3f}  vs blank mean={bl.mean():.3f}  MWU p={mannwhitneyu(hi,bl).pvalue:.2g}")
print(f"high>=1e3  vs mid(10-500) mean={mid.mean():.3f}  MWU p={mannwhitneyu(hi,mid).pvalue:.2g}")
rho,p=spearmanr(df["copies"],df["int_detr"]); print(f"Spearman(copies, detrended) rho={rho:+.2f} p={p:.2g}")
# decisive single point: c11 = 1e5 at far right, low neighbors
c11=df[df.copies==1e5]
print(f"\nDECISIVE: c11=1e5 (far-right, low neighbors): raw_rn={c11.int_rn.mean():.3f}, "
      f"pos_base={c11.pos_base.mean():.3f}, EXCESS over spatial baseline={c11.int_detr.mean()-1:+.3f}")
