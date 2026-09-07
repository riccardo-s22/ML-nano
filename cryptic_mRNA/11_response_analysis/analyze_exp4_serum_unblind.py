#!/usr/bin/env python3
"""exp4 SERUM UNBLINDED (STMN_22 = 22nt GT15-STMN2-CE sensor, in serum).
Plate order (1-indexed col -> STMN2-CE mRNA copies/uL): 1=1e6, 2=0.1, 3=1e4, 4=1, 5=1e3, 6=10,
 7=0, 8=20, 9=50, 10=100, 11=500, 12=1e5.  => 0-indexed col c -> copies:
 c0=1e6,c1=0.1,c2=1e4,c3=1,c4=1e3,c5=10,c6=0(blank),c7=20,c8=50,c9=100,c10=500,c11=1e5.
Column 7 (1-idx) = 0 copies is the SHORT column (missing grid pos 90). Row-mean normalization only."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/serum"; PREFIX="STMN_22_Serum__"
EXC=[570,640,670,750,780]; NCOL=12; MISS=90
GOOD=["ch8_3","ch6_5","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
COPIES={0:1e6,1:0.1,2:1e4,3:1.,4:1e3,5:10.,6:0.,7:20.,8:50.,9:100.,10:500.,11:1e5}

# model-free ∫I
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp4_serum.csv")
fit["int_all"]=[int_all[int(r.w)] for _,r in fit.iterrows()]
fit["fit_mean"]=fit[[f"{ch}_gauss_max" for ch in GOOD]].mean(axis=1)
fit["copies"]=fit["col"].map(COPIES)
df=fit[fit.int_all>0].reset_index(drop=True)
df["fit_rn"]=df["fit_mean"]/df.groupby("row")["fit_mean"].transform("mean")
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

print("="*78); print("exp4 SERUM UNBLINDED — STMN2-CE titration, 22nt sensor (row-normalized)"); print("="*78)
order=[0,0.1,1,10,20,50,100,500,1e3,1e4,1e5,1e6]
print(f"{'copies/uL':>10} | {'fitted_rn':>9} {'sem':>6} | {'modelfree_rn':>12} {'sem':>6} | n")
for cp in order:
    s=df[df.copies==cp]
    print(f"{cp:>10g} | {s.fit_rn.mean():>9.3f} {s.fit_rn.sem():>6.3f} | {s.int_rn.mean():>12.3f} {s.int_rn.sem():>6.3f} | {len(s)}")

print("\nDose-response Spearman over ALL wells (copies vs intensity):")
for lab,c in [("fitted","fit_rn"),("model-free","int_rn")]:
    rho,p=spearmanr(df["copies"],df[c]); print(f"  {lab:11s} rho={rho:+.2f} p={p:.2g}")
# nonzero only, log copies
nz=df[df.copies>0]
for lab,c in [("fitted","fit_rn"),("model-free","int_rn")]:
    rho,p=spearmanr(np.log10(nz["copies"]),nz[c]); print(f"  {lab:11s} (copies>0, log) rho={rho:+.2f} p={p:.2g}")

# high (>=1e3) vs blank(0), and vs mid (10-500)
blank=df[df.copies==0]["int_rn"]
high=df[df.copies>=1e3]["int_rn"]; mid=df[(df.copies>=10)&(df.copies<=500)]["int_rn"]
low=df[(df.copies>0)&(df.copies<10)]["int_rn"]
print("\ngrouped model-free (int_rn):")
for lab,g in [("blank 0cp",blank),("low 0.1-1cp",low),("mid 10-500cp",mid),("high >=1e3cp",high)]:
    print(f"   {lab:14s} mean={g.mean():.3f} sem={g.sem():.3f} n={len(g)}")
print(f"   MWU high vs blank: p={mannwhitneyu(high,blank).pvalue:.2g}")
print(f"   MWU high vs mid  : p={mannwhitneyu(high,mid).pvalue:.2g}")
print(f"   MWU mid  vs blank: p={mannwhitneyu(mid,blank).pvalue:.2g}")

# ---- figure ----
fig,ax=plt.subplots(figsize=(11,6))
xa={cp:i for i,cp in enumerate(order)}
for col,mk,lc,lab in [("fit_rn","o","#1f77b4","fitted mean (11 chir)"),("int_rn","s","#ff7f0e","model-free ∫I")]:
    g=df.groupby("copies")[col].agg(["mean","sem"])
    ax.errorbar([xa[cp] for cp in order],[g.loc[cp,"mean"] for cp in order],
                [g.loc[cp,"sem"] for cp in order],marker=mk,ms=7,capsize=3,color=lc,lw=1.6,label=lab)
ax.axhline(1,color="k",lw=0.6,ls=":")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["0\n(blank)"]+[f"{c:g}" for c in order[1:]])
ax.set_xlabel("STMN2-CE mRNA (copies/µL) in serum"); ax.set_ylabel("row-normalized intensity")
ax.set_title("exp4 SERUM UNBLINDED — 22nt sensor: STMN2-CE titration (row-drift removed)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{BASE}/exp4_serum_unblinded.png",dpi=130,bbox_inches="tight")
print("\nSaved exp4_serum_unblinded.png")
