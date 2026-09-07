#!/usr/bin/env python3
"""Same UPPER vs LOWER row split on the SERUM plate (where the high-copy response was robust).
Geometry: missing = col6 blank last row (grid pos 90); p=w if w<90 else w+1; row=p//12,col=p%12."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

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
    rows.append(dict(w=w,row=r,col=c,copies=COPIES[c],lc=np.log10(COPIES[c]+1),int_all=int_all[w]))
df=pd.DataFrame(rows); df=df[df.int_all>0].reset_index(drop=True)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

print("="*70); print("SERUM PER-ROW dose-response Spearman(log copies, raw ∫I) across cols"); print("="*70)
print(f"{'row':>3} {'mean∫I':>9} {'rho':>6} {'p':>8}")
for r in sorted(df.row.unique()):
    s=df[df.row==r]; rho,p=spearmanr(s["lc"],s["int_all"])
    print(f"{r:>3} {s.int_all.mean():>9.3g} {rho:>+6.2f} {p:>8.2g}")

def dose_group(sub,label):
    lo=sub[sub.copies<=500]
    b1,b0=np.polyfit(lo["col"],lo["int_rn"],1)
    sub=sub.copy(); sub["detr"]=sub["int_rn"]-(b0+b1*sub["col"])+1.0
    rr,pr=spearmanr(sub["copies"],sub["int_rn"]); rd,pd_=spearmanr(sub["copies"],sub["detr"])
    hi=sub[sub.copies>=1e3]["detr"]; bl=sub[sub.copies==0]["detr"]; mid=sub[(sub.copies>=10)&(sub.copies<=500)]["detr"]
    ph=mannwhitneyu(hi,mid).pvalue if len(hi) and len(mid) else np.nan
    Fd=anova_lm(smf.ols("detr ~ 1",data=sub).fit(),smf.ols("detr ~ C(copies)",data=sub).fit())["Pr(>F)"].iloc[1]
    print(f"  {label:12s} monotonic raw ρ={rr:+.2f} p={pr:.2g} | detr ρ={rd:+.2f} p={pd_:.2g} "
          f"|| high≥1e3 vs mid(10-500) detr: hi={hi.mean():.3f} mid={mid.mean():.3f} MWU p={ph:.2g} | C(copies) detr p={Fd:.2g}")
    return sub

print("\n"+"="*70); print("SERUM UPPER (rows 0-3) vs LOWER (rows 4-7)"); print("="*70)
up=dose_group(df[df.row<=3],"UPPER 0-3"); lo=dose_group(df[df.row>=4],"LOWER 4-7"); allg=dose_group(df,"ALL rows")

order=[0,0.1,1,10,20,50,100,500,1e3,1e4,1e5,1e6]
fig,ax=plt.subplots(figsize=(11,6)); xa={cp:i for i,cp in enumerate(order)}
for sub,lab,cl,mk in [(up,"upper rows 0-3","#1f77b4","o"),(lo,"lower rows 4-7","#d62728","s")]:
    g=sub.groupby("copies")["detr"].agg(["mean","sem"])
    ax.errorbar([xa[cp] for cp in order],[g.loc[cp,"mean"] for cp in order],[g.loc[cp,"sem"] for cp in order],
                marker=mk,ms=7,capsize=3,color=cl,lw=1.7,label=lab)
ax.axhline(1,color="k",lw=0.6,ls=":"); ax.set_xticks(range(len(order)))
ax.set_xticklabels(["0"]+[f"{c:g}" for c in order[1:]],rotation=45,fontsize=8)
ax.set_xlabel("STMN2-CE (copies/µL) in serum"); ax.set_ylabel("position-detrended row-norm ∫I")
ax.set_title("exp4 SERUM — dose-response, UPPER vs LOWER rows (spatial-detrended)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{BASE}/exp4_serum_rowdep.png",dpi=130,bbox_inches="tight")
print("\nSaved exp4_serum_rowdep.png")
