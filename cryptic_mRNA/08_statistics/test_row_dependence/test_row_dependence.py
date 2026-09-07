#!/usr/bin/env python3
"""Test user's observation: STMN2-CE concentration trend in WATER is more evident in LOWER rows
(later-imaged, brighter, higher SNR). Per-row and upper-vs-lower-half dose-response, raw AND after
removing the left-right spatial gradient (spatial baseline fit within each row-group from <=200cp cols)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/water"; PREFIX="STMN_22_WATER__"
EXC=[570,640,670,750,780]; NCOL=12
COPIES={0:1e6,1:10.,2:1e5,3:100.,4:20.,5:200.,6:0.,7:1e3,8:1.,9:1e4,10:50.,11:0.1}
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
rows=[]
for w in range(len(int_all)):
    r=w//NCOL; c=w%NCOL
    rows.append(dict(w=w,row=r,col=c,copies=COPIES[c],lc=np.log10(COPIES[c]+1),int_all=int_all[w]))
df=pd.DataFrame(rows)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

print("="*70); print("PER-ROW dose-response  Spearman(log10(copies+1), raw ∫I) across 12 cols"); print("="*70)
print(f"{'row':>3} {'mean∫I':>9} {'rho':>6} {'p':>8}   (row 0=top/first-imaged/dim ... 7=bottom/bright)")
for r in sorted(df.row.unique()):
    s=df[df.row==r]; rho,p=spearmanr(s["lc"],s["int_all"])
    print(f"{r:>3} {s.int_all.mean():>9.3g} {rho:>+6.2f} {p:>8.2g}")

def dose_group(sub,label):
    # spatial baseline from <=200cp columns within this group, then detrend
    lo=sub[sub.copies<=200]
    b1,b0=np.polyfit(lo["col"],lo["int_rn"],1)
    sub=sub.copy(); sub["detr"]=sub["int_rn"]-(b0+b1*sub["col"])+1.0
    rr,pr=spearmanr(sub["copies"],sub["int_rn"]); rd,pd_=spearmanr(sub["copies"],sub["detr"])
    # categorical: reproducible up-and-down concentration pattern (non-monotonic) via C(copies) F-test
    Fr=anova_lm(smf.ols("int_rn ~ 1",data=sub).fit(),smf.ols("int_rn ~ C(copies)",data=sub).fit())["Pr(>F)"].iloc[1]
    Fd=anova_lm(smf.ols("detr ~ 1",data=sub).fit(),smf.ols("detr ~ C(copies)",data=sub).fit())["Pr(>F)"].iloc[1]
    print(f"  {label:12s} monotonic: raw ρ={rr:+.2f} p={pr:.2g} | detr ρ={rd:+.2f} p={pd_:.2g}"
          f"  || categorical C(copies) F-test: raw p={Fr:.2g} | detr p={Fd:.2g}  (slope={b1:+.4f})")
    return sub

print("\n"+"="*70); print("UPPER (rows 0-3) vs LOWER (rows 4-7) pooled dose-response"); print("="*70)
up=dose_group(df[df.row<=3],"UPPER 0-3"); lo=dose_group(df[df.row>=4],"LOWER 4-7")
allg=dose_group(df,"ALL rows")

# also: lower-half detrended per-concentration table
print("\nLOWER rows (4-7), position-detrended, per concentration:")
order=[0,0.1,1,10,20,50,100,200,1e3,1e4,1e5,1e6]
g=lo.groupby("copies")["detr"].agg(["mean","sem"])
for cp in order:
    if cp in g.index: print(f"   {cp:>8g}: {g.loc[cp,'mean']:.3f} +/- {g.loc[cp,'sem']:.3f}")

# ---- figure: dose curves upper vs lower (detrended) ----
fig,ax=plt.subplots(figsize=(11,6)); xa={cp:i for i,cp in enumerate(order)}
for sub,lab,cl,mk in [(up,"upper rows 0-3","#1f77b4","o"),(lo,"lower rows 4-7","#d62728","s")]:
    g=sub.groupby("copies")["detr"].agg(["mean","sem"])
    ax.errorbar([xa[cp] for cp in order],[g.loc[cp,"mean"] for cp in order],[g.loc[cp,"sem"] for cp in order],
                marker=mk,ms=7,capsize=3,color=cl,lw=1.7,label=lab)
ax.axhline(1,color="k",lw=0.6,ls=":"); ax.set_xticks(range(len(order)))
ax.set_xticklabels(["0"]+[f"{c:g}" for c in order[1:]],rotation=45,fontsize=8)
ax.set_xlabel("STMN2-CE (copies/µL)"); ax.set_ylabel("position-detrended row-norm ∫I")
ax.set_title("exp4 WATER — dose-response, UPPER vs LOWER rows (spatial-detrended)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{BASE}/exp4_water_rowdep.png",dpi=130,bbox_inches="tight")
print("\nSaved exp4_water_rowdep.png")
