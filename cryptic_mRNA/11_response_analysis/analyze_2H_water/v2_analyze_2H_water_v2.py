#!/usr/bin/env python3
"""
exp2 / 2H_WATER  -- CORRECTED layout (per user).
Plate = 6 concentration-columns x 8 replicate rows, exported ROW-MAJOR:
each 6 consecutive data-cols = one plate row; reshape(512, 8rows, 6cols).
The big monotonic trend across the 8 ROWS is a position/drift artifact (nuisance block).
Concentration is the within-row 6, RANDOMIZED:
  col index -> conc:  0:'0'  1:'100'  2:'0.1'  3:'10'  4:'1'  5:'water'
Goal: concentration effect AFTER removing the row block.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare
import statsmodels.formula.api as smf

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2/2H_WATER"
EXC=[570,640,670,750,780]
CONC=[0,100,0.1,10,1,np.nan]                  # nan = water
CLAB=["0","100","0.1","10","1","water"]
ORDER=[0,2,4,3,1,5]                            # display order: 0,0.1,1,10,100,water

def load(e):
    a=np.loadtxt(f"{HERE}/STMN_2_1H__{e}.txt"); return a[:,0], a[:,2:]
wl,_=load(570)

def cube(e):
    _,d=load(e); return d.reshape(d.shape[0],8,6)        # emission x 8row x 6conc
def integ(d): return np.trapz(np.clip(d,0,None),wl,axis=0)

print("="*74)
print("Concentration effect WITHIN row (row block removed) — 2H water")
print("conc order shown: 0, 0.1, 1, 10, 100, water")
print("="*74)
for e in EXC:
    d=cube(e); tot=integ(d)                              # 8row x 6conc
    rowmean=tot.mean(1,keepdims=True)                    # per-row mean over 6 conc
    rel=tot/rowmean                                      # row-normalized
    m=rel.mean(0); sd=rel.std(0)
    # build long df for OLS with row block
    df=pd.DataFrame({"y":np.log10(tot.ravel()),
                     "row":np.repeat(np.arange(8),6),
                     "ci":np.tile(np.arange(6),8)})
    df["conc"]=df["ci"].map(dict(enumerate(CONC)))
    df["clab"]=df["ci"].map(dict(enumerate(CLAB)))
    # dose arm only (exclude water), ordered factor via log conc; 0 -> rank handling
    dose=df[df.clab!="water"].copy()
    dose["rank"]=dose["clab"].map({"0":0,"0.1":1,"1":2,"10":3,"100":4})
    # Spearman of row-centered signal vs dose rank
    dose["yc"]=dose["y"]-dose.groupby("row")["y"].transform("mean")
    rho,p=spearmanr(dose["rank"],dose["yc"])
    # two-way OLS: does concentration matter at all, controlling row?
    m_full=smf.ols("y ~ C(row) + C(clab)",data=df).fit()
    m_row =smf.ols("y ~ C(row)",data=df).fit()
    from statsmodels.stats.anova import anova_lm
    fp=anova_lm(m_row,m_full)["Pr(>F)"].iloc[1]
    # Friedman across the 6 conc with rows as blocks
    fr=friedmanchisquare(*[tot[:,c] for c in range(6)])
    print(f"\nex{e}:")
    print("  row-normalized mean +/- sd (conc 0,0.1,1,10,100,water):")
    print("   ", "  ".join(f"{CLAB[i]}:{m[i]:.3f}±{sd[i]:.3f}" for i in ORDER))
    print(f"  Spearman(dose rank, row-centered logI), arm 0..100: rho={rho:+.2f} p={p:.2g}")
    print(f"  conc effect controlling row (F-test add C(clab)): p={fp:.2g}")
    print(f"  Friedman (6 conc, 8 row-blocks): chi2={fr.statistic:.2f} p={fr.pvalue:.2g}")
    # water vs 0 (within-row paired)
    w=tot[:,5]; z=tot[:,0]; from scipy.stats import wilcoxon
    try: wp=wilcoxon(w,z).pvalue
    except Exception: wp=np.nan
    print(f"  water vs 0 (paired by row): water/0 = {(w/z).mean():.2f}x  Wilcoxon p={wp:.2g}")

# ---- Figure: row-normalized concentration response, mean over rows+excitations ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
allrel=[]
for i,e in enumerate(EXC):
    d=cube(e); tot=integ(d); rel=tot/tot.mean(1,keepdims=True)
    allrel.append(rel)
    m=rel.mean(0); sd=rel.std(0)/np.sqrt(8)
    ax[0].errorbar(range(6),[m[j] for j in ORDER],yerr=[sd[j] for j in ORDER],
                   marker="o",capsize=3,label=f"ex{e}")
ax[0].set_xticks(range(6)); ax[0].set_xticklabels([CLAB[j] for j in ORDER])
ax[0].set_xlabel("STMN2-CE concentration (blind units)"); ax[0].set_ylabel("row-normalized intensity")
ax[0].axhline(1,color="k",lw=0.6,ls=":"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[0].set_title("Concentration effect (row block removed)")

# pooled across excitations
pooled=np.concatenate(allrel,0)  # (40, 6)
mp=pooled.mean(0); sp=pooled.std(0)/np.sqrt(pooled.shape[0])
ax[1].bar(range(6),[mp[j] for j in ORDER],yerr=[sp[j] for j in ORDER],capsize=4,
          color=["#444"]+["#2c7fb8"]*4+["#888"])
ax[1].set_xticks(range(6)); ax[1].set_xticklabels([CLAB[j] for j in ORDER])
ax[1].axhline(1,color="k",lw=0.6,ls=":"); ax[1].set_ylim(0.9,1.1)
ax[1].set_xlabel("STMN2-CE concentration"); ax[1].set_ylabel("row-normalized intensity (pooled)")
ax[1].set_title("Pooled over 5 excitations x 8 rows")
fig.suptitle("2H water: STMN2-CE concentration effect within rows (artifact-corrected)",y=1.02)
fig.tight_layout(); fig.savefig(f"{HERE}/conc_effect_2H_water.png",dpi=130,bbox_inches="tight")
print("\nSaved conc_effect_2H_water.png")
