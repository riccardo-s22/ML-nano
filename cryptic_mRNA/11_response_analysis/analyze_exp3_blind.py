#!/usr/bin/env python3
"""
exp3 BLIND per-condition analysis (same analysis types as exp2).
Plate: water_2h, 8 replicate rows x 11 condition-columns (row-major). Conditions kept BLIND (C00..C10).
- Drift (acquisition heating down the 8 rows) removed by row-normalization within the plate.
- Fig 1: per-chirality row-normalized gauss_max per condition (clean chiralities) + across-chirality mean.
- Fig 2: (7,5) Delta-lambda per condition (paper readout), drift-removed center residual.
- Stats: is there condition-dependent structure beyond drift? OLS F-test C(cond) and Friedman (rows=blocks).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3"
df=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp3.csv")
GOOD=["ch8_3","ch7_5","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6"]   # frac r2>0.95 >= .94 here
CONDS=sorted(df["cond"].unique())        # C00..C10 (blind)
xpos={c:i for i,c in enumerate(CONDS)}

def rownorm(col):
    return df[col]/df.groupby("row")[col].transform("mean")

# ---------- stats: between-condition structure after drift removal ----------
print("="*76)
print("Q: after removing row (heating) drift, is there CONDITION-dependent brightness")
print("   structure? (blind - conditions are just columns C00..C10)")
print("="*76)
recs=[]
for ch in GOOD:
    df["rn"]=rownorm(f"{ch}_gauss_max")
    m0=smf.ols("rn ~ 1",data=df).fit()
    m1=smf.ols("rn ~ C(cond)",data=df).fit()
    p=anova_lm(m0,m1)["Pr(>F)"].iloc[1]
    print(f"  {ch:7s}  C(cond) F-test p={p:.2g}")
    for _,r in df.iterrows():
        recs.append(dict(rn=r["rn"],cond=r["cond"],row=r["row"],ch=ch))
L=pd.DataFrame(recs)
# across-chirality mean per well
df["rn_mean"]=np.mean([rownorm(f"{ch}_gauss_max").values for ch in GOOD],axis=0)
mF=anova_lm(smf.ols("rn_mean ~ 1",data=df).fit(),
            smf.ols("rn_mean ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
print(f"\n  ACROSS-CHIRALITY MEAN  C(cond) F-test p={mF:.2g}")
# Friedman: rows(8) as blocks, conditions as treatments, on across-chirality mean
piv=df.pivot_table(index="row",columns="cond",values="rn_mean")
fr=friedmanchisquare(*[piv[c].values for c in CONDS])
print(f"  Friedman (rows=blocks): chi2={fr.statistic:.2f} p={fr.pvalue:.2g}")
print("\n  across-chirality mean row-norm intensity per BLIND condition (sorted):")
cm=df.groupby("cond")["rn_mean"].agg(["mean","sem"])
for c in CONDS:
    bar="#"*int(round((cm.loc[c,'mean']-0.9)*200)) if cm.loc[c,'mean']>0.9 else ""
    print(f"    {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}  {bar}")

# ---------- Fig 1: brightness per condition ----------
COLORS=plt.cm.tab10(np.linspace(0,1,len(GOOD)))
fig,ax=plt.subplots(figsize=(11,6))
for ci,ch in enumerate(GOOD):
    df["rn"]=rownorm(f"{ch}_gauss_max")
    g=df.groupby("cond")["rn"]; m=g.mean(); se=g.sem()
    x=[xpos[c] for c in CONDS]
    ax.errorbar(x,m[CONDS].values,yerr=se[CONDS].values,marker="o",ms=4,capsize=2,
                color=COLORS[ci],lw=1.1,alpha=0.7,label=ch)
# across-chirality mean bold
gm=df.groupby("cond")["rn_mean"]; m=gm.mean(); se=gm.sem()
ax.errorbar([xpos[c] for c in CONDS],m[CONDS].values,yerr=se[CONDS].values,
            marker="s",ms=8,capsize=4,color="k",lw=2.4,zorder=6,label="MEAN (7 chir)")
ax.axhline(1,color="k",lw=0.6,ls=":")
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CONDS,rotation=0)
ax.set_xlabel("BLIND condition (plate column index)"); ax.set_ylabel("row-normalized gauss_max (drift removed)")
ax.set_title(f"exp3 water_2h — per-condition fitted brightness (BLIND)   "
             f"across-chir C(cond) F-test p={mF:.2g}")
ax.grid(alpha=0.3); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(f"{BASE}/exp3_brightness_by_condition_blind.png",dpi=130,bbox_inches="tight")
print("\nSaved exp3_brightness_by_condition_blind.png")

# ---------- Fig 2: (7,5) Delta-lambda per condition ----------
fig2,ax2=plt.subplots(figsize=(11,5))
cc="ch7_5_gauss_em_center"
df["ccdt"]=df[cc]-df.groupby("row")[cc].transform("mean")   # drift-removed center residual
s=df[df["ch7_5_gauss_r2"]>0.95]
base=s.groupby("cond")["ccdt"].mean().mean()   # reference = grand mean (blind, no dose-0 known)
g=s.groupby("cond")["ccdt"]; m=g.mean(); se=g.sem()
dl=-(m-base)   # Dlam = lam0 - lam (blue +), referenced to grand mean
ax2.errorbar([xpos[c] for c in CONDS],dl[CONDS].values,yerr=se[CONDS].values,
             marker="o",ms=6,capsize=3,color="#d62728",lw=1.5)
ax2.axhline(0,color="k",lw=0.6,ls=":")
ax2.axhspan(-1.62/2,1.62/2,color="gray",alpha=0.12)
ax2.text(0.02,0.95,"gray band = ±½ pixel (1.62 nm/pt)",transform=ax2.transAxes,fontsize=8,va="top")
ax2.set_xticks(range(len(CONDS))); ax2.set_xticklabels(CONDS)
ax2.set_ylim(-2.5,2.5); ax2.grid(alpha=0.3)
ax2.set_xlabel("BLIND condition (plate column index)"); ax2.set_ylabel("Δλ = λ0−λ  (nm, blue +, vs grand mean)")
ax2.set_title("exp3 (7,5) peak shift per condition (BLIND)")
fig2.tight_layout(); fig2.savefig(f"{BASE}/exp3_ch75_dlambda_by_condition_blind.png",dpi=130,bbox_inches="tight")
print("Saved exp3_ch75_dlambda_by_condition_blind.png")
