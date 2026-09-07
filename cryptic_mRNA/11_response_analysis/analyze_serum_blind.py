#!/usr/bin/env python3
"""exp3 SERUM — BLIND per-condition analysis (fitted + model-free), same pipeline as water.
8 rows x 12 columns row-major (col11 has 7 reps, last well missing). Conditions BLIND (C00..C11).
Drift (heating down rows) removed by row-normalization. Tests between-condition structure."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3/serum"; PREFIX="serum__"
EXC=[570,640,670,750,780]; NCOL=12
GOOD=["ch8_3","ch10_2","ch9_4","ch8_4","ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]  # frac r2>.95 ~1.0

# ---- model-free integrated intensity per well ----
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
fit=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_serum.csv")
fit["int_all"]=[int_all[int(r.w)] for _,r in fit.iterrows()]   # index model-free ∫I by file-well w
fit["fit_mean"]=fit[[f"{ch}_gauss_max" for ch in GOOD]].mean(axis=1)
df=fit[fit.int_all>0].reset_index(drop=True)
df["fit_rn"]=df["fit_mean"]/df.groupby("row")["fit_mean"].transform("mean")
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")
CONDS=sorted(df["cond"].unique()); xpos={c:i for i,c in enumerate(CONDS)}

print("="*74); print("SERUM blind: between-condition structure after row(heating)-drift removal"); print("="*74)
for lab,col in [("fitted mean","fit_rn"),("model-free ∫I","int_rn")]:
    F=anova_lm(smf.ols(f"{col} ~ 1",data=df).fit(),smf.ols(f"{col} ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
    piv=df.pivot_table(index="row",columns="cond",values=col).apply(lambda s:s.fillna(s.median()))
    fr=friedmanchisquare(*[piv[c].values for c in CONDS])
    print(f"  {lab:14s} C(cond) F-test p={F:.2g}   Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")
print("\nper-BLIND-condition intensity (model-free, sorted):")
cm=df.groupby("cond")["int_rn"].agg(["mean","sem"])
for c in cm["mean"].sort_values(ascending=False).index:
    print(f"   {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}")

# ---- figure ----
fig,ax=plt.subplots(figsize=(12,6))
for col,mk,lc,lab in [("fit_rn","o","#1f77b4","fitted mean (10 chir)"),("int_rn","s","#ff7f0e","model-free ∫I")]:
    g=df.groupby("cond")[col]; m=g.mean(); se=g.sem()
    ax.errorbar([xpos[c] for c in CONDS],m[CONDS].values,yerr=se[CONDS].values,
                marker=mk,ms=7,capsize=3,color=lc,lw=1.8,label=lab)
ax.axhline(1,color="k",lw=0.6,ls=":")
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CONDS)
ax.set_xlabel("BLIND condition (plate column index)")
ax.set_ylabel("row-normalized intensity (heating drift removed)")
ax.set_title("exp3 SERUM — per-condition intensity (BLIND)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{BASE}/serum_intensity_by_condition_blind.png",dpi=130,bbox_inches="tight")
print("\nSaved serum_intensity_by_condition_blind.png")
