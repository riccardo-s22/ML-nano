#!/usr/bin/env python3
"""
exp3 BLIND per-condition analysis WITHOUT peak fitting (model-free cross-check).
Plate water_2h, 8 replicate rows x 11 condition-columns (row-major). col0=em, col1=ignore, cols2+=wells.
Intensity metric per well = integrated emission (sum over 900-1350 nm) - flat baseline, per excitation,
plus a combined sum across all 5 excitations. Drift (heating down rows) removed by row-normalization.
Fig: per-excitation + combined row-normalized integrated intensity per BLIND condition.
Stats: OLS F-test C(cond) and Friedman (rows=blocks) on the combined metric.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3"
PREFIX="water_2h__"; EXC=[570,640,670,750,780]
NCOL=11; NROW=8
PEAK=(900,1350)     # integration window (nm) covering the main chiralities
BASE_WIN=(1400,1600)# flat off-peak region for baseline

# load & integrate
em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None:
        em=a[:,0]; pk=(em>=PEAK[0])&(em<=PEAK[1]); bw=(em>=BASE_WIN[0])&(em<=BASE_WIN[1])
    W=a[:,2:]                                   # em x wells
    bl=W[bw,:].mean(axis=0)                      # per-well flat baseline
    inten[e]=(W[pk,:]-bl).sum(axis=0)            # baseline-subtracted integrated intensity

nwell=inten[EXC[0]].shape[0]
assert nwell==NCOL*NROW
rows=[]
for w in range(nwell):
    r=w//NCOL; c=w%NCOL
    d=dict(row=r,col=c,cond=f"C{c:02d}")
    for e in EXC: d[f"int_{e}"]=inten[e][w]
    d["int_all"]=sum(inten[e][w] for e in EXC)
    rows.append(d)
df=pd.DataFrame(rows)
# drop unphysical wells (negative integrated intensity = dead/bubble/failed baseline)
bad=df["int_all"]<=0
if bad.any():
    print(f"[dropping {bad.sum()} unphysical well(s) with integrated intensity<=0: "
          f"{df.loc[bad,['cond','row']].to_dict('records')}]")
df=df[~bad].reset_index(drop=True)
CONDS=sorted(df["cond"].unique()); xpos={c:i for i,c in enumerate(CONDS)}

def rownorm(col):
    return df[col]/df.groupby("row")[col].transform("mean")

for col in [f"int_{e}" for e in EXC]+["int_all"]:
    df["rn_"+col]=rownorm(col)

# ---- stats on combined metric ----
print("="*76)
print("MODEL-FREE (no fitting): integrated NIR intensity 900-1350 nm, baseline-subtracted,")
print("drift removed by row-normalization. Conditions BLIND (C00..C10).")
print("="*76)
mF=anova_lm(smf.ols("rn_int_all ~ 1",data=df).fit(),
            smf.ols("rn_int_all ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
piv=df.pivot_table(index="row",columns="cond",values="rn_int_all")
piv=piv.apply(lambda s:s.fillna(s.median()))   # keep blocks balanced after dropping bad well
fr=friedmanchisquare(*[piv[c].values for c in CONDS])
print(f"combined (all exc)  C(cond) F-test p={mF:.2g}   Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")
print("per-excitation C(cond) F-test:")
for e in EXC:
    p=anova_lm(smf.ols(f"rn_int_{e} ~ 1",data=df).fit(),
               smf.ols(f"rn_int_{e} ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
    print(f"   {e} nm  p={p:.2g}")
print("\ncombined row-norm integrated intensity per BLIND condition (sorted by value):")
cm=df.groupby("cond")["rn_int_all"].agg(["mean","sem"])
for c in cm["mean"].sort_values(ascending=False).index:
    print(f"   {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}")

# ---- figure ----
fig,ax=plt.subplots(figsize=(11,6))
cmap=plt.cm.viridis(np.linspace(0,1,len(EXC)))
for i,e in enumerate(EXC):
    g=df.groupby("cond")[f"rn_int_{e}"]; m=g.mean(); se=g.sem()
    ax.errorbar([xpos[c] for c in CONDS],m[CONDS].values,yerr=se[CONDS].values,
                marker="o",ms=3,capsize=2,color=cmap[i],lw=1.0,alpha=0.6,label=f"{e} nm")
g=df.groupby("cond")["rn_int_all"]; m=g.mean(); se=g.sem()
ax.errorbar([xpos[c] for c in CONDS],m[CONDS].values,yerr=se[CONDS].values,
            marker="s",ms=8,capsize=4,color="k",lw=2.4,zorder=6,label="combined (all exc)")
ax.axhline(1,color="k",lw=0.6,ls=":")
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CONDS)
ax.set_xlabel("BLIND condition (plate column index)")
ax.set_ylabel("row-normalized integrated intensity (drift removed)")
ax.set_title(f"exp3 water_2h — MODEL-FREE per-condition intensity (BLIND)   "
             f"combined C(cond) F-test p={mF:.2g}")
ax.grid(alpha=0.3); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(f"{BASE}/exp3_intensity_nofit_by_condition_blind.png",dpi=130,bbox_inches="tight")
print("\nSaved exp3_intensity_nofit_by_condition_blind.png")
