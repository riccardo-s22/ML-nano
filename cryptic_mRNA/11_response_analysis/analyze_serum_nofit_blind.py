#!/usr/bin/env python3
"""exp3 SERUM — BLIND model-free (NO fitting) per-condition analysis (mirrors water no-fit).
8 rows x 12 cols row-major (col11=7 reps). Intensity = integrated emission 900-1350nm minus flat
baseline, per excitation + combined. Drift removed by row-normalization. BLIND conditions C00..C11."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp3/serum"; PREFIX="serum__"
EXC=[570,640,670,750,780]; NCOL=12
PEAK=(900,1350); BASE_WIN=(1400,1600)

em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=PEAK[0])&(em<=PEAK[1]); bw=(em>=BASE_WIN[0])&(em<=BASE_WIN[1])
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
nwell=inten[EXC[0]].shape[0]
rows=[]
for w in range(nwell):
    p=w+1; r=p//NCOL; c=p%NCOL; d=dict(row=r,col=c,cond=f"C{c:02d}")  # first grid cell missing
    for e in EXC: d[f"int_{e}"]=inten[e][w]
    d["int_all"]=sum(inten[e][w] for e in EXC); rows.append(d)
df=pd.DataFrame(rows)
bad=df["int_all"]<=0
if bad.any(): print(f"[dropping {bad.sum()} unphysical well(s): {df.loc[bad,['cond','row']].to_dict('records')}]")
df=df[~bad].reset_index(drop=True)
CONDS=sorted(df["cond"].unique()); xpos={c:i for i,c in enumerate(CONDS)}
for col in [f"int_{e}" for e in EXC]+["int_all"]:
    df["rn_"+col]=df[col]/df.groupby("row")[col].transform("mean")

print("="*74); print("SERUM MODEL-FREE (no fitting): ∫I 900-1350nm, drift removed. BLIND C00..C11."); print("="*74)
mF=anova_lm(smf.ols("rn_int_all ~ 1",data=df).fit(),smf.ols("rn_int_all ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
piv=df.pivot_table(index="row",columns="cond",values="rn_int_all").apply(lambda s:s.fillna(s.median()))
fr=friedmanchisquare(*[piv[c].values for c in CONDS])
print(f"combined (all exc)  C(cond) F-test p={mF:.2g}   Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")
print("per-excitation C(cond) F-test:")
for e in EXC:
    p=anova_lm(smf.ols(f"rn_int_{e} ~ 1",data=df).fit(),smf.ols(f"rn_int_{e} ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
    print(f"   {e} nm  p={p:.2g}")
print("\ncombined row-norm intensity per BLIND condition (sorted):")
cm=df.groupby("cond")["rn_int_all"].agg(["mean","sem"])
for c in cm["mean"].sort_values(ascending=False).index:
    print(f"   {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}")

fig,ax=plt.subplots(figsize=(12,6))
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
ax.set_title(f"exp3 SERUM — MODEL-FREE per-condition intensity (BLIND)   combined F-test p={mF:.2g}")
ax.grid(alpha=0.3); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(f"{BASE}/serum_intensity_nofit_by_condition_blind.png",dpi=130,bbox_inches="tight")
print("\nSaved serum_intensity_nofit_by_condition_blind.png")
