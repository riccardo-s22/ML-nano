#!/usr/bin/env python3
"""exp4 SERUM (STMN_22 = 22nt GT15-STMN2-CE sensor) -- BLIND model-free per-condition analysis.
Grid 8 rows x 12 columns, row-major. LAST ROW of column 7 (0-indexed) is MISSING -> 95 wells
(col7 has 7 reps, all others 8). File well w maps to grid pos p = w if w<91 else w+1;
row=p//12, col=p%12  (missing grid pos = 7*12+7 = 91).
Model-free intensity = integrated emission 900-1350nm minus flat baseline (1400-1600), per exc + combined.
Conditions BLIND -> labelled C00..C11.  Explicitly tests BOTH row (top-down) and column (left-right) gradients."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/serum"; PREFIX="STMN_22_Serum__"
EXC=[570,640,670,750,780]; NCOL=12; MISS=91   # missing grid cell = row7,col7
PEAK=(900,1350); BASE_WIN=(1400,1600)

def gridpos(w): return w if w<MISS else w+1

em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=PEAK[0])&(em<=PEAK[1]); bw=(em>=BASE_WIN[0])&(em<=BASE_WIN[1])
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
nwell=inten[EXC[0]].shape[0]
rows=[]
for w in range(nwell):
    p=gridpos(w); r=p//NCOL; c=p%NCOL
    d=dict(w=w,gp=p,row=r,col=c,cond=f"C{c:02d}")
    for e in EXC: d[f"int_{e}"]=inten[e][w]
    d["int_all"]=sum(inten[e][w] for e in EXC); rows.append(d)
df=pd.DataFrame(rows)
print(f"n wells={nwell}  reps per column:",df.groupby('col').size().to_dict())
bad=df["int_all"]<=0
if bad.any(): print(f"[dropping {bad.sum()} unphysical well(s): {df.loc[bad,['cond','row']].to_dict('records')}]")
df=df[~bad].reset_index(drop=True)
CONDS=sorted(df["cond"].unique()); xpos={c:i for i,c in enumerate(CONDS)}

# ============ GRADIENT DIAGNOSTICS (RAW, before any normalization) ============
print("\n"+"="*74); print("GRADIENT DIAGNOSTICS on raw combined ∫I (no normalization yet)"); print("="*74)
# acquisition order = file well index w (raster). heating drift correlates with w.
rho_w,p_w=spearmanr(df["w"],df["int_all"])
print(f"raw ∫I vs acquisition order w (raster):  Spearman rho={rho_w:+.2f} p={p_w:.2g}")
# ROW gradient (top-down): mean per row + correlation
print("\nROW gradient (top-down): mean raw ∫I per row")
rm=df.groupby("row")["int_all"].agg(["mean","sem","size"])
for r in sorted(df.row.unique()): print(f"   row{r}: {rm.loc[r,'mean']:.4g} +/- {rm.loc[r,'sem']:.2g}  (n={int(rm.loc[r,'size'])})")
rho_r,p_r=spearmanr(df["row"],df["int_all"]); print(f"   Spearman(row, ∫I) rho={rho_r:+.2f} p={p_r:.2g}")
# COLUMN gradient (left-right) WITHIN row: remove row means first, then test residual vs col
df["res_rowdemean"]=df["int_all"]-df.groupby("row")["int_all"].transform("mean")
rho_c,p_c=spearmanr(df["col"],df["res_rowdemean"])
print(f"\nCOLUMN gradient (left-right, within-row residual): Spearman(col, resid) rho={rho_c:+.2f} p={p_c:.2g}")
# joint OLS: intensity ~ row + col (linear), to see independent slopes
mfit=smf.ols("int_all ~ row + col",data=df).fit()
print(f"   OLS ∫I ~ row + col :  row slope={mfit.params['row']:.4g} (p={mfit.pvalues['row']:.2g}) "
      f"col slope={mfit.params['col']:.4g} (p={mfit.pvalues['col']:.2g})")
print("   -> NOTE: col is confounded with CONDITION (each column is a different blind condition);")
print("      a col slope may be a real condition trend, not a spatial gradient. Reported for the user's check.")

# ============ normalization: row-mean only (top-down heating) ============
for col in [f"int_{e}" for e in EXC]+["int_all"]:
    df["rn_"+col]=df[col]/df.groupby("row")[col].transform("mean")

print("\n"+"="*74); print("BLIND per-condition (row-normalized, top-down drift removed)"); print("="*74)
mF=anova_lm(smf.ols("rn_int_all ~ 1",data=df).fit(),smf.ols("rn_int_all ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
piv=df.pivot_table(index="row",columns="cond",values="rn_int_all").apply(lambda s:s.fillna(s.median()))
fr=friedmanchisquare(*[piv[c].values for c in CONDS])
print(f"combined  C(cond) F-test p={mF:.2g}   Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")
print("per-excitation C(cond) F-test:")
for e in EXC:
    p=anova_lm(smf.ols(f"rn_int_{e} ~ 1",data=df).fit(),smf.ols(f"rn_int_{e} ~ C(cond)",data=df).fit())["Pr(>F)"].iloc[1]
    print(f"   {e} nm  p={p:.2g}")
print("\ncombined row-norm ∫I per BLIND condition (sorted, brightest first):")
cm=df.groupby("cond")["rn_int_all"].agg(["mean","sem"])
for c in cm["mean"].sort_values(ascending=False).index:
    print(f"   {c}:  {cm.loc[c,'mean']:.3f} +/- {cm.loc[c,'sem']:.3f}")
print("\nraw (NO normalization) combined ∫I per column mean (drift self-cancels per col):")
raw=df.groupby("cond")["int_all"].agg(["mean","sem"])
for c in CONDS: print(f"   {c}:  {raw.loc[c,'mean']:.4g} +/- {raw.loc[c,'sem']:.2g}")

# ============ figures ============
fig,(axA,axB)=plt.subplots(1,2,figsize=(16,6))
# panel A: gradient map (raw ∫I heat by row/col)
hm=df.pivot_table(index="row",columns="col",values="int_all")
im=axA.imshow(hm.values,aspect="auto",cmap="viridis")
axA.set_xticks(range(NCOL)); axA.set_xticklabels([f"C{c:02d}" for c in range(NCOL)],rotation=45,fontsize=7)
axA.set_yticks(range(8)); axA.set_yticklabels([f"r{r}" for r in range(8)])
axA.set_title(f"RAW ∫I heat-map  (row rho={rho_r:+.2f}, within-row col rho={rho_c:+.2f})")
axA.set_xlabel("column (blind condition)"); axA.set_ylabel("row")
fig.colorbar(im,ax=axA,fraction=0.046)
# panel B: per-condition row-normalized intensity
cmap=plt.cm.viridis(np.linspace(0,1,len(EXC)))
for i,e in enumerate(EXC):
    g=df.groupby("cond")[f"rn_int_{e}"]; m=g.mean(); se=g.sem()
    axB.errorbar([xpos[c] for c in CONDS],m[CONDS].values,yerr=se[CONDS].values,
                marker="o",ms=3,capsize=2,color=cmap[i],lw=1.0,alpha=0.6,label=f"{e} nm")
g=df.groupby("cond")["rn_int_all"]; m=g.mean(); se=g.sem()
axB.errorbar([xpos[c] for c in CONDS],m[CONDS].values,yerr=se[CONDS].values,
            marker="s",ms=8,capsize=4,color="k",lw=2.4,zorder=6,label="combined")
axB.axhline(1,color="k",lw=0.6,ls=":")
axB.set_xticks(range(len(CONDS))); axB.set_xticklabels(CONDS,rotation=45,fontsize=7)
axB.set_xlabel("BLIND condition (plate column)"); axB.set_ylabel("row-normalized ∫I (top-down drift removed)")
axB.set_title(f"BLIND per-condition intensity   combined F-test p={mF:.2g}")
axB.grid(alpha=0.3); axB.legend(fontsize=8,ncol=2)
fig.suptitle("exp4 SERUM (STMN_22 sensor) — BLIND per-condition + gradient diagnostics",y=1.02,fontsize=12)
fig.tight_layout(); fig.savefig(f"{BASE}/exp4_serum_blind.png",dpi=130,bbox_inches="tight")
print("\nSaved exp4_serum_blind.png")
