#!/usr/bin/env python3
"""
SERUM plate analyzer. Layout: 8 rows x 5 columns, ROW-MAJOR (5 cols/row). Rows = technical
replicates carrying the acquisition heating drift; EXCLUDE the last (8th) row per user.
Concentration map for the 5 columns is UNKNOWN (water had 6: [0,100,0.1,10,1,water]) -> label
columns positionally col1..col5. Removes row drift (block + flexible) and tests the column effect.
Usage: FOLDER PREFIX TAG
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

FOLDER,PREFIX,TAG=sys.argv[1],sys.argv[2],sys.argv[3]
EXC=[570,640,670,750,780]; NCOL=5; NROW=8
def load(e):
    a=np.loadtxt(f"{FOLDER}/{PREFIX}{e}.txt"); return a[:,0],a[:,2:]
wl,_=load(570)
def tot(e):
    _,d=load(e); d=d.reshape(d.shape[0],NROW,NCOL)
    return np.trapz(np.clip(d,0,None),wl,axis=0)        # 8row x 5col

recs=[]
for e in EXC:
    T=tot(e)
    for r in range(NROW-1):        # DROP last row
        for c in range(NCOL):
            recs.append(dict(ex=e,row=r,col=c+1,acq=r*NCOL+c,y=np.log10(T[r,c])))
df=pd.DataFrame(recs)
df["yc"]=df["y"]-df.groupby(["ex","row"])["y"].transform("mean")

print("="*70); print(f"{TAG}: column effect, last row excluded, drift removed (7 reps x 5 col)"); print("="*70)
for e in EXC:
    s=df[df.ex==e]; mp=s.groupby("col")["yc"].mean()
    m0=smf.ols("y~C(row)",data=s).fit(); m1=smf.ols("y~C(row)+C(col)",data=s).fit()
    fp=anova_lm(m0,m1)["Pr(>F)"].iloc[1]
    rho,pp=spearmanr(s["col"],s["yc"])
    print(f"ex{e}: "+" ".join(f"c{c}:{10**mp[c]:.3f}" for c in range(1,6))+
          f" | colF p={fp:.2g}  monotonic-rho={rho:+.2f} p={pp:.2g}")

print("\n--- POOLED ---")
for tag,base in [("row block","C(ex)+C(row)"),("quintic drift","C(ex)+acq+I(acq**2)+I(acq**3)+I(acq**4)+I(acq**5)")]:
    m0=smf.ols(f"y~{base}",data=df).fit(); m1=smf.ols(f"y~{base}+C(col)",data=df).fit()
    print(f"  add C(col) over [{tag}]: p={anova_lm(m0,m1)['Pr(>F)'].iloc[1]:.2g}")
mp=df.groupby("col")["yc"].mean()
print("  pooled row-norm by column:", " ".join(f"c{c}:{10**mp[c]:.3f}" for c in range(1,6)))
fr=friedmanchisquare(*[df[df.col==c]["yc"].values for c in range(1,6)])
print(f"  Friedman(5 col): chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")

fig,ax=plt.subplots(1,2,figsize=(13,5))
e=670; T=tot(e)
for c in range(NCOL): ax[0].plot(range(1,NROW+1),T[:,c]/1e-12,marker="o",label=f"col{c+1}",color=plt.cm.tab10(c))
ax[0].axvspan(NROW-0.5,NROW+0.5,color="red",alpha=0.08); ax[0].text(NROW,ax[0].get_ylim()[1]*0.2,"excluded",color="red",fontsize=8,ha="center")
ax[0].set_xlabel("replicate row (acq order)"); ax[0].set_ylabel("intensity x1e-12")
ax[0].set_title(f"{TAG} ex{e}: acquisition drift (last row dropped)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
mpv=df.groupby("col")["yc"].mean(); sev=df.groupby("col")["yc"].sem()
ax[1].bar(range(1,6),[10**mpv[c] for c in range(1,6)],yerr=[np.log(10)*10**mpv[c]*sev[c] for c in range(1,6)],capsize=4,color="#2c7fb8")
ax[1].set_xticks(range(1,6)); ax[1].axhline(1,color="k",lw=0.6,ls=":")
ax[1].set_xlabel("plate column (concentration map UNKNOWN)"); ax[1].set_ylabel("row-normalized intensity (pooled)")
ax[1].set_title("Column effect (drift removed)")
fig.suptitle(f"{TAG}: serum column effect — positional (need conc map)",y=1.02)
fig.tight_layout(); out=f"{FOLDER}/conc_effect_{TAG}.png"; fig.savefig(out,dpi=130,bbox_inches="tight")
print("Saved",out)
