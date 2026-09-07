#!/usr/bin/env python3
"""
Generic 'water' plate analyzer (corrected layout): 6 conc-COLUMNS x 8 replicate ROWS,
ROW-MAJOR export. Columns (randomized) -> conc: [0,100,0.1,10,1,water] at read pos 0..5.
Removes the between-row acquisition drift (block + flexible detrend) and tests the
within-row concentration effect + read-order direction. Usage: pass FOLDER and PREFIX.
"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, friedmanchisquare
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

FOLDER=sys.argv[1]; PREFIX=sys.argv[2]; TAG=sys.argv[3]
EXC=[570,640,670,750,780]
CLAB=["0","100","0.1","10","1","water"]; ORDER=[0,2,4,3,1,5]  # display 0,0.1,1,10,100,water

def load(e):
    a=np.loadtxt(f"{FOLDER}/{PREFIX}{e}.txt"); return a[:,0],a[:,2:]
wl,_=load(570)
def tot(e):
    _,d=load(e); d=d.reshape(d.shape[0],8,6)
    return np.trapz(np.clip(d,0,None),wl,axis=0)        # 8row x 6col

# long table pooled over excitations
recs=[]
for e in EXC:
    T=tot(e)
    for r in range(8):
        for c in range(6):
            recs.append(dict(ex=e,row=r,pos=c,clab=CLAB[c],acq=r*6+c,y=np.log10(T[r,c])))
df=pd.DataFrame(recs)
df["yc"]=df["y"]-df.groupby(["ex","row"])["y"].transform("mean")   # row-normalized residual

print("="*72); print(f"{TAG}: within-row concentration effect (row drift removed)"); print("="*72)
# per-excitation
for e in EXC:
    s=df[df.ex==e]; mp=s.groupby("clab")["yc"].mean()
    m_full=smf.ols("y~C(row)+C(clab)",data=s).fit(); m_row=smf.ols("y~C(row)",data=s).fit()
    fp=anova_lm(m_row,m_full)["Pr(>F)"].iloc[1]
    dd=s[s.clab!="water"].copy(); dd["lr"]=dd["clab"].map({"0":0,"0.1":1,"1":2,"10":3,"100":4})
    rho,pp=spearmanr(dd["lr"],dd["yc"])
    rel=10**mp  # approx fold vs row-mean
    line=" ".join(f"{CLAB[i]}:{rel[CLAB[i]]:.3f}" for i in ORDER)
    print(f"ex{e}: {line} | conc F-test p={fp:.2g}  dose-rho={rho:+.2f} p={pp:.2g}")

print("\n--- POOLED over 5 excitations ---")
for tag,base in [("row block","C(ex)+C(row)"),("cubic drift","C(ex)+acq+I(acq**2)+I(acq**3)"),
                 ("quintic drift","C(ex)+acq+I(acq**2)+I(acq**3)+I(acq**4)+I(acq**5)")]:
    m0=smf.ols(f"y~{base}",data=df).fit(); m1=smf.ols(f"y~{base}+C(clab)",data=df).fit()
    print(f"  add C(conc) over [{tag}]: p={anova_lm(m0,m1)['Pr(>F)'].iloc[1]:.2g}")
mp=df.groupby("clab")["yc"].mean();
print("  pooled row-norm fold (vs row mean):", " ".join(f"{CLAB[i]}:{10**mp[CLAB[i]]:.3f}" for i in ORDER))
rho_read,p_read=spearmanr(df["pos"],df["yc"])
dd=df[df.clab!="water"].copy(); dd["lr"]=dd["clab"].map({"0":0,"0.1":1,"1":2,"10":3,"100":4})
rho_d,p_d=spearmanr(dd["lr"],dd["yc"])
print(f"  read-order: water resid={mp['water']:+.4f} (read last); Spearman(readpos,resid)={rho_read:+.2f} p={p_read:.2g}")
print(f"  Spearman(true dose 0..100, resid)={rho_d:+.2f} p={p_d:.2g}")
fr=friedmanchisquare(*[df[(df.clab==CLAB[c])]["yc"].values for c in range(6)])
print(f"  Friedman(6 conc): chi2={fr.statistic:.1f} p={fr.pvalue:.2g}")

# figure: pooled bar + drift
fig,ax=plt.subplots(1,2,figsize=(13,5))
e=670; T=tot(e)
for c in range(6): ax[0].plot(range(1,9),T[:,c]/1e-12,marker="o",label=CLAB[c],color=plt.cm.tab10(c))
ax[0].set_xlabel("replicate row (acquisition order)"); ax[0].set_ylabel("intensity x1e-12")
ax[0].set_title(f"{TAG} ex{e}: acquisition drift"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8,title="conc")
mpv=df.groupby("clab")["yc"].mean(); sev=df.groupby("clab")["yc"].sem()
ax[1].bar(range(6),[10**mpv[CLAB[i]] for i in ORDER],
          yerr=[np.log(10)*10**mpv[CLAB[i]]*sev[CLAB[i]] for i in ORDER],capsize=4,
          color=["#444"]+["#2c7fb8"]*4+["#888"])
ax[1].set_xticks(range(6)); ax[1].set_xticklabels([CLAB[i] for i in ORDER])
ax[1].axhline(1,color="k",lw=0.6,ls=":"); ax[1].set_ylabel("row-normalized intensity (pooled)")
ax[1].set_xlabel("STMN2-CE concentration"); ax[1].set_title("Concentration effect (drift removed)")
fig.suptitle(f"{TAG}: concentration effect within rows (artifact-corrected)",y=1.02)
fig.tight_layout(); out=f"{FOLDER}/conc_effect_{TAG}.png"; fig.savefig(out,dpi=130,bbox_inches="tight")
print("Saved",out)
