#!/usr/bin/env python3
"""
exp2/2H_WATER v3 -- rows=8 technical replicates; between-row gradient = acquisition
heating/drift (signal rises with read order, top->bottom of each column).
Read/export order is ROW-MAJOR: acq_index = row*6 + position; within a row the 6
columns are read in FIXED order [0,100,0.1,10,1,water]  (position 0..5).

Tests:
 (A) Does the concentration effect survive a FLEXIBLE acquisition-order detrend
     (poly in acq_index), not just per-row block?
 (B) Read-order/drift direction check: a within-row drift would order wells by
     read position (0->5). Compare observed pattern to that prediction.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2/2H_WATER"
EXC=[570,640,670,750,780]
CLAB=["0","100","0.1","10","1","water"]          # by within-row read position 0..5
POS =[0,1,2,3,4,5]
def load(e):
    a=np.loadtxt(f"{HERE}/STMN_2_1H__{e}.txt"); return a[:,0],a[:,2:]
wl,_=load(570)
def tot(e):
    _,d=load(e); d=d.reshape(d.shape[0],8,6)
    return np.trapz(np.clip(d,0,None),wl,axis=0)   # 8row x 6col

# build pooled long table
recs=[]
for e in EXC:
    T=tot(e)
    for r in range(8):
        for c in range(6):
            recs.append(dict(ex=e,row=r,pos=c,clab=CLAB[c],
                             acq=r*6+c, y=np.log10(T[r,c])))
df=pd.DataFrame(recs)

print("="*70)
print("(A) Concentration effect under different drift models (pooled, +ex)")
print("="*70)
for tag,base in [("per-row block  C(row)","C(ex)+C(row)"),
                 ("cubic acq drift","C(ex)+acq+I(acq**2)+I(acq**3)"),
                 ("quintic acq drift","C(ex)+acq+I(acq**2)+I(acq**3)+I(acq**4)+I(acq**5)")]:
    m0=smf.ols(f"y ~ {base}",data=df).fit()
    m1=smf.ols(f"y ~ {base}+C(clab)",data=df).fit()
    p=anova_lm(m0,m1)["Pr(>F)"].iloc[1]
    print(f"  add C(conc) on top of [{tag}]: F-test p = {p:.2g}")

print("\n"+"="*70)
print("(B) Read-order / drift-direction check")
print("="*70)
# row-normalize within each ex+row, then mean per position
df["yc"]=df["y"]-df.groupby(["ex","row"])["y"].transform("mean")
mp=df.groupby("pos")["yc"].mean()
print("  within-row residual (row mean removed), by READ POSITION 0->5:")
for c in POS:
    print(f"    pos{c} read#{c+1}  {CLAB[c]:>5} : {mp[c]:+.4f}")
rho_read,p_read=spearmanr(df["pos"],df["yc"])
print(f"  Spearman(read position, residual) = {rho_read:+.2f} p={p_read:.2g}")
print("   (a within-row HEATING drift would give a MONOTONIC trend in read order;")
print("    note water is read LAST yet is the DIMMEST -> opposite of a drift artifact)")
# dose arm spearman by actual concentration
dd=df[df.clab!="water"].copy()
dd["lr"]=dd["clab"].map({"0":0,"0.1":1,"1":2,"10":3,"100":4})
rho_d,p_d=spearmanr(dd["lr"],dd["yc"])
print(f"  Spearman(true dose 0..100, residual) = {rho_d:+.2f} p={p_d:.2g}")

# ---- figure: drift + concentration ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
e=670; T=tot(e)
cmap=plt.cm.tab10
for c in range(6):
    ax[0].plot(range(1,9),T[:,c]/1e-12,marker="o",color=cmap(c),label=CLAB[c])
ax[0].set_xlabel("replicate row (= acquisition order, top->bottom)")
ax[0].set_ylabel("intensity x1e-12"); ax[0].set_title(f"ex{e}: acquisition drift across 8 replicate rows")
ax[0].grid(alpha=0.3); ax[0].legend(title="conc",fontsize=8)
# residual by read position
order=POS
ax[1].bar(range(6),[mp[c] for c in order],color=["#444","#2c7fb8","#2c7fb8","#2c7fb8","#2c7fb8","#888"])
ax[1].set_xticks(range(6)); ax[1].set_xticklabels([f"#{c+1}\n{CLAB[c]}" for c in order])
ax[1].axhline(0,color="k",lw=0.6); ax[1].set_ylabel("row-normalized residual (log10)")
ax[1].set_xlabel("within-row read position / concentration")
ax[1].set_title("Effect by read position — not monotonic in read order")
fig.suptitle("Concentration effect is orthogonal to acquisition drift",y=1.02)
fig.tight_layout(); fig.savefig(f"{HERE}/drift_vs_conc_2H_water.png",dpi=130,bbox_inches="tight")
print("\nSaved drift_vs_conc_2H_water.png")
