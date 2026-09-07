#!/usr/bin/env python3
"""Test the NEIGHBOR / cross-talk hypothesis in exp4 serum:
does a column read brighter when its physically adjacent columns are high-concentration?
Layout (0-idx col -> copies): 0=1e6,1=0.1,2=1e4,3=1,4=1e3,5=10,6=0,7=20,8=50,9=100,10=500,11=1e5.
Restrict to BELOW-THRESHOLD columns (<=500 cp) where the sensor's OWN response is flat, so any
brightness variation there is not own-dose. Then correlate with (a) neighbor concentration and
(b) column position, which are CONFOUNDED (all highs are on the left)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp4/serum"; PREFIX="STMN_22_Serum__"
EXC=[570,640,670,750,780]; NCOL=12; MISS=90
COPIES={0:1e6,1:0.1,2:1e4,3:1.,4:1e3,5:10.,6:0.,7:20.,8:50.,9:100.,10:500.,11:1e5}
LOGC={c:np.log10(v+1) for c,v in COPIES.items()}   # log10(copies+1); blank->0

em=None; inten={}
for e in EXC:
    a=np.loadtxt(f"{BASE}/{PREFIX}{e}.txt")
    if em is None: em=a[:,0]; pk=(em>=900)&(em<=1350); bw=(em>=1400)&(em<=1600)
    W=a[:,2:]; inten[e]=(W[pk,:]-W[bw,:].mean(axis=0)).sum(axis=0)
int_all=sum(inten[e] for e in EXC)
rows=[]
for w in range(len(int_all)):
    p=w if w<MISS else w+1; r=p//NCOL; c=p%NCOL
    rows.append(dict(w=w,row=r,col=c,copies=COPIES[c],int_all=int_all[w]))
df=pd.DataFrame(rows)
df=df[df.int_all>0].reset_index(drop=True)
df["int_rn"]=df["int_all"]/df.groupby("row")["int_all"].transform("mean")

# neighbor CONCENTRATION score = mean log10(copies+1) of adjacent columns (edge = single neighbor)
def nbr_score(c):
    ns=[cc for cc in (c-1,c+1) if 0<=cc<NCOL]
    return np.mean([LOGC[cc] for cc in ns])
df["nbr_conc"]=df["col"].map(nbr_score)

# ---- (1) column-level, below-threshold only ----
low=df[df.copies<=500]
cm=low.groupby("col").agg(colmean=("int_rn","mean"),copies=("copies","first"),
                          nbr=("nbr_conc","first")).reset_index()
cm["own_log"]=cm["copies"].map(lambda v:np.log10(v+1))
print("="*72); print("BELOW-THRESHOLD columns (<=500 cp): own signal ~flat -> test neighbor effect"); print("="*72)
print(f"{'col':>3} {'copies':>7} {'colmean_rn':>10} {'nbr_concScore':>13} {'ownlog':>6}")
for _,r in cm.sort_values("col").iterrows():
    print(f"{int(r.col):>3} {r.copies:>7g} {r.colmean:>10.3f} {r.nbr:>13.2f} {r.own_log:>6.2f}")
rho_n,p_n=spearmanr(cm["nbr"],cm["colmean"]); rho_x,p_x=spearmanr(cm["col"],cm["colmean"])
rho_o,p_o=spearmanr(cm["own_log"],cm["colmean"])
print(f"\nSpearman colmean vs NEIGHBOR conc score : rho={rho_n:+.2f} p={p_n:.2g}   (n={len(cm)} cols)")
print(f"Spearman colmean vs COLUMN position      : rho={rho_x:+.2f} p={p_x:.2g}   <- confounded w/ neighbor")
print(f"Spearman colmean vs OWN log-copies       : rho={rho_o:+.2f} p={p_o:.2g}   (should be ~0 if flat)")

# ---- (2) well-level OLS on below-threshold wells: neighbor vs position ----
print("\n"+"-"*72); print("well-level OLS (below-threshold wells), neighbor vs position:"); print("-"*72)
for f in ["int_rn ~ nbr_conc","int_rn ~ col","int_rn ~ nbr_conc + col"]:
    m=smf.ols(f,data=low).fit()
    terms=" ".join(f"{k}={m.params[k]:+.4f}(p={m.pvalues[k]:.2g})" for k in m.params.index if k!="Intercept")
    print(f"  {f:28s} R2={m.rsquared:.3f}  {terms}")

# ---- (3) physical same-row adjacency optical cross-talk (uses MEASURED neighbor brightness) ----
print("\n"+"-"*72); print("same-row adjacency: does a well track its LEFT/RIGHT same-row neighbor's int_rn?"); print("-"*72)
rn={(int(r.row),int(r.col)):r.int_rn for _,r in df.iterrows()}
def samerow_nbr(r,c):
    vs=[rn[(r,cc)] for cc in (c-1,c+1) if (r,cc) in rn]
    return np.mean(vs) if vs else np.nan
df["nbr_meas"]=[samerow_nbr(int(r.row),int(r.col)) for _,r in df.iterrows()]
sub=df.dropna(subset=["nbr_meas"])
m=smf.ols("int_rn ~ C(col) + nbr_meas",data=sub).fit()
print(f"  int_rn ~ C(col) + samerow_neighbor_measured : neighbor coef={m.params['nbr_meas']:+.3f} "
      f"p={m.pvalues['nbr_meas']:.2g}  (within-column crosstalk test)")

# ---- figure ----
fig,ax=plt.subplots(figsize=(9,6))
sc=ax.scatter(cm["nbr"],cm["colmean"],c=cm["col"],cmap="coolwarm",s=90,zorder=5)
for _,r in cm.iterrows():
    ax.annotate(f"c{int(r.col)}={r.copies:g}",(r.nbr,r.colmean),fontsize=8,
                xytext=(4,4),textcoords="offset points")
ax.axhline(1,color="k",lw=0.5,ls=":")
ax.set_xlabel("neighbor concentration score  [mean log10(copies+1) of adjacent columns]")
ax.set_ylabel("column-mean row-normalized ∫I")
ax.set_title(f"Below-threshold columns: brightness vs NEIGHBOR concentration\nSpearman rho={rho_n:+.2f} (p={p_n:.2g}); position rho={rho_x:+.2f}  — the two are confounded")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{BASE}/exp4_serum_neighbor_test.png",dpi=130,bbox_inches="tight")
print("\nSaved exp4_serum_neighbor_test.png")
