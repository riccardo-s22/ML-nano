#!/usr/bin/env python3
"""
DECISIVE dose-vs-edge test across all 4 exp2 plates.
Each plate: 8 rows (technical reps + acquisition drift) x N columns (randomized conc).
Row-normalize within (excitation,row) to remove drift -> residual yc (log10).

Key idea: water plates and serum plates use DIFFERENT conc->column mappings, so pooling
breaks the confound between target concentration and plate-column EDGE position.

Plates (folder, prefix, ncol, drop_last_row, conc map by column index):
  2H_WATER STMN_2_1H__  6  no   [0,100,0.1,10,1,'water']
  3H_WATER 3H_WATER__   6  no   [0,100,0.1,10,1,'water']
  2H_SERUM SERUM__      5  yes  [1000,10,0,1,100]
  3H_SERUM 3HSERUM__    5  yes  [1000,10,0,1,100]
'water' = pure-water matrix control (no mRNA, not a target conc) -> excluded from dose analysis.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2"
EXC=[570,640,670,750,780]
RANK={0:0,0.1:1,1:2,10:3,100:4,1000:5}
PLATES=[("2H_WATER","STMN_2_1H__",6,False,[0,100,0.1,10,1,"water"]),
        ("3H_WATER","3H_WATER__", 6,False,[0,100,0.1,10,1,"water"]),
        ("2H_SERUM","SERUM__",    5,True ,[1000,10,0,1,100]),
        ("3H_SERUM","3HSERUM__",  5,True ,[1000,10,0,1,100])]

def integ(folder,prefix,e,ncol):
    a=np.loadtxt(f"{BASE}/{folder}/{prefix}{e}.txt"); wl=a[:,0]; d=a[:,2:]
    return np.trapz(np.clip(d,0,None),wl,axis=0).reshape(8,ncol),

recs=[]
for folder,prefix,ncol,drop,cmap in PLATES:
    for e in EXC:
        T=integ(folder,prefix,e,ncol)[0]
        nrow=7 if drop else 8
        for r in range(nrow):
            for c in range(ncol):
                lab=cmap[c]
                edge=min(c,ncol-1-c)
                recs.append(dict(plate=folder,matrix=folder[3:],ex=e,row=r,col=c,
                                 conc=lab,edge=edge,y=np.log10(T[r,c])))
df=pd.DataFrame(recs)
df["yc"]=df["y"]-df.groupby(["plate","ex","row"])["y"].transform("mean")

# ---- serum re-summary by dose ----
print("="*72); print("SERUM by DOSE (mapped, drift-removed, last row excluded)"); print("="*72)
for folder in ["2H_SERUM","3H_SERUM"]:
    s=df[df.plate==folder]
    mp=s.groupby("conc")["yc"].mean()
    order=[0,1,10,100,1000]
    print(f"{folder}: "+" ".join(f"{c}:{10**mp[c]:.3f}" for c in order))
    s2=s.copy(); s2["rank"]=s2["conc"].map(RANK)
    rho,p=spearmanr(s2["rank"],s2["yc"])
    print(f"   dose Spearman = {rho:+.2f} p={p:.2g}")

# ---- DECISIVE pooled dose-vs-edge (exclude pure-water) ----
print("\n"+"="*72); print("POOLED dose-vs-EDGE (all 4 plates, pure-water excluded)"); print("="*72)
d=df[df.conc!="water"].copy(); d["rank"]=d["conc"].map(RANK); d["lconc"]=np.log10(d["conc"].astype(float).clip(lower=0.05))
rho_d,p_d=spearmanr(d["rank"],d["yc"]); rho_e,p_e=spearmanr(d["edge"],d["yc"])
print(f"  Spearman(dose rank, yc) = {rho_d:+.2f} p={p_d:.2g}")
print(f"  Spearman(edge dist, yc) = {rho_e:+.2f} p={p_e:.2g}  (edge=0 outer .. 2 center)")
print("\n  edge direction SEPARATELY (note: should agree if it's a fixed position artifact):")
for m in ["WATER","SERUM"]:
    sub=d[d.matrix==m]; re,pe=spearmanr(sub["edge"],sub["yc"])
    print(f"    {m}: Spearman(edge,yc)={re:+.2f} p={pe:.2g}")
print("\n  OLS  yc ~ rank + edge + C(plate):")
m=smf.ols("yc ~ rank + edge + C(plate)",data=d).fit()
for term in ["rank","edge"]:
    print(f"    {term}: beta={m.params[term]:+.4f}  p={m.pvalues[term]:.2g}")
print("\n  OLS  yc ~ rank + C(plate)  (edge dropped):")
m2=smf.ols("yc ~ rank + C(plate)",data=d).fit()
print(f"    rank: beta={m2.params['rank']:+.4f}  p={m2.pvalues['rank']:.2g}")

# ---- figure: all 4 plates by dose ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
col={"2H_WATER":"#1f77b4","3H_WATER":"#2ca02c","2H_SERUM":"#d62728","3H_SERUM":"#ff7f0e"}
for folder,prefix,ncol,drop,cmap in PLATES:
    s=df[df.plate==folder]; concs=[c for c in [0,0.1,1,10,100,1000] if c in set(s["conc"])]
    mp=s.groupby("conc")["yc"].mean(); se=s.groupby("conc")["yc"].sem()
    x=[RANK[c] for c in concs]
    ax[0].errorbar(x,[10**mp[c] for c in concs],yerr=[np.log(10)*10**mp[c]*se[c] for c in concs],
                   marker="o",capsize=3,color=col[folder],label=folder)
    if "water" in set(s["conc"]):
        ax[0].scatter([-1],[10**s[s.conc=='water']['yc'].mean()],marker="x",color=col[folder],s=60)
ax[0].set_xticks([-1,0,1,2,3,4,5]); ax[0].set_xticklabels(["pureH2O","0","0.1","1","10","100","1000"])
ax[0].axhline(1,color="k",lw=0.6,ls=":"); ax[0].set_xlabel("STMN2-CE concentration (x = pure-water control)")
ax[0].set_ylabel("row-normalized intensity"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[0].set_title("Dose-response, all 4 plates (drift removed)")
# edge panel: yc vs edge dist, water vs serum
for m,cc in [("WATER","#1f77b4"),("SERUM","#d62728")]:
    sub=d[d.matrix==m]; g=sub.groupby("edge")["yc"].mean(); gs=sub.groupby("edge")["yc"].sem()
    ax[1].errorbar(g.index,[10**v for v in g.values],yerr=[np.log(10)*10**v*e for v,e in zip(g.values,gs.values)],
                   marker="s",capsize=3,label=m,color=cc)
ax[1].axhline(1,color="k",lw=0.6,ls=":"); ax[1].set_xlabel("edge distance (0=outer column .. 2=center)")
ax[1].set_ylabel("row-normalized intensity"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_title("Edge effect: OPPOSITE in water vs serum -> not a fixed position artifact")
fig.suptitle("exp2: dose response is consistent across plates; edge is not",y=1.02)
fig.tight_layout(); fig.savefig(f"{BASE}/dose_vs_edge_allplates.png",dpi=130,bbox_inches="tight")
print("\nSaved dose_vs_edge_allplates.png")
