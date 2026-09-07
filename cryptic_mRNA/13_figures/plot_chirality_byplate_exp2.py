#!/usr/bin/env python3
"""
Per-chirality fitted dose-response, broken out by time point x matrix (4 plates).
Two figures:
  (1) brightness: row-normalized gauss_max vs dose, 7 clean chiralities, one panel/plate.
  (2) (7,5) Delta-lambda (paper readout): drift-removed fitted center vs dose, one panel/plate.
Drift removed by normalizing within (plate,row). Water plates show the pure-water column as 'H2O'.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2"
df=pd.read_csv(f"{BASE}/chirality_gaussian_descriptors_exp2.csv")
GOOD=["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4","ch7_6"]
PLATES=["2H_WATER","3H_WATER","2H_SERUM","3H_SERUM"]
TITLE={"2H_WATER":"2 h — water","3H_WATER":"3 h — water","2H_SERUM":"2 h — serum","3H_SERUM":"3 h — serum"}
# per-plate dose order (numeric only); water plates additionally have the pure-water control
DOSES={"2H_WATER":[0,0.1,1,10,100],"3H_WATER":[0,0.1,1,10,100],
       "2H_SERUM":[0,1,10,100,1000],"3H_SERUM":[0,1,10,100,1000]}
COLORS=plt.cm.tab10(np.linspace(0,1,len(GOOD)))

def rownorm(plate,col):
    d=df[df.plate==plate]
    return d[col]/d.groupby("row")[col].transform("mean")

# ---------- Figure 1: brightness ----------
fig,axes=plt.subplots(2,2,figsize=(13,9)); axes=axes.ravel()
for pi,plate in enumerate(PLATES):
    ax=axes[pi]; doses=DOSES[plate]; xpos={d:i for i,d in enumerate(doses)}
    d=df[df.plate==plate].copy()
    for ci,ch in enumerate(GOOD):
        rn=rownorm(plate,f"{ch}_gauss_max"); d=d.assign(rn=rn.values)
        sub=d[~d.is_water]
        g=sub.groupby("dose")["rn"].mean(); se=sub.groupby("dose")["rn"].sem()
        x=[xpos[v] for v in g.index]
        ax.errorbar(x,g.values,yerr=se.values,marker="o",ms=4,capsize=2,color=COLORS[ci],label=ch,lw=1.2)
    # pure-water control (water plates): mean over good chiralities' row-norm, plotted at far left
    if (d.is_water).any():
        wvals=[]
        for ch in GOOD:
            rn=rownorm(plate,f"{ch}_gauss_max"); dd=d.assign(rn=rn.values)
            wvals.append(dd[dd.is_water]["rn"].mean())
        ax.scatter([-1],[np.mean(wvals)],marker="x",s=70,color="k",zorder=5)
        ax.text(-1,np.mean(wvals),"  pure\n  H2O",fontsize=7,va="center")
    ax.axhline(1,color="k",lw=0.6,ls=":")
    ax.set_xticks(range(len(doses))); ax.set_xticklabels([str(v) for v in doses])
    ax.set_xlim(-1.6,len(doses)-0.5); ax.grid(alpha=0.3)
    ax.set_title(TITLE[plate]); ax.set_xlabel("STMN2-CE concentration"); ax.set_ylabel("row-norm gauss_max")
    if pi==0: ax.legend(fontsize=7,ncol=2)
fig.suptitle("exp2 per-chirality brightness dose-response — by time point x matrix (drift removed)",y=1.0)
fig.tight_layout(); fig.savefig(f"{BASE}/chirality_dose_byplate_exp2.png",dpi=130,bbox_inches="tight")
print("Saved chirality_dose_byplate_exp2.png")

# ---------- Figure 2: (7,5) Delta-lambda ----------
fig2,axes2=plt.subplots(2,2,figsize=(13,9)); axes2=axes2.ravel()
for pi,plate in enumerate(PLATES):
    ax=axes2[pi]; doses=DOSES[plate]; xpos={d:i for i,d in enumerate(doses)}
    d=df[df.plate==plate].copy()
    cc="ch7_5_gauss_em_center"
    d["ccdt"]=d[cc]-d.groupby("row")[cc].transform("mean")     # drift-removed center residual
    sub=d[~d.is_water & d["ch7_5_gauss_r2"].gt(0.95)]
    base=sub[sub.dose==0]["ccdt"].mean()
    g=sub.groupby("dose")["ccdt"].mean(); se=sub.groupby("dose")["ccdt"].sem()
    # Dlam = lam0 - lam (blue positive), referenced to dose 0
    dl=-(g-base); x=[xpos[v] for v in g.index]
    rho,p=spearmanr(sub["dose"].map({v:i for i,v in enumerate(doses)}),-(sub["ccdt"]-base))
    ax.errorbar(x,dl.values,yerr=se.values,marker="o",ms=5,capsize=3,color="#d62728",lw=1.5)
    ax.axhline(0,color="k",lw=0.6,ls=":")
    ax.axhspan(-1.62/2,1.62/2,color="gray",alpha=0.12)  # +/- half a sampling step
    ax.text(0.02,0.95,"gray band = ±½ pixel (1.62 nm/pt)",transform=ax.transAxes,fontsize=7,va="top")
    ax.set_xticks(range(len(doses))); ax.set_xticklabels([str(v) for v in doses])
    ax.set_ylim(-2.5,2.5); ax.grid(alpha=0.3)
    ax.set_title(f"{TITLE[plate]}  (7,5) Δλ   rho={rho:+.2f} p={p:.2g}")
    ax.set_xlabel("STMN2-CE concentration"); ax.set_ylabel("Δλ = λ0−λ  (nm, blue +)")
fig2.suptitle("exp2 (7,5) peak shift (paper readout) — no Δλ in any plate (well within ±1 pixel)",y=1.0)
fig2.tight_layout(); fig2.savefig(f"{BASE}/ch75_dlambda_byplate_exp2.png",dpi=130,bbox_inches="tight")
print("Saved ch75_dlambda_byplate_exp2.png")
