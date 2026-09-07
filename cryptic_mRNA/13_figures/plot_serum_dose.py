#!/usr/bin/env python3
"""
Serum dose-response figures, columns mapped to concentration and ordered by dose.
Layout: 8 rows (technical reps + acquisition drift) x 5 cols; LAST ROW EXCLUDED.
Column->conc map: [1000,10,0,1,100]; dose order shown 0,1,10,100,1000.
Per plate: (left) per-excitation row-normalized dose curves, (right) pooled dose-response
with SEM. Plus emission spectra by concentration (ex670).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr

BASE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2"
EXC=[570,640,670,750,780]
CMAP=[1000,10,0,1,100]                       # column index -> conc
DOSE=[0,1,10,100,1000]                        # display order
COL_OF={c:CMAP.index(c) for c in CMAP}        # conc -> column index
RANK={0:0,1:1,10:2,100:3,1000:4}

def load(folder,prefix,e):
    a=np.loadtxt(f"{BASE}/{folder}/{prefix}{e}.txt"); return a[:,0],a[:,2:]
wl,_=load("2H_SERUM","SERUM__",570)

def cube(folder,prefix,e):
    _,d=load(folder,prefix,e); return d.reshape(d.shape[0],8,5)   # em x 8row x 5col
def integ(c): return np.trapz(np.clip(c,0,None),wl,axis=0)        # 8 x 5

def make(folder,prefix,tag):
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    cmap=cm.viridis
    # left: per-excitation row-normalized dose curves (rows 0..6)
    allrel=[]
    for i,e in enumerate(EXC):
        T=integ(cube(folder,prefix,e))[:7]          # drop last row
        rel=T/T.mean(1,keepdims=True)               # row-normalize (drift out)
        allrel.append(rel)
        m=[rel[:,COL_OF[c]].mean() for c in DOSE]
        s=[rel[:,COL_OF[c]].std()/np.sqrt(7) for c in DOSE]
        ax[0].errorbar(range(5),m,yerr=s,marker="o",capsize=3,color=cmap(i/4),label=f"ex{e}")
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels([str(c) for c in DOSE])
    ax[0].axhline(1,color="k",lw=0.6,ls=":"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("STMN2-CE concentration"); ax[0].set_ylabel("row-normalized intensity")
    ax[0].set_title(f"{tag}: per-excitation dose curves (drift removed)")
    # right: pooled over excitations
    pooled=np.concatenate(allrel,0)                 # (35,5) per excitation stacked
    mp=[pooled[:,COL_OF[c]].mean() for c in DOSE]
    sp=[pooled[:,COL_OF[c]].std()/np.sqrt(pooled.shape[0]) for c in DOSE]
    ax[1].bar(range(5),mp,yerr=sp,capsize=4,color=["#444"]+["#2c7fb8"]*4)
    ax[1].plot(range(5),mp,color="#d62728",marker="o",lw=1.5)
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels([str(c) for c in DOSE])
    ax[1].axhline(1,color="k",lw=0.6,ls=":"); ax[1].set_ylim(0.93,1.06)
    ax[1].set_xlabel("STMN2-CE concentration"); ax[1].set_ylabel("row-normalized intensity (pooled)")
    # stats annotation
    rk=np.array([RANK[c] for c in DOSE for _ in range(pooled.shape[0])])
    yv=np.concatenate([pooled[:,COL_OF[c]] for c in DOSE])
    rho,p=spearmanr(rk,yv)
    ax[1].set_title(f"{tag}: pooled dose-response  (Spearman {rho:+.2f}, p={p:.1g})")
    fig.suptitle(f"{tag} serum: STMN2-CE dose-response (8th row excluded, acquisition drift removed)",y=1.02)
    fig.tight_layout(); out=f"{BASE}/{folder}/dose_response_{tag}.png"
    fig.savefig(out,dpi=130,bbox_inches="tight"); print("Saved",out)

    # emission spectra by concentration (ex670), mean over 7 rows
    fig2,a2=plt.subplots(1,1,figsize=(9,5)); e=670; C=cube(folder,prefix,e)[:, :7, :]
    for c in DOSE:
        mean=C[:,:,COL_OF[c]].mean(1)
        a2.plot(wl,mean,color=cmap(RANK[c]/4),lw=1.4,label=str(c))
    a2.set_xlim(900,1400); a2.set_xlabel("emission wavelength (nm)"); a2.set_ylabel("intensity (a.u.)")
    a2.grid(alpha=0.3); a2.legend(title="STMN2-CE",fontsize=8); a2.set_title(f"{tag} ex{e}: emission spectra by concentration")
    fig2.tight_layout(); out2=f"{BASE}/{folder}/spectra_by_conc_{tag}.png"
    fig2.savefig(out2,dpi=130,bbox_inches="tight"); print("Saved",out2)

make("2H_SERUM","SERUM__","2H")
make("3H_SERUM","3HSERUM__","3H")
