#!/usr/bin/env python3
"""
exp2 / 2H_WATER  --- BLIND STMN2-CE titration in neuronal-mRNA-enriched water.
File format: col0 = emission wavelength (852-1675 nm, 512 pts), col1 = reference ramp
(ignored), cols 2-49 = 48 wells exported COLUMN-MAJOR -> 8 plate-columns x 6 rows.
Plate-column = concentration: col1 = 0 STMN2-CE (mRNA-enriched water background),
cols 2-7 = increasing STMN2-CE (blind), col8 = pure water. 6 rows = replicates.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp2/STMN_2/STMN_2/2H_WATER"
EXC=[570,640,670,750,780]
NCOL,NROW=8,6
LAB=["C1\n(0 target)","C2","C3","C4","C5","C6","C7","C8\n(water)"]

def load(e):
    a=np.loadtxt(f"{HERE}/STMN_2_1H__{e}.txt")
    return a[:,0], a[:,2:]            # wl, 48 wells

wl,_=load(570)
# tensor: ex x conc(8) x rep(6) x emission
cube={}
for e in EXC:
    _,d=load(e)
    cube[e]=d.reshape(d.shape[0],NCOL,NROW)   # emission x 8conc x 6rep

# ---- total intensity per well ----
def integ(d): return np.trapz(np.clip(d,0,None),wl,axis=0)

print("="*70); print("TOTAL INTENSITY vs CONCENTRATION (blind, 8 plate-cols)"); print("="*70)
rows=[]
for e in EXC:
    d=cube[e]                                   # em x 8 x 6
    tot=integ(d)                                # 8 x 6
    m=tot.mean(1)
    # Spearman over the titration arm cols 1-7 (index 0..6), per-replicate
    x=np.repeat(np.arange(7),NROW); y=tot[:7].ravel()
    rho,p=spearmanr(x,y)
    fold=m[6]/m[0]
    print(f"ex{e}: means(x1e-12)= {np.array2string(m/1e-12,precision=1,floatmode='fixed')}")
    print(f"        fold C7/C1={fold:.2f}x  Spearman(dose,cols1-7) rho={rho:+.2f} p={p:.1e}"
          f"   water(C8)/C1={m[7]/m[0]:.2f}x")
    rows.append(m)

# ---- Figure 1: dose-response curves, all excitations ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
cmap=cm.viridis
for i,e in enumerate(EXC):
    d=cube[e]; tot=integ(d); m=tot.mean(1); sd=tot.std(1)
    ax[0].errorbar(range(1,9),m,yerr=sd,marker="o",capsize=3,color=cmap(i/4),label=f"ex{e}")
    ax[1].plot(range(1,9),m/m[0],marker="o",color=cmap(i/4),label=f"ex{e}")
for a in ax:
    a.axvspan(7.5,8.5,color="gray",alpha=0.12)
    a.set_xticks(range(1,9)); a.set_xticklabels(LAB,fontsize=7)
    a.grid(alpha=0.3); a.legend(fontsize=8)
ax[0].set_ylabel("integrated intensity (a.u.)"); ax[0].set_title("2H water: total intensity vs plate column")
ax[1].set_ylabel("fold vs C1 (0 target)"); ax[1].set_title("normalized (C1=1)")
ax[1].text(7.6,ax[1].get_ylim()[1]*0.9,"pure\nwater",fontsize=7,color="gray")
fig.suptitle("STMN2-CE titration in neuronal-mRNA water (blind) — sensor brightens with column",y=1.02)
fig.tight_layout(); fig.savefig(f"{HERE}/doseresponse_2H_water.png",dpi=130,bbox_inches="tight")

# ---- Figure 2: mean emission spectra per concentration (ex670, brightest) ----
fig2,ax2=plt.subplots(1,1,figsize=(9,5))
e=670; d=cube[e]
for c in range(NCOL):
    mean=d[:,c,:].mean(1)
    col="gray" if c==7 else cmap(c/6)
    ls="--" if c==7 else "-"
    ax2.plot(wl,mean,color=col,ls=ls,lw=1.4,label=LAB[c].replace("\n"," "))
ax2.set_xlim(900,1400); ax2.set_xlabel("emission wavelength (nm)")
ax2.set_ylabel("intensity (a.u.)"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8,title="plate col")
ax2.set_title(f"Emission spectra by concentration (ex{e} nm) — whole envelope rises with STMN2-CE")
fig2.tight_layout(); fig2.savefig(f"{HERE}/spectra_by_conc_2H_water.png",dpi=130,bbox_inches="tight")
print("\nSaved doseresponse_2H_water.png, spectra_by_conc_2H_water.png")
