#!/usr/bin/env python3
"""
Global emission spectrum (intensity vs emission wavelength) vs STMN2-CE concentration.
The 12 chiralities are read at 5 excitations, so we build a resonance-stitched
COMPOSITE: at each emission wavelength, take the value from the excitation file
assigned to the nearest chirality (its resonant excitation). Average replicate
wells per concentration; color by dose. GT15-STMN2 sensor vs GT15 control (water).
Also a per-excitation small-multiples view (rawest).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
EXC=[570,640,670,750,780]
CH={"ch8_3":(973.98,673.94),"ch6_5":(987.82,577.12),"ch7_5":(1047.81,653.32),
    "ch10_2":(1080.60,745.92),"ch9_4":(1131.96,731.39),"ch8_4":(1130.34,599.78),
    "ch7_6":(1138.19,659.79),"ch8_6":(1200.03,727.40),"ch8_7":(1288.27,740.87),
    "ch9_5":(1262.98,685.15),"ch10_3":(1267.70,648.97),"ch10_5":(1282.97,801.23)}
EXMAP={c:min(EXC,key=lambda e:abs(e-CH[c][1])) for c in CH}
COPIES=[0,10,100,1000,10000,100000]; LAB=["0","10","1e2","1e3","1e4","1e5"]

em=pd.read_csv(f"{HERE}/emission.txt")["Emission"].to_numpy(float)
spectra={e:pd.read_csv(f"{HERE}/{e}.txt",sep="\t") for e in EXC}
wells=list(spectra[570].columns)

# emission point -> resonant excitation (via nearest chirality center)
centers=np.array([CH[c][0] for c in CH]); chnames=list(CH)
pt_ex=np.array([EXMAP[chnames[np.argmin(np.abs(centers-w))]] for w in em])

def composite(well):
    out=np.empty_like(em)
    for e in EXC:
        msk=pt_ex==e
        out[msk]=spectra[e][well].to_numpy(float)[msk]
    return out

def wells_for(sensor):  # water arm
    rows="ABCDEF" if sensor=="GT15-STMN2" else "GH"
    return {col:[f"{r}{col}" for r in rows if f"{r}{col}" in wells] for col in range(1,7)}

cmap=cm.viridis
# ---- Figure 1: composite spectra, sensor vs control ----
fig,ax=plt.subplots(1,2,figsize=(13,5),sharex=True)
for k,sensor in enumerate(["GT15-STMN2","GT15"]):
    wf=wells_for(sensor)
    for ci,col in enumerate(range(1,7)):
        ws=wf[col]
        if not ws: continue
        mean=np.mean([composite(w) for w in ws],axis=0)
        ax[k].plot(em,mean,color=cmap(ci/5),lw=1.2,label=f"{LAB[ci]} cp/uL")
    ax[k].set_title(f"{sensor}  (water)  — composite emission")
    ax[k].set_xlabel("emission wavelength (nm)"); ax[k].set_ylabel("intensity (a.u.)")
    ax[k].grid(alpha=0.3); ax[k].legend(title="STMN2-CE",fontsize=8)
    for c in CH:
        ax[k].axvline(CH[c][0],color="gray",lw=0.4,ls=":",alpha=0.5)
fig.suptitle("Global emission spectrum brightens with STMN2-CE (sensor) — composite over 5 excitations",y=1.02)
fig.tight_layout(); fig.savefig(f"{HERE}/global_spectra_composite.png",dpi=130,bbox_inches="tight")

# ---- Figure 2: per-excitation small multiples (sensor only, rawest view) ----
fig2,ax2=plt.subplots(2,3,figsize=(14,7))
wf=wells_for("GT15-STMN2"); ax2=ax2.ravel()
for ei,e in enumerate(EXC):
    for ci,col in enumerate(range(1,7)):
        ws=wf[col]
        mean=np.mean([spectra[e][w].to_numpy(float) for w in ws],axis=0)
        ax2[ei].plot(em,mean,color=cmap(ci/5),lw=1.0,label=LAB[ci])
    owners=[c for c in CH if EXMAP[c]==e]
    ax2[ei].set_title(f"ex {e} nm  (resonant: {', '.join(owners)})",fontsize=9)
    ax2[ei].set_xlabel("emission nm"); ax2[ei].set_ylabel("intensity")
    ax2[ei].grid(alpha=0.3); ax2[ei].legend(fontsize=6,title="cp/uL")
ax2[5].axis("off")
fig2.suptitle("GT15-STMN2 (water): emission spectrum per excitation, by STMN2-CE concentration",y=1.01)
fig2.tight_layout(); fig2.savefig(f"{HERE}/global_spectra_byexcitation.png",dpi=130,bbox_inches="tight")
print("Saved global_spectra_composite.png, global_spectra_byexcitation.png")
