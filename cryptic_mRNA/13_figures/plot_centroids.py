#!/usr/bin/env python3
"""Fit-free centroid (Delta-lambda) dose curves, GT15-STMN2 vs control, water.
Each tube referenced to its own 0-conc centroid so shifts are directly comparable.
A real hybridization signal = coherent DOWNWARD (blue) shift; we expect incoherent."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
ff = pd.read_csv(f"{HERE}/fit_free_descriptors.csv")
COPIES=[0,10,100,1000,10000,100000]; XLAB=["0","10","1e2","1e3","1e4","1e5"]
TUBES=["ch6_5","ch7_5","ch8_3","ch7_6"]; COL=dict(zip(TUBES,["C0","C1","C2","C3"]))

def curve(sensor, tube):
    g=ff[(ff.sensor==sensor)&(ff.matrix=="water")]
    s=g.groupby("copies")[f"{tube}_ff_centroid"].agg(["mean","sem"]).reindex(COPIES)
    m=s["mean"].to_numpy(); return m-m[0], s["sem"].to_numpy()  # delta vs 0-conc

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for k,(sensor,title) in enumerate([("GT15-STMN2","GT15-STMN2 (sensor)"),
                                   ("GT15","GT15 (control)")]):
    for t in TUBES:
        d,e=curve(sensor,t)
        ax[k].errorbar(range(6), d, yerr=e, marker="o", color=COL[t], capsize=3, label=t)
    ax[k].axhline(0, color="gray", lw=0.8, ls="--")
    ax[k].axhspan(-1,1, color="gray", alpha=0.12)  # +/-1nm = sampling/noise band
    ax[k].set_title(title); ax[k].set_xticks(range(6)); ax[k].set_xticklabels(XLAB)
    ax[k].set_xlabel("STMN2-CE copies/uL")
    if k==0: ax[k].set_ylabel("centroid shift vs 0-conc (nm)\n(down = blue-shift = signal)")
    ax[k].grid(alpha=0.3); ax[k].legend(fontsize=8)
fig.suptitle("Fit-free centroid shift — incoherent, <2nm: no coherent blue-shift", y=1.02)
fig.tight_layout()
fig.savefig(f"{HERE}/centroids_dose.png", dpi=130, bbox_inches="tight")
print("Saved centroids_dose.png")
print("\nSensor centroid shift vs 0 (nm):")
for t in TUBES:
    d,_=curve("GT15-STMN2",t)
    print(f"  {t:7} " + " ".join(f"{v:+5.2f}" for v in d))
