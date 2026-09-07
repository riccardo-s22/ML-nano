#!/usr/bin/env python3
"""Intensity-INDEPENDENT (ratiometric / wavelength) readouts for exp5/22, which cancel the
multiplicative row-heating (+0.99) and left-right (-0.83) gradients that dominate total intensity.
Families scanned:
  (A) pairwise chirality peak-intensity RATIOS  gauss_max[i]/gauss_max[j]   (spectral weight redistribution)
  (B) per-chirality PEAK WAVELENGTH  gauss_em_center                        (red/blue shift, like exp2 dlambda)
  (C) per-chirality FWHM                                                     (broadening)
Dose test = cross-half agreement: each concentration is at DIFFERENT columns in top vs bottom, so a real
dose effect must show the SAME-SIGN Spearman(log copies, readout) in BOTH halves independently.
1e5 excluded (its top wells are the wrong 20nt sensor); wrong-sensor well excluded."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from itertools import combinations
from statsmodels.stats.multitest import multipletests

GOOD=['ch8_3','ch10_2','ch9_4','ch8_4','ch7_6','ch8_6','ch8_7','ch9_5','ch10_3','ch10_5']
df=pd.read_csv("chirality_gaussian_descriptors_exp5_22.csv")
df=df[(df.label!='wrong')&(df.copies!=1e5)].copy()
df["lc"]=np.log10(df["copies"]+1)

def crosshalf(col):
    """same-sign per-half Spearman(lc, readout). Returns (rho_top,p_top,rho_bot,p_bot,combined_ok)."""
    top=df[df.half=='top']; bot=df[df.half=='bot']
    rt,pt=spearmanr(top["lc"],top[col]); rb,pb=spearmanr(bot["lc"],bot[col])
    ok = (np.sign(rt)==np.sign(rb)) and pt<0.05 and pb<0.05
    return rt,pt,rb,pb,ok

readouts={}
# (A) pairwise intensity ratios (log ratio, symmetric)
for a,b in combinations(GOOD,2):
    df[f"R_{a}/{b}"]=np.log(df[f"{a}_gauss_max"]/df[f"{b}_gauss_max"]); readouts[f"R_{a}/{b}"]="ratio"
# (B) peak wavelengths
for ch in GOOD: readouts[f"{ch}_gauss_em_center"]="wavelength"
# (C) fwhm
for ch in GOOD: readouts[f"{ch}_gauss_fwhm"]="fwhm"

res=[]
for col,fam in readouts.items():
    rt,pt,rb,pb,ok=crosshalf(col)
    # also pooled Spearman controlling nothing (ratios are gradient-free)
    rp,pp=spearmanr(df["lc"],df[col])
    res.append(dict(readout=col,family=fam,rho_top=rt,p_top=pt,rho_bot=rb,p_bot=pb,
                    both_signif_sameSign=ok,rho_pool=rp,p_pool=pp,minp=max(pt,pb)))
R=pd.DataFrame(res)
# FDR across pooled p-values
R["fdr_pool"]=multipletests(R["p_pool"],method="fdr_bh")[1]

print(f"Scanned {len(R)} intensity-independent readouts "
      f"({(R.family=='ratio').sum()} ratios, {(R.family=='wavelength').sum()} wavelengths, {(R.family=='fwhm').sum()} fwhm)")
print("\n### STRINGENT dose filter: significant SAME-SIGN in BOTH halves independently ###")
hit=R[R.both_signif_sameSign].sort_values("minp")
if len(hit)==0:
    print("  -> NONE. No ratiometric/wavelength readout shows a consistent dose response in both halves.")
else:
    print(hit[["readout","family","rho_top","p_top","rho_bot","p_bot","rho_pool","fdr_pool"]].to_string(index=False))

print("\n### Best pooled correlations (any family), for reference ###")
print(R.reindex(R.p_pool.abs().sort_values().index)[["readout","family","rho_pool","p_pool","fdr_pool","both_signif_sameSign"]].head(12).to_string(index=False))

print("\n### Any pooled hit surviving FDR<0.05? ###")
fdrhit=R[R.fdr_pool<0.05]
print(fdrhit[["readout","family","rho_pool","p_pool","fdr_pool","both_signif_sameSign"]].to_string(index=False) if len(fdrhit) else "  -> NONE survive FDR correction.")

# Physically-motivated single readout: global spectral centroid (intensity-weighted mean peak wavelength)
w=[f"{ch}_gauss_max" for ch in GOOD]; c=[f"{ch}_gauss_em_center" for ch in GOOD]
df["spec_centroid"]=(df[w].values*df[c].values).sum(1)/df[w].values.sum(1)
rt,pt,rb,pb,ok=crosshalf("spec_centroid"); rp,pp=spearmanr(df["lc"],df["spec_centroid"])
print(f"\nGlobal spectral centroid (nm): top ρ={rt:+.2f} p={pt:.2g} | bot ρ={rb:+.2f} p={pb:.2g} | pooled ρ={rp:+.2f} p={pp:.2g}")
