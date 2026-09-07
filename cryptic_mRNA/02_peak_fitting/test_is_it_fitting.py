#!/usr/bin/env python3
"""
Is the brightness->center coupling a FITTING problem (estimator artifact) or a
real shift in the raw peak?

TEST A (empirical): for ch6_5/ch7_5/ch8_3, correlate THREE position estimators
  with amplitude, in sensor & control:
    - argmax_nm  : raw grid maximum, NO model, NO baseline (coarse ~1.6nm/pt)
    - centroid   : fit-free moment (baseline+window dependent)
    - gauss_emc  : full split-Gaussian fit
  If only gauss moves -> Gaussian-model problem. If centroid moves too but argmax
  doesn't -> sub-pixel estimation/baseline artifact. If argmax moves too -> raw data.

TEST B (simulation, decisive): take a real high-SNR peak shape, FIX its true
  center, rescale amplitude across the observed range on a fixed baseline + matched
  noise, re-run the SAME estimators. If recovered center correlates with amplitude
  while truth is constant -> the procedure manufactures the shift = analysis artifact.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from extract_gaussian import (fit_chirality, CHIRALITY, WINDOW_NM, HERE, EXCITATIONS,
                              assign_excitation)

em=pd.read_csv(f"{HERE}/emission.txt")["Emission"].to_numpy(float)
spectra={e:pd.read_csv(f"{HERE}/{e}.txt",sep="\t") for e in EXCITATIONS}
exmap=assign_excitation()
g=pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
g["logc"]=np.log10(g["copies"]+1)

def win(ch,well):
    sp=spectra[exmap[ch]][well].to_numpy(float)
    sel=np.abs(em-CHIRALITY[ch]["em"])<=WINDOW_NM
    return em[sel], sp[sel]

def estimators(x,y):
    amax=x[np.argmax(y)]
    yl,yr=np.median(y[:3]),np.median(y[-3:])
    base=np.interp(x,[x[0],x[-1]],[yl,yr]); ip=np.clip(y-base,0,None)
    cen=np.trapz(x*ip,x)/np.trapz(ip,x) if np.trapz(ip,x)>0 else np.nan
    return amax,cen

print("="*84)
print("TEST A — three position estimators vs amplitude (water)")
print("="*84)
print(f"{'tube':6} {'arm':11} | {'rho(argmax,amp)':>15} {'rho(centroid,amp)':>17} "
      f"{'rho(gaussEMC,amp)':>17}")
for ch in ["ch6_5","ch7_5","ch8_3"]:
    for sensor in ["GT15-STMN2","GT15"]:
        sub=g[(g.sensor==sensor)&(g.matrix=="water")&(g[f"{ch}_gauss_r2"]>0.95)]
        amax=[]; cen=[]; amp=[]; emc=[]
        for _,r in sub.iterrows():
            x,y=win(ch,r["well"]); a,c=estimators(x,y)
            amax.append(a); cen.append(c); amp.append(r[f"{ch}_gauss_max"]); emc.append(r[f"{ch}_gauss_em_center"])
        if len(amp)<6: continue
        ra=spearmanr(amp,amax)[0]; rc=spearmanr(amp,cen)[0]; rg=spearmanr(amp,emc)[0]
        print(f"{ch:6} {sensor:11} | {ra:+15.2f} {rc:+17.2f} {rg:+17.2f}")

print("\n"+"="*84)
print("TEST B — SIMULATION: fixed true center, vary amplitude. Does center drift?")
print("="*84)
rng=np.random.default_rng(7)
for ch in ["ch6_5","ch7_5","ch8_3"]:
    # template shape from the single brightest, high-R2 well
    sub=g[(g.sensor=="GT15-STMN2")&(g[f"{ch}_gauss_r2"]>0.97)]
    bw=sub.loc[sub[f"{ch}_gauss_max"].idxmax(),"well"]
    x,y=win(ch,bw); yl,yr=np.median(y[:3]),np.median(y[-3:])
    base=np.interp(x,[x[0],x[-1]],[yl,yr]); shape=np.clip(y-base,0,None)
    shape=shape/shape.max()                         # unit-amplitude template
    true_c=np.trapz(x*shape,x)/np.trapz(shape,x)    # its fixed centroid
    # amplitude range & noise level observed for this tube
    amps=g[g[f"{ch}_gauss_max"].notna()][f"{ch}_gauss_max"]
    a_lo,a_hi=np.percentile(amps,[10,95])
    # noise: residual scatter in the bright window
    noise=np.std(np.diff(y))/np.sqrt(2)
    base_lvl=min(yl,yr)
    A=np.linspace(a_lo,a_hi,40); rec_emc=[]; rec_cen=[]; rec_amax=[]
    for a in A:
        yi=base_lvl + a*shape + rng.normal(0,noise,size=len(x))
        d=fit_chirality(em, np.where(np.isin(em,x),  # place window back on full axis
                                     np.interp(em,x,yi), 0.0), CHIRALITY[ch]["em"])
        amax,cen=estimators(x,yi)
        rec_emc.append(d["gauss_em_center"]); rec_cen.append(cen); rec_amax.append(amax)
    rec_emc=np.array(rec_emc,float); rec_cen=np.array(rec_cen,float)
    re=spearmanr(A,rec_emc,nan_policy="omit")[0]; rc=spearmanr(A,rec_cen)[0]
    print(f"{ch:6}: true_center fixed={true_c:.2f}nm | "
          f"recovered gauss span={np.nanmax(rec_emc)-np.nanmin(rec_emc):.2f}nm "
          f"rho(emc,amp)={re:+.2f} | centroid span={np.nanmax(rec_cen)-np.nanmin(rec_cen):.2f}nm "
          f"rho={rc:+.2f}")
