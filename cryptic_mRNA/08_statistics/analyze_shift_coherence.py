#!/usr/bin/env python3
"""
Deeper look at the FITTED peak centers: are the shifts COHERENT (all tubes move
the same way with STMN2-CE, as a real corona rearrangement would) or incoherent
(brightness-coupled fitting drift)?

Uses Gaussian gauss_em_center, R2>0.95 gated per (well,tube). For each tube,
sensor/water arm:
  - slope of em_center vs logc, sign & p
  - PARTIAL test: em_center ~ logc + log10(amp). Does logc survive after
    controlling for brightness? (if not -> shift is a brightness artifact)
Coherence across tubes:
  - sign test on the per-tube dose slopes (consistent direction?)
  - brightness-coupling sanity: in the CONTROL arm (no dose response) does
    em_center still move with amplitude? If yes -> the coupling is instrumental.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
import statsmodels.formula.api as smf

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR=["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4",
      "ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
df=pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["logc"]=np.log10(df["copies"]+1)

def arm(sensor): return df[(df.sensor==sensor)&(df.matrix=="water")]

print("="*86)
print("PER-TUBE FITTED-CENTER SHIFT (R2>0.95), sensor/water")
print("="*86)
print(f"{'tube':7} {'n_ok':>4} {'slope nm/dec':>12} {'p_dose':>8} | "
      f"{'partial p(logc|amp)':>19} | {'dir':>5}")
slopes=[]
for ch in CHIR:
    s=arm("GT15-STMN2")[[f"{ch}_gauss_em_center",f"{ch}_gauss_max",
                         f"{ch}_gauss_r2","logc"]].copy()
    s.columns=["emc","amp","r2","logc"]
    s=s[s.r2>0.95].dropna()
    if len(s)<8 or s.logc.nunique()<3:
        print(f"{ch:7} {len(s):>4}   (too few good wells)"); continue
    s["lamp"]=np.log10(s["amp"].clip(lower=1e-30))
    m1=smf.ols("emc ~ logc",data=s).fit()
    sl=m1.params["logc"]; pd_=m1.pvalues["logc"]
    m2=smf.ols("emc ~ logc + lamp",data=s).fit()
    pp=m2.pvalues.get("logc",np.nan)
    slopes.append((ch,sl,pd_,pp))
    print(f"{ch:7} {len(s):>4} {sl:+12.3f} {pd_:8.1e} | {pp:19.2f} | "
          f"{'blue' if sl<0 else 'red':>5}")

# ---- coherence: are the per-tube dose slopes consistently signed? ----
sl=np.array([x[1] for x in slopes]); names=[x[0] for x in slopes]
nblue=int((sl<0).sum()); nred=int((sl>0).sum())
bt=binomtest(max(nblue,nred), nblue+nred, 0.5)
print("\n" + "="*86)
print("COHERENCE OF DIRECTION ACROSS TUBES")
print("="*86)
print(f"  blue (slope<0): {nblue}   red (slope>0): {nred}   of {len(sl)} tubes")
print(f"  sign test vs 50/50 random: p = {bt.pvalue:.2f}")
print(f"  mean slope = {sl.mean():+.3f} nm/decade (SD {sl.std():.3f})  "
      f"-> {'COHERENT' if bt.pvalue<0.05 else 'NOT coherent (signs ~random)'}")

# ---- brightness coupling sanity: does em_center move with amp in CONTROL? ----
print("\n" + "="*86)
print("BRIGHTNESS-COUPLING CHECK  (control arm: no dose response, but amp varies)")
print("  if em_center tracks amplitude here, the 'shift' is an instrumental/fit artifact")
print("="*86)
print(f"{'tube':7} {'rho(emc,amp) SENSOR':>20} {'rho(emc,amp) CONTROL':>21}")
for ch in ["ch6_5","ch7_5","ch8_3","ch10_2","ch7_6"]:
    out=[]
    for sensor in ["GT15-STMN2","GT15"]:
        g=arm(sensor)[[f"{ch}_gauss_em_center",f"{ch}_gauss_max",f"{ch}_gauss_r2"]].copy()
        g.columns=["emc","amp","r2"]; g=g[g.r2>0.95].dropna()
        if len(g)<6: out.append(np.nan); continue
        r,_=spearmanr(g["amp"],g["emc"]); out.append(r)
    print(f"{ch:7} {out[0]:+20.2f} {out[1]:+21.2f}")
