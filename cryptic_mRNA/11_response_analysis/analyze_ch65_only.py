#!/usr/bin/env python3
"""
ch6_5-only analysis. (6,5) is the single tube with R2>=0.95 in ALL 95 wells
(min R2=0.977), so its descriptors are trustworthy everywhere -- the cleanest
single-tube test of the STMN2-CE response.

Tests, on ch6_5 alone:
 1. amplitude (gauss_max) dose trend, sensor vs control, water & serum
 2. log-linear interaction  log(amp) ~ logc * is_sensor  (sensor extra slope)
 3. shape readouts em_center (Dlambda) and fwhm dose trends
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["amp"]   = df["ch6_5_gauss_max"].clip(lower=0)
df["emc"]   = df["ch6_5_gauss_em_center"]
df["fwhm"]  = df["ch6_5_gauss_fwhm"]
df["logc"]  = np.log10(df["copies"] + 1)
df["is_sensor"] = (df.sensor == "GT15-STMN2").astype(int)

print("="*64)
print("ch6_5 (6,5) ONLY  --  R2 in [0.977, 0.999] across all 95 wells")
print("="*64)

# 1) dose trends per arm
print("\n(1) amplitude dose trend (Spearman vs logc):")
for matrix in ["water", "serum"]:
    for sensor in ["GT15-STMN2", "GT15"]:
        s = df[(df.matrix==matrix)&(df.sensor==sensor)]
        r,p = spearmanr(s["logc"], s["amp"])
        fc = s[s.copies==s.copies.max()].amp.mean()/s[s.copies==0].amp.mean()
        print(f"  {matrix:>5} {sensor:>11}: rho={r:+.2f} p={p:.1e}  "
              f"fold(max/0)={fc:.2f}x  n={len(s)}")

# 2) log-linear interaction: sensor-specific extra slope over control
print("\n(2) log-linear interaction  log10(amp) ~ logc * is_sensor:")
for matrix in ["water", "serum"]:
    sub = df[df.matrix==matrix].copy()
    sub["y"] = np.log10(sub["amp"].clip(lower=1e-9))
    m = smf.ols("y ~ logc * is_sensor", data=sub).fit()
    b = m.params.get("logc:is_sensor", np.nan)
    p = m.pvalues.get("logc:is_sensor", np.nan)
    print(f"  {matrix:>5}: sensor extra slope = {b:+.3f} per decade  p={p:.2f}")

# 3) shape readouts (the real molecular-recognition signature)
print("\n(3) shape readouts, sensor/water (should be FLAT if brightness-only):")
sw = df[(df.matrix=="water")&(df.sensor=="GT15-STMN2")]
for name, col in [("em_center (Dlambda)","emc"), ("fwhm","fwhm")]:
    r,p = spearmanr(sw["logc"], sw[col])
    span = sw.groupby("copies")[col].mean()
    print(f"  {name:20}: rho={r:+.2f} p={p:.1e}  "
          f"range across conc = {span.max()-span.min():.2f}")
print(f"\n  em_center by copies (nm):")
print("   ", sw.groupby("copies")["emc"].mean().round(2).to_dict())
