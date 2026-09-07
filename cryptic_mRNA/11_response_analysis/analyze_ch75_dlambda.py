#!/usr/bin/env python3
"""
(7,5)-specific Delta-lambda analysis -- the paper's designated sensor readout.
Convention (paper): Dlambda = lambda0 - lambda, lambda0 = mean center at 0 copies.
  -> positive Dlambda = BLUE shift = expected hybridization signal.

(7,5) is reportedly the most sensitive chirality. Tests, R2>0.95 gated, water:
 1. Dlambda vs dose, GT15-STMN2 sensor vs GT15 control (sensor should blue-shift,
    control -- which lacks the STMN2 capture domain -- should not).
 2. Brightness isolation: em_center ~ log_amp * is_sensor. A real hybridization
    blue-shift = sensor sits BLUER than control at matched brightness (negative
    is_sensor / interaction), separating it from the generic amp-center coupling.
 3. Magnitude vs the estimator noise floor (~4 nm from the fixed-center simulation).
Reports both Gaussian center and fit-free centroid for cross-check.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

HERE="/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
g=pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
f=pd.read_csv(f"{HERE}/fit_free_descriptors.csv")
g=g.merge(f[["well","ch7_5_ff_centroid","ch7_5_ff_integ"]],on="well")
g["logc"]=np.log10(g["copies"]+1)
g["lamp"]=np.log10(g["ch7_5_gauss_max"].clip(lower=1e-30))
g["is_sensor"]=(g.sensor=="GT15-STMN2").astype(int)
W=g[(g.matrix=="water")&(g["ch7_5_gauss_r2"]>0.95)].copy()

print("="*78); print("(7,5) Delta-lambda  [paper readout]  -- water, R2>0.95"); print("="*78)

for est,col,unit in [("Gaussian center","ch7_5_gauss_em_center","gauss"),
                     ("fit-free centroid","ch7_5_ff_centroid","centroid")]:
    print(f"\n--- estimator: {est} ---")
    for sensor in ["GT15-STMN2","GT15"]:
        s=W[W.sensor==sensor]
        lam0=s[s.copies==0][col].mean()
        s=s.assign(dlam=lam0 - s[col])  # paper convention: + = blue
        r,p=spearmanr(s["logc"],s["dlam"])
        prog=s.groupby("copies")["dlam"].mean()
        print(f"  {sensor:11}: rho(Dlam,dose)={r:+.2f} p={p:.2f}  "
              f"Dlam@maxconc={prog.iloc[-1]:+.2f}nm  (n={len(s)})")
        print(f"      Dlam by conc: " +
              " ".join(f"{c}:{v:+.2f}" for c,v in prog.items()))

# brightness isolation: does the sensor sit bluer than control at matched amp?
print("\n--- brightness isolation: em_center ~ lamp * is_sensor ---")
m=smf.ols("ch7_5_gauss_em_center ~ lamp * is_sensor",data=W).fit()
print(f"  is_sensor (sensor offset at matched brightness): "
      f"{m.params['is_sensor']:+.2f} nm  p={m.pvalues['is_sensor']:.3f}")
print(f"  lamp:is_sensor (extra blue-slope per brightness): "
      f"{m.params['lamp:is_sensor']:+.2f}  p={m.pvalues['lamp:is_sensor']:.3f}")
print(f"  (negative + significant => sensor genuinely bluer than control "
      f"beyond the generic brightness coupling)")

# pure dose effect after removing brightness
m2=smf.ols("ch7_5_gauss_em_center ~ logc + lamp + is_sensor + logc:is_sensor",data=W).fit()
print(f"\n  dose:is_sensor (sensor-specific dose shift, brightness-controlled): "
      f"{m2.params.get('logc:is_sensor',np.nan):+.3f} nm/dec  "
      f"p={m2.pvalues.get('logc:is_sensor',np.nan):.3f}")

print(f"\n  estimator noise floor (fixed-center sim, (7,5)): gauss ~9 nm, centroid ~4 nm")
print(f"  emission sampling: {np.median(np.diff(pd.read_csv(f'{HERE}/emission.txt')['Emission'])):.2f} nm/point")
