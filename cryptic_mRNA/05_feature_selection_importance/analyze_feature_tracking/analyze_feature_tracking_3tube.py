#!/usr/bin/env python3
"""
Feature-vs-total analysis restricted to the 3 reliably-fit tubes:
ch6_5, ch7_5, ch8_3  (the only chiralities with R2>=0.95 in >=81/95 wells).

total3 = sum of their amplitudes.  For every descriptor family (mean over the 3
tubes) we ask: does it co-move with total3, and does it follow the same dose
trend in the responding GT15-STMN2/water arm?
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
TUBES = ["ch6_5", "ch7_5", "ch8_3"]
FEATS = ["gauss_max","gauss_em_center","gauss_fwhm","gauss_auc","gauss_skew","gauss_kurt"]

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["total3"] = df[[f"{c}_gauss_max" for c in TUBES]].clip(lower=0).sum(axis=1)
df["logc"]   = np.log10(df["copies"] + 1)

sw = df[(df.sensor == "GT15-STMN2") & (df.matrix == "water")]
rt, pt = spearmanr(sw["logc"], sw["total3"])
print(f"REFERENCE  total3 (ch6_5+ch7_5+ch8_3) vs logc, sensor/water: "
      f"rho={rt:+.2f} p={pt:.1e}\n")

def family(feat, frame):
    return frame[[f"{c}_{feat}" for c in TUBES]].mean(axis=1)

print(f"{'feature':14} | {'rho vs TOTAL3':>13} {'p':>8} | {'dose rho':>9} {'p':>8} | trend")
print("-"*70)
for feat in FEATS:
    fall = family(feat, df); fsw = family(feat, sw)
    r_tot,p_tot = spearmanr(df["total3"], fall, nan_policy="omit")
    r_dose,p_dose = spearmanr(sw["logc"], fsw, nan_policy="omit")
    same = (np.sign(r_dose)==np.sign(rt)) and p_dose<0.05
    flag = "YES" if same else ("weak" if p_dose<0.2 and np.sign(r_dose)==np.sign(rt) else "no")
    print(f"{feat:14} | {r_tot:+13.2f} {p_tot:8.1e} | {r_dose:+9.2f} {p_dose:8.1e} | {flag}")

# per-tube breakdown: amplitude vs em_center (the shape/Dlambda readout)
print("\nPer-tube dose trend (sensor/water), rho vs logc:")
print(f"{'tube':7} | {'amp rho':>8} {'p':>7} | {'em_center rho':>13} {'p':>7} | "
      f"{'fwhm rho':>9} {'p':>7}")
for c in TUBES:
    ra,pa = spearmanr(sw["logc"], sw[f"{c}_gauss_max"])
    rc,pc = spearmanr(sw["logc"], sw[f"{c}_gauss_em_center"])
    rf,pf = spearmanr(sw["logc"], sw[f"{c}_gauss_fwhm"])
    print(f"{c:7} | {ra:+8.2f} {pa:7.1e} | {rc:+13.2f} {pc:7.1e} | {rf:+9.2f} {pf:7.1e}")

# control comparison: does total3 in the GT15-only control also rise?
cw = df[(df.sensor == "GT15") & (df.matrix == "water")]
rcw,pcw = spearmanr(cw["logc"], cw["total3"])
print(f"\nGT15-only control, total3 vs logc (water): rho={rcw:+.2f} p={pcw:.1e}  (n={len(cw)})")
print(f"fold-change sensor:  total3@max / total3@0 = "
      f"{sw[sw.copies==sw.copies.max()].total3.mean()/sw[sw.copies==0].total3.mean():.2f}x")
print(f"fold-change control: total3@max / total3@0 = "
      f"{cw[cw.copies==cw.copies.max()].total3.mean()/cw[cw.copies==0].total3.mean():.2f}x")
