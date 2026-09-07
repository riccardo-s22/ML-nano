#!/usr/bin/env python3
"""
Feature-vs-total analysis, but using ONLY peaks with fit R2 > 0.95.

Per (well, chirality): if gauss_r2 <= 0.95, that peak's features -> NaN.
- total_good = nan-aware sum of passing amplitudes (per well)
- family aggregates = nan-aware mean over passing chiralities
Shape features (em_center, fwhm) benefit most: a bad fit has a meaningless center.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR = ["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4",
        "ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
FEATS = ["gauss_max","gauss_em_center","gauss_fwhm","gauss_auc",
         "gauss_skew","gauss_kurt"]
R2MIN = 0.95

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")

# build a masked copy: each chirality's features NaN where its R2<=0.95
M = {}
for c in CHIR:
    ok = df[f"{c}_gauss_r2"] > R2MIN
    for f in FEATS:
        M[f"{c}_{f}"] = df[f"{c}_{f}"].where(ok)
M = pd.DataFrame(M)
for k in ["sensor","matrix","copies"]:
    M[k] = df[k]

# total over passing peaks (nan-aware), plus original all-12 total for reference
M["total_good"] = M[[f"{c}_gauss_max" for c in CHIR]].sum(axis=1, min_count=1)
M["total_all"]  = df[[f"{c}_gauss_max" for c in CHIR]].clip(lower=0).sum(axis=1)
M["logc"] = np.log10(M["copies"] + 1)

sw = M[(M.sensor == "GT15-STMN2") & (M.matrix == "water")]

rt, pt = spearmanr(sw["logc"], sw["total_good"])
ra, pa = spearmanr(sw["logc"], sw["total_all"])
print(f"total_good (R2>0.95 peaks) vs logc, sensor/water: rho={rt:+.2f} p={pt:.1e}")
print(f"total_all  (all 12 peaks)  vs logc, sensor/water: rho={ra:+.2f} p={pa:.1e}")
print(f"agreement total_good~total_all (all wells): rho={spearmanr(M.total_good,M.total_all)[0]:+.2f}\n")

def family(feat, frame):
    return frame[[f"{c}_{feat}" for c in CHIR]].mean(axis=1)  # skips NaN by default

print(f"{'feature':14} | {'rho vs TOTAL':>12} {'p':>8} | {'dose rho':>9} {'p':>8} | trend")
print("-"*68)
for feat in FEATS:
    fall = family(feat, M); fsw = family(feat, sw)
    r_tot,p_tot = spearmanr(M["total_good"], fall, nan_policy="omit")
    r_dose,p_dose = spearmanr(sw["logc"], fsw, nan_policy="omit")
    same = (np.sign(r_dose)==np.sign(rt)) and p_dose<0.05
    flag = "YES" if same else ("weak" if p_dose<0.2 and np.sign(r_dose)==np.sign(rt) else "no")
    print(f"{feat:14} | {r_tot:+12.2f} {p_tot:8.1e} | {r_dose:+9.2f} {p_dose:8.1e} | {flag}")

# per-chirality amplitude & em_center dose trend, R2-filtered
print("\nPer-chirality (R2>0.95) dose rho vs logc, sensor/water:")
print(f"{'chir':7} {'n_ok':>4} | {'amp rho':>8} {'p':>7} | {'em_center rho':>13} {'p':>7}")
for c in CHIR:
    s = sw[[f"{c}_gauss_max", f"{c}_gauss_em_center", "logc"]].dropna()
    n = len(s)
    if n < 5:
        print(f"{c:7} {n:>4} |  (too few passing wells)"); continue
    ra,pa = spearmanr(s["logc"], s[f"{c}_gauss_max"])
    rc,pc = spearmanr(s["logc"], s[f"{c}_gauss_em_center"])
    print(f"{c:7} {n:>4} | {ra:+8.2f} {pa:7.1e} | {rc:+13.2f} {pc:7.1e}")
