#!/usr/bin/env python3
"""
Which features move with total intensity?

total = sum_ch(gauss_max).  For every other descriptor we ask two things:
 (A) per-well co-movement: Spearman(feature, total) across all 95 wells, and
     restricted to the GT15-STMN2 / water arm (where total actually rises).
 (B) same dose trend: Spearman(feature, log10 copies) in the sensor/water arm,
     compared to total's own dose Spearman. "Same trend" = same sign & sig.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR = ["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4",
        "ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
FEATS = ["gauss_max","gauss_em_center","gauss_fwhm","gauss_auc",
         "gauss_skew","gauss_kurt","gauss_r2"]

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["total"] = df[[f"{c}_gauss_max" for c in CHIR]].clip(lower=0).sum(axis=1)
df["logc"]  = np.log10(df["copies"] + 1)

sw = df[(df.sensor == "GT15-STMN2") & (df.matrix == "water")]   # the arm that responds

# total's own dose trend (reference)
rt, pt = spearmanr(sw["logc"], sw["total"])
print(f"REFERENCE  total intensity vs logc (sensor/water): rho={rt:+.2f} p={pt:.1e}\n")

# ---- aggregate each feature-family across chiralities (mean over 12 tubes) ----
def family(feat, frame):
    cols = [f"{c}_{feat}" for c in CHIR]
    return frame[cols].mean(axis=1)

print(f"{'feature':14} | {'rho vs TOTAL':>12} {'p':>8} | "
      f"{'rho vs logc':>11} {'p':>8} | same trend?")
print("-"*78)
rows=[]
for feat in FEATS:
    fall = family(feat, df)
    fsw  = family(feat, sw)
    r_tot, p_tot = spearmanr(df["total"], fall)          # co-move with total (all wells)
    r_dose, p_dose = spearmanr(sw["logc"], fsw)          # own dose trend (sensor/water)
    same = (np.sign(r_dose) == np.sign(rt)) and (p_dose < 0.05)
    rows.append((feat, r_tot, p_tot, r_dose, p_dose, same))
    flag = "YES" if same else ("weak" if p_dose < 0.2 and np.sign(r_dose)==np.sign(rt) else "no")
    print(f"{feat:14} | {r_tot:+12.2f} {p_tot:8.1e} | "
          f"{r_dose:+11.2f} {p_dose:8.1e} | {flag}")

# ---- per-chirality amplitude: do all 12 tubes rise, or just some? ----
print("\nPer-chirality gauss_max dose trend (sensor/water), rho vs logc:")
amp = []
for c in CHIR:
    r,p = spearmanr(sw["logc"], sw[f"{c}_gauss_max"])
    amp.append((c,r,p))
for c,r,p in sorted(amp, key=lambda x:-x[1]):
    print(f"  {c:7} rho={r:+.2f} p={p:.1e}{'  *' if p<0.05 else ''}")

# ---- per-chirality AUC (area), the other 'amount of light' feature ----
print("\nPer-chirality gauss_auc dose trend (sensor/water), rho vs logc:")
for c in CHIR:
    r,p = spearmanr(sw["logc"], sw[f"{c}_gauss_auc"])
    print(f"  {c:7} rho={r:+.2f} p={p:.1e}{'  *' if p<0.05 else ''}")
