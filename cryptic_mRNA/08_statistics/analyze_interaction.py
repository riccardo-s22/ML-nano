#!/usr/bin/env python3
"""
analyze_interaction.py
======================
Specificity test: does the GT15-STMN2 sensor's response to STMN2-CE concentration
EXCEED the plain-GT15 control's? Fit, per chirality x descriptor x matrix:

    y ~ logc * is_sensor

The interaction coefficient (logc:is_sensor) is the EXTRA slope of the sensor
over the control. A specific, STMN2-CE-driven response => interaction > 0
(for intensity) and significant. This is the fair test (control's own drift is
absorbed by the main logc term; unequal n handled by the joint model).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR = ["ch8_3", "ch6_5", "ch7_5", "ch10_2", "ch9_4", "ch8_4",
        "ch7_6", "ch8_6", "ch8_7", "ch9_5", "ch10_3", "ch10_5"]
KEY_DESC = ["gauss_em_center", "gauss_max", "gauss_auc", "gauss_fwhm"]

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["logc"] = np.log10(df["copies"] + 1)
df["is_sensor"] = (df.sensor == "GT15-STMN2").astype(int)

rows = []
for matrix in ["water", "serum"]:
    sub0 = df[df.matrix == matrix]
    for ch in CHIR:
        for d in KEY_DESC:
            col = f"{ch}_{d}"
            sub = sub0[["logc", "is_sensor", col]].dropna()
            sub = sub.rename(columns={col: "y"})
            # need both groups varying in logc and y
            if sub.is_sensor.nunique() < 2 or sub.y.nunique() < 3:
                continue
            try:
                m = smf.ols("y ~ logc * is_sensor", data=sub).fit()
                rows.append({
                    "matrix": matrix, "chirality": ch, "descriptor": d,
                    "ctrl_slope": m.params.get("logc", np.nan),
                    "extra_slope": m.params.get("logc:is_sensor", np.nan),
                    "inter_p": m.pvalues.get("logc:is_sensor", np.nan),
                    "n": int(m.nobs),
                })
            except Exception:
                pass

res = pd.DataFrame(rows)
# FDR across all tests within each matrix
res["inter_q"] = np.nan
for matrix in ["water", "serum"]:
    mask = res.matrix == matrix
    p = res.loc[mask, "inter_p"].to_numpy()
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    res.loc[mask, "inter_q"] = q

res = res.sort_values(["matrix", "inter_p"])
res.to_csv(f"{HERE}/interaction_summary.csv", index=False)

pd.set_option("display.width", 200)
for matrix in ["water", "serum"]:
    print(f"\n===== {matrix.upper()}: sensor EXTRA slope over control (interaction term) =====")
    sub = res[res.matrix == matrix].head(8)
    for _, r in sub.iterrows():
        sig = "  <-- FDR<0.05" if r.inter_q < 0.05 else ""
        print(f"  {r.chirality:7s} {r.descriptor:16s} "
              f"extra_slope={r.extra_slope:+.3e}  raw_p={r.inter_p:.1e}  "
              f"FDR_q={r.inter_q:.2f}{sig}")
    n_sig = int((res[res.matrix == matrix].inter_q < 0.05).sum())
    print(f"  significant after FDR: {n_sig}")
print("\nSaved interaction_summary.csv")
