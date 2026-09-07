#!/usr/bin/env python3
"""
analyze_response.py
===================
Test whether GT15-STMN2 chirality features respond to spiked STMN2-CE mRNA
(in water and serum), using plain GT15 as the non-specific control.

For each chirality x descriptor, Spearman correlation vs log10(copies+1) within
each (sensor, matrix) group. A real, specific response = significant trend in
GT15-STMN2 but NOT in GT15.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR = ["ch8_3", "ch6_5", "ch7_5", "ch10_2", "ch9_4", "ch8_4",
        "ch7_6", "ch8_6", "ch8_7", "ch9_5", "ch10_3", "ch10_5"]
KEY_DESC = ["gauss_em_center", "gauss_max", "gauss_auc", "gauss_fwhm"]

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["logc"] = np.log10(df["copies"] + 1)

rows = []
for ch in CHIR:
    for d in KEY_DESC:
        col = f"{ch}_{d}"
        rec = {"chirality": ch, "descriptor": d}
        for matrix in ["water", "serum"]:
            for sensor, tag in [("GT15-STMN2", "sens"), ("GT15", "ctrl")]:
                sub = df[(df.sensor == sensor) & (df.matrix == matrix)]
                x = sub["logc"].to_numpy()
                y = sub[col].to_numpy()
                ok = ~np.isnan(y)
                if ok.sum() >= 5 and np.ptp(y[ok]) > 0:
                    r, p = spearmanr(x[ok], y[ok])
                else:
                    r, p = np.nan, np.nan
                rec[f"{matrix}_{tag}_r"] = r
                rec[f"{matrix}_{tag}_p"] = p
        rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(f"{HERE}/response_summary.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)


def show(matrix):
    print(f"\n===== {matrix.upper()}: sensor responds (p<0.05) but control does NOT (p>=0.05) =====")
    sub = res[(res[f"{matrix}_sens_p"] < 0.05) &
              ((res[f"{matrix}_ctrl_p"] >= 0.05) | res[f"{matrix}_ctrl_p"].isna())].copy()
    sub["abs_r"] = sub[f"{matrix}_sens_r"].abs()
    sub = sub.sort_values("abs_r", ascending=False)
    if len(sub) == 0:
        print("  (none)")
        return
    for _, r in sub.iterrows():
        cr = r[f"{matrix}_ctrl_r"]
        crs = "nan" if pd.isna(cr) else f"{cr:+.2f}"
        print(f"  {r.chirality:7s} {r.descriptor:16s} "
              f"sensor r={r[f'{matrix}_sens_r']:+.2f} (p={r[f'{matrix}_sens_p']:.1e})   "
              f"control r={crs}")


show("water")
show("serum")
print("\nSaved response_summary.csv")
