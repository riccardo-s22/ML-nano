#!/usr/bin/env python3
"""
analyze_position_ratio.py
=========================
(1) Column-position sanity check: is the apparent dose response a plate-column
    artifact / common-mode drift rather than STMN2-CE-specific?
    - per-column total-intensity profiles for sensor vs control
    - reset test: both water (cols 1-6) and serum (cols 7-12) run 0..1e5 copies,
      so a concentration effect RESETS at col 7; a left->right positional
      gradient would instead keep climbing.

(2) Self-normalizing features that cancel common-mode amplitude:
    - fraction f_ch = gauss_max_ch / sum_ch(gauss_max)   (per well)
    Interaction test (f ~ logc * is_sensor) finds chirality-specific responses.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR = ["ch8_3", "ch6_5", "ch7_5", "ch10_2", "ch9_4", "ch8_4",
        "ch7_6", "ch8_6", "ch8_7", "ch9_5", "ch10_3", "ch10_5"]

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
maxcols = [f"{ch}_gauss_max" for ch in CHIR]
df["total"] = df[maxcols].clip(lower=0).sum(axis=1)
df["logc"] = np.log10(df["copies"] + 1)
df["is_sensor"] = (df.sensor == "GT15-STMN2").astype(int)

# ----------------------------------------------------------------------------
# (1) Per-column profiles + reset test
# ----------------------------------------------------------------------------
print("=" * 70)
print("(1) COLUMN-POSITION SANITY CHECK  -- total intensity (sum gauss_max)")
print("=" * 70)
prof = (df.groupby(["sensor", "col"])["total"]
          .agg(["mean", "sem", "count"]).reset_index())
COPIES = [0, 10, 100, 1000, 10000, 100000]
print(f"\n{'col':>3} {'copies':>7} {'matrix':>6} | "
      f"{'GT15-STMN2 mean':>16} | {'GT15 ctrl mean':>15}")
for col in range(1, 13):
    copies = COPIES[(col - 1) if col <= 6 else (col - 7)]
    matrix = "water" if col <= 6 else "serum"
    s = prof[(prof.sensor == "GT15-STMN2") & (prof.col == col)]["mean"]
    c = prof[(prof.sensor == "GT15") & (prof.col == col)]["mean"]
    sv = s.values[0] if len(s) else np.nan
    cv = c.values[0] if len(c) else np.nan
    print(f"{col:>3} {copies:>7} {matrix:>6} | {sv:>16.3e} | {cv:>15.3e}")

# correlation of sensor vs control column profiles (common-mode signature)
piv = prof.pivot(index="col", columns="sensor", values="mean")
r_cm = piv["GT15-STMN2"].corr(piv["GT15"])
print(f"\nSensor vs control column-profile correlation: r = {r_cm:.3f}")
print("(high r => both move together => common-mode, not capture-specific)")

# reset test: compare col6 (water 1e5) -> col7 (serum 0)
for sensor in ["GT15-STMN2", "GT15"]:
    g = prof[prof.sensor == sensor].set_index("col")["mean"]
    print(f"  {sensor:11s}: col6(1e5)={g.get(6, np.nan):.3e}  "
          f"col7(0)={g.get(7, np.nan):.3e}  "
          f"-> {'RESETS down' if g.get(7,1) < g.get(6,0) else 'keeps climbing'}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, matrix, cols in [(axes[0], "water", range(1, 7)),
                          (axes[1], "serum", range(7, 13))]:
    xcop = [COPIES[(c - 1) if c <= 6 else (c - 7)] for c in cols]
    for sensor, color in [("GT15-STMN2", "C0"), ("GT15", "C3")]:
        sub = prof[(prof.sensor == sensor) & (prof.col.isin(cols))].sort_values("col")
        ax.errorbar(range(len(sub)), sub["mean"], yerr=sub["sem"],
                    marker="o", color=color, label=sensor, capsize=3)
    ax.set_xticks(range(6))
    ax.set_xticklabels(xcop, rotation=45)
    ax.set_xlabel("STMN2-CE copies/uL")
    ax.set_title(f"{matrix}")
    ax.set_ylabel("total intensity (sum gauss_max)")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{HERE}/column_profile.png", dpi=130)
print(f"\nSaved column_profile.png")

# ----------------------------------------------------------------------------
# (2) Self-normalizing fraction features  -> interaction test
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("(2) COMMON-MODE-CANCELLED FRACTION FEATURES  f_ch = max_ch / total")
print("=" * 70)
for ch in CHIR:
    df[f"frac_{ch}"] = df[f"{ch}_gauss_max"].clip(lower=0) / df["total"]

rows = []
for matrix in ["water", "serum"]:
    sub0 = df[df.matrix == matrix]
    for ch in CHIR:
        sub = sub0[["logc", "is_sensor", f"frac_{ch}"]].dropna()
        sub = sub.rename(columns={f"frac_{ch}": "y"})
        if sub.is_sensor.nunique() < 2 or sub.y.nunique() < 3:
            continue
        m = smf.ols("y ~ logc * is_sensor", data=sub).fit()
        rows.append({"matrix": matrix, "chirality": ch,
                     "extra_slope": m.params.get("logc:is_sensor", np.nan),
                     "inter_p": m.pvalues.get("logc:is_sensor", np.nan)})
res = pd.DataFrame(rows)
res["inter_q"] = np.nan
for matrix in ["water", "serum"]:
    mask = res.matrix == matrix
    p = res.loc[mask, "inter_p"].to_numpy()
    res.loc[mask, "inter_q"] = multipletests(p, method="fdr_bh")[1]
res = res.sort_values(["matrix", "inter_p"])
res.to_csv(f"{HERE}/fraction_interaction_summary.csv", index=False)

for matrix in ["water", "serum"]:
    print(f"\n--- {matrix.upper()}: sensor-specific change in chirality fraction ---")
    sub = res[res.matrix == matrix].head(6)
    for _, r in sub.iterrows():
        sig = "  <-- FDR<0.05" if r.inter_q < 0.05 else ""
        print(f"  {r.chirality:7s} extra_slope={r.extra_slope:+.3e}  "
              f"raw_p={r.inter_p:.1e}  FDR_q={r.inter_q:.2f}{sig}")
    print(f"  significant after FDR: {int((res[res.matrix==matrix].inter_q<0.05).sum())}")
print("\nSaved fraction_interaction_summary.csv")
