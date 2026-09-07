#!/usr/bin/env python3
"""Separated GT15-STMN2 vs GT15 dose curves (absolute + normalized-to-0-conc)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
CHIR = ["ch8_3","ch6_5","ch7_5","ch10_2","ch9_4","ch8_4",
        "ch7_6","ch8_6","ch8_7","ch9_5","ch10_3","ch10_5"]
COPIES = [0, 10, 100, 1000, 10000, 100000]
XLAB = ["0", "10", "1e2", "1e3", "1e4", "1e5"]

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["total"] = df[[f"{c}_gauss_max" for c in CHIR]].clip(lower=0).sum(axis=1)

# aggregate: mean / sem / n per (sensor, matrix, copies)
agg = (df.groupby(["sensor", "matrix", "copies"])["total"]
         .agg(["mean", "sem", "count"]).reset_index())

def get(sensor, matrix):
    s = agg[(agg.sensor == sensor) & (agg.matrix == matrix)].set_index("copies")
    s = s.reindex(COPIES)
    return s["mean"].to_numpy(), s["sem"].to_numpy()

SENSORS = ["GT15-STMN2", "GT15"]
MATRICES = ["water", "serum"]
COLOR = {"GT15-STMN2": "C0", "GT15": "C3"}

# ---- Figure 1: absolute, separated (each its own y-scale) ----
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for i, matrix in enumerate(MATRICES):
    for j, sensor in enumerate(SENSORS):
        m, e = get(sensor, matrix)
        a = ax[i, j]
        a.errorbar(range(6), m, yerr=e, marker="o", color=COLOR[sensor], capsize=3)
        a.set_title(f"{sensor}  |  {matrix}")
        a.set_xticks(range(6)); a.set_xticklabels(XLAB)
        a.set_xlabel("STMN2-CE copies/uL")
        a.set_ylabel("total intensity (sum gauss_max)")
        a.grid(alpha=0.3)
fig.suptitle("Absolute dose curves (separate y-axes)", y=1.00)
fig.tight_layout()
fig.savefig(f"{HERE}/curves_absolute.png", dpi=130, bbox_inches="tight")

# ---- Figure 2: normalized to 0-concentration (fold-change) ----
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.4))
for i, matrix in enumerate(MATRICES):
    a = ax2[i]
    for sensor in SENSORS:
        m, e = get(sensor, matrix)
        m0 = m[0]
        a.errorbar(range(6), m / m0, yerr=e / m0, marker="o",
                   color=COLOR[sensor], capsize=3, label=sensor)
    a.axhline(1.0, color="gray", lw=0.8, ls="--")
    a.set_title(f"{matrix}  (normalized to 0 copies)")
    a.set_xticks(range(6)); a.set_xticklabels(XLAB)
    a.set_xlabel("STMN2-CE copies/uL")
    a.set_ylabel("fold-change vs 0 conc")
    a.grid(alpha=0.3); a.legend()
fig2.suptitle("Normalized dose curves (each curve / its own 0-conc value)", y=1.02)
fig2.tight_layout()
fig2.savefig(f"{HERE}/curves_normalized.png", dpi=130, bbox_inches="tight")

# ---- print underlying numbers ----
print(f"{'matrix':>6} {'sensor':>11} | " + " ".join(f"{x:>9}" for x in XLAB))
for matrix in MATRICES:
    for sensor in SENSORS:
        m, _ = get(sensor, matrix)
        print(f"{matrix:>6} {sensor:>11} | " + " ".join(f"{v:9.2e}" for v in m))
print("\nNormalized to 0-conc (fold-change):")
print(f"{'matrix':>6} {'sensor':>11} | " + " ".join(f"{x:>6}" for x in XLAB))
for matrix in MATRICES:
    for sensor in SENSORS:
        m, _ = get(sensor, matrix)
        print(f"{matrix:>6} {sensor:>11} | " + " ".join(f"{v:6.2f}" for v in m / m[0]))
print("\nSaved curves_absolute.png, curves_normalized.png")
