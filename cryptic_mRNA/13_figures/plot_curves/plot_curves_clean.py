#!/usr/bin/env python3
"""Dose curves on the clean tube set: total3 (ch6_5+ch7_5+ch8_3) and ch6_5 alone.
Absolute (own y-scales) + normalized-to-0-conc.  Replaces the all-12-sum version."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/mnt/c/Users/riccardo-s/Documents/CNT/cryptic_mRNA/exp1"
TUBES = ["ch6_5", "ch7_5", "ch8_3"]
COPIES = [0, 10, 100, 1000, 10000, 100000]
XLAB = ["0", "10", "1e2", "1e3", "1e4", "1e5"]
SENSORS = ["GT15-STMN2", "GT15"]; MATRICES = ["water", "serum"]
COLOR = {"GT15-STMN2": "C0", "GT15": "C3"}

df = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")
df["total3"] = df[[f"{c}_gauss_max" for c in TUBES]].clip(lower=0).sum(axis=1)
df["ch6_5"]  = df["ch6_5_gauss_max"].clip(lower=0)

def agg(metric):
    return (df.groupby(["sensor","matrix","copies"])[metric]
              .agg(["mean","sem"]).reset_index())

def get(a, sensor, matrix):
    s = a[(a.sensor==sensor)&(a.matrix==matrix)].set_index("copies").reindex(COPIES)
    return s["mean"].to_numpy(), s["sem"].to_numpy()

for metric, tag, title in [("total3","total3","total3 = ch6_5+ch7_5+ch8_3"),
                            ("ch6_5","ch6_5","ch6_5 (6,5) amplitude only")]:
    a = agg(metric)
    # absolute, separated
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, matrix in enumerate(MATRICES):
        for j, sensor in enumerate(SENSORS):
            m, e = get(a, sensor, matrix)
            ax[i,j].errorbar(range(6), m, yerr=e, marker="o", color=COLOR[sensor], capsize=3)
            ax[i,j].set_title(f"{sensor} | {matrix}")
            ax[i,j].set_xticks(range(6)); ax[i,j].set_xticklabels(XLAB)
            ax[i,j].set_xlabel("STMN2-CE copies/uL"); ax[i,j].set_ylabel(f"{metric} (gauss_max)")
            ax[i,j].grid(alpha=0.3)
    fig.suptitle(f"Absolute dose curves — {title}", y=1.00); fig.tight_layout()
    fig.savefig(f"{HERE}/curves_{tag}_absolute.png", dpi=130, bbox_inches="tight")

    # normalized to 0-conc
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, matrix in enumerate(MATRICES):
        for sensor in SENSORS:
            m, e = get(a, sensor, matrix); m0 = m[0]
            ax2[i].errorbar(range(6), m/m0, yerr=e/m0, marker="o",
                            color=COLOR[sensor], capsize=3, label=sensor)
        ax2[i].axhline(1.0, color="gray", lw=0.8, ls="--")
        ax2[i].set_title(f"{matrix} (norm to 0)"); ax2[i].set_xticks(range(6))
        ax2[i].set_xticklabels(XLAB); ax2[i].set_xlabel("STMN2-CE copies/uL")
        ax2[i].set_ylabel("fold-change vs 0 conc"); ax2[i].grid(alpha=0.3); ax2[i].legend()
    fig2.suptitle(f"Normalized dose curves — {title}", y=1.02); fig2.tight_layout()
    fig2.savefig(f"{HERE}/curves_{tag}_normalized.png", dpi=130, bbox_inches="tight")

    print(f"\n=== {title} ===")
    print(f"{'matrix':>6} {'sensor':>11} | " + " ".join(f"{x:>9}" for x in XLAB))
    for matrix in MATRICES:
        for sensor in SENSORS:
            m,_ = get(a, sensor, matrix)
            print(f"{matrix:>6} {sensor:>11} | " + " ".join(f"{v:9.2e}" for v in m))
    print("fold-change vs 0:")
    for matrix in MATRICES:
        for sensor in SENSORS:
            m,_ = get(a, sensor, matrix)
            print(f"{matrix:>6} {sensor:>11} | " + " ".join(f"{v:9.2f}" for v in m/m[0]))
print("\nSaved curves_total3_*.png, curves_ch6_5_*.png")
