#!/usr/bin/env python3
"""Diagnose whether low R2 = genuine dim peaks or a fitting bug.
Plots the actual emission window + fitted split-Gaussian for a good tube (ch6_5)
and the failing ones (ch8_6, ch10_5), in a bright well vs the worst-R2 well.
Also: R2-vs-amplitude scatter, and how much real signal sits in each window."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from extract_gaussian import (split_gaussian_with_baseline, fit_chirality,
                              assign_excitation, CHIRALITY, WINDOW_NM, HERE, EXCITATIONS)

em_axis = pd.read_csv(f"{HERE}/emission.txt")["Emission"].to_numpy(float)
spectra = {e: pd.read_csv(f"{HERE}/{e}.txt", sep="\t") for e in EXCITATIONS}
ex_map = assign_excitation()
desc = pd.read_csv(f"{HERE}/chirality_gaussian_descriptors.csv")

def window(ch, well):
    ex = ex_map[ch]; spec = spectra[ex][well].to_numpy(float)
    emc = CHIRALITY[ch]["em"]
    sel = np.abs(em_axis - emc) <= WINDOW_NM
    return em_axis[sel], spec[sel], emc, ex

def refit(ch, well):
    _, spec, emc, _ = window(ch, well)
    return fit_chirality(em_axis, spectra[ex_map[ch]][well].to_numpy(float), emc)

# ---- (A) Is low R2 explained by low amplitude? per-tube correlation ----
print("(A) Within each tube: does R2 fall because the peak is DIM?")
print(f"{'tube':7} {'r(R2,amp)':>10} {'peak/noise@worst':>17}  interpretation")
TUBES_BAD = ["ch8_6","ch8_7","ch10_5","ch9_5","ch10_3"]
for ch in ["ch6_5","ch7_5","ch8_3"] + TUBES_BAD:
    a = desc[f"{ch}_gauss_max"]; r2 = desc[f"{ch}_gauss_r2"]
    ok = a.notna() & r2.notna()
    rr = np.corrcoef(np.log10(a[ok].clip(lower=1e-30)), r2[ok])[0,1]
    # worst-R2 well: signal range vs noise (std of window after detrend)
    worst = desc.loc[r2.idxmin(), "well"] if r2.notna().any() else None
    em,sp,emc,ex = window(ch, worst)
    pk2noise = (sp.max()-np.median(sp)) / (np.std(np.diff(sp))/np.sqrt(2) + 1e-30)
    print(f"{ch:7} {rr:+10.2f} {pk2noise:17.1f}   "
          f"{'dim->lowR2' if rr>0.4 else 'R2 low even when bright?'}")

# ---- (B) Visual: window + fit for good vs bad tubes, bright vs dim well ----
def bright_dim(ch):
    sub = desc[(desc.sensor=="GT15-STMN2")]
    b = sub.loc[sub[f"{ch}_gauss_max"].idxmax(), "well"]
    d = sub.loc[sub[f"{ch}_gauss_r2"].idxmin(), "well"]
    return b, d

panels = ["ch6_5","ch7_5","ch8_6","ch10_5"]
fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.0*len(panels)))
for i, ch in enumerate(panels):
    bwell, dwell = bright_dim(ch)
    for j, well in enumerate([bwell, dwell]):
        em, sp, emc, ex = window(ch, well)
        d = refit(ch, well)
        ax = axes[i, j]
        ax.plot(em, sp, "k.-", ms=3, lw=0.7, label="data")
        # reconstruct fitted curve in physical units
        if not np.isnan(d["gauss_em_center"]):
            scale = sp.max()
            # re-run fit to get popt-equivalent curve: use stored descriptors approx
            xx = np.linspace(em.min(), em.max(), 200)
            ax.axvline(d["gauss_em_center"], color="C0", ls=":", lw=1)
        ax.axvline(emc, color="C3", ls="--", lw=1, label="nominal em")
        ax.set_title(f"{ch} ex{ex} | {well} | R2={d['gauss_r2'] if not np.isnan(d['gauss_r2']) else float('nan'):.2f} "
                     f"amp={d['gauss_max']:.1e}", fontsize=8)
        ax.set_xlabel("emission nm"); ax.tick_params(labelsize=7)
        if j==0: ax.set_ylabel("intensity")
        ax.legend(fontsize=6)
fig.suptitle("Emission window + nominal(red)/fitted(blue) center — bright (L) vs worst-R2 (R)", y=1.0)
fig.tight_layout()
fig.savefig(f"{HERE}/diagnose_fits.png", dpi=130, bbox_inches="tight")
print("\nSaved diagnose_fits.png")

# ---- (C) what fraction of window variance is real signal vs noise, per tube ----
print("\n(C) Median (peak-baseline)/noise in fit window, GT15-STMN2 wells:")
for ch in ["ch6_5","ch7_5","ch8_3"] + TUBES_BAD:
    vals=[]
    for well in desc[desc.sensor=="GT15-STMN2"]["well"]:
        em,sp,emc,ex = window(ch, well)
        noise = np.std(np.diff(sp))/np.sqrt(2) + 1e-30
        vals.append((sp.max()-np.median(sp))/noise)
    print(f"  {ch:7} ex{ex_map[ch]}  median peak/noise = {np.median(vals):6.1f}")
