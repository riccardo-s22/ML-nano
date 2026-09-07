"""
Diagnose the sharp red (ALS>CTRL) ridges in the difference surface:
 (1) do they follow a scatter line (Rayleigh em=2*ex / em=ex, Raman)?
 (2) are they an artifact of the MEAN (outlier-driven)?  -> compare median.
 (3) are they a rendering downsample artifact?            -> full-res 2D maps.
Also lists the top positive-difference pixels with their em/ex ratio and whether
a single subject dominates. Run under `als-prs` python.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "dim477_region_pilot_outputs", "pilot_cache.npz")
OUT = os.path.join(ROOT, "dim477_surface_compare_outputs"); os.makedirs(OUT, exist_ok=True)

z = np.load(CACHE, allow_pickle=True)
x24, y = z["x24"], z["y"]
ex_axis, em_axis = z["ex_axis"], z["em_axis"]   # (71,), (512,)
N, H, W = x24.shape
als, ctrl = x24[y == 1], x24[y == 0]
diff_mean = als.mean(0) - ctrl.mean(0)
diff_med = np.median(als, 0) - np.median(ctrl, 0)
sd_all = x24.std(0)

# chirality peaks
chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    nm = e = x = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): nm = ln.strip("[]")
        elif ln.startswith("Emission"): e = float(ln.split("=")[1])
        elif ln.startswith("Excitation"): x = float(ln.split("=")[1]); chir.append((nm, x, e))

# ---- (1)+(2) top positive-diff pixels: location, scatter ratio, outlier check ----
flat = diff_mean.ravel()
top = np.argsort(-flat)[:15]
rows = []
for idx in top:
    r, c = np.unravel_index(idx, diff_mean.shape)
    em_nm, ex_nm = em_axis[r], ex_axis[c]
    # which subjects drive it: how much does removing the single most extreme ALS subject change it?
    col_als = als[:, r, c]
    lead = col_als.max() - np.median(col_als)      # gap of top ALS value above median
    dmin = min(abs(em_nm - e) + abs(ex_nm - x)*2 for _, x, e in chir)  # rough nm distance to any chirality
    rows.append(dict(ex_nm=round(float(ex_nm), 0), em_nm=round(float(em_nm), 0),
                     em_over_ex=round(float(em_nm / ex_nm), 2),
                     diff_mean=round(float(diff_mean[r, c]), 4),
                     diff_median=round(float(diff_med[r, c]), 4),
                     top_ALS_minus_median=round(float(lead), 3),
                     nm_to_chirality=round(float(dmin), 0)))
tbl = pd.DataFrame(rows)
tbl.to_csv(os.path.join(OUT, "ridge_top_pixels.csv"), index=False)
print("=== top-15 positive (ALS>CTRL) difference pixels ===")
print(tbl.to_string(index=False))
print("\nRayleigh 2nd-order scatter would give em/ex = 2.00; 1st-order em/ex = 1.00")
print(f"median-diff at these pixels vs mean-diff: "
      f"median {'MOSTLY VANISHES' if np.mean(np.abs(tbl.diff_median)) < 0.4*np.mean(np.abs(tbl.diff_mean)) else 'persists'} "
      f"(mean|meandiff|={np.mean(np.abs(tbl.diff_mean)):.3f}, mean|mediandiff|={np.mean(np.abs(tbl.diff_median)):.3f})")

# ---- figure: mean vs median diff (full-res) + SD map, with scatter lines ----
extent = [ex_axis.min(), ex_axis.max(), em_axis.min(), em_axis.max()]
exg = np.linspace(ex_axis.min(), ex_axis.max(), 200)
def overlay(ax):
    ax.plot(exg, 2*exg, "g-", lw=1.3, label="Rayleigh 2nd order (em=2*ex)")
    ax.plot(exg, exg, "g--", lw=1.0, label="Rayleigh 1st order (em=ex)")
    ax.plot(exg, 2*exg - 2*exg**2*1e-3*0, "none")  # placeholder (no-op)
    for nm, x, e in chir: ax.plot(x, e, "kx", ms=6, mew=1.3)
    ax.set_xlim(ex_axis.min(), ex_axis.max()); ax.set_ylim(em_axis.min(), em_axis.max())
    ax.set_xlabel("excitation (nm)"); ax.set_ylabel("emission (nm)")

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), dpi=160)
dv = np.nanmax(np.abs(diff_mean))
im0 = axes[0].imshow(diff_mean, aspect="auto", origin="lower", extent=extent, cmap="seismic", vmin=-dv, vmax=dv)
axes[0].set_title("MEAN diff (ALS-CTRL)"); overlay(axes[0]); axes[0].legend(fontsize=7, loc="upper left")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
im1 = axes[1].imshow(diff_med, aspect="auto", origin="lower", extent=extent, cmap="seismic", vmin=-dv, vmax=dv)
axes[1].set_title("MEDIAN diff (robust to outliers)"); overlay(axes[1])
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
im2 = axes[2].imshow(sd_all, aspect="auto", origin="lower", extent=extent, cmap="magma",
                     vmax=np.nanpercentile(sd_all, 99))
axes[2].set_title("between-subject SD (all 39)"); overlay(axes[2])
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
# mark the top positive pixels
for _, rr in tbl.iterrows():
    axes[0].plot(rr.ex_nm, rr.em_nm, "o", mfc="none", mec="lime", ms=9, mew=1.4)
fig.suptitle("Are the sharp ALS>CTRL ridges real chirality signal, scatter, or outliers?", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, "ridge_diagnosis.png")); plt.close(fig)
print("\nsaved ridge_diagnosis.png + ridge_top_pixels.csv")
