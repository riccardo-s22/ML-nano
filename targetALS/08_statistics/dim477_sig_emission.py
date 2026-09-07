"""
(1) Find where the ALS vs CTRL difference in the raw normalized 24h EEM is
    significant. Per-pixel testing on 36k pixels is underpowered at n~20, so the
    PRIMARY test is region-level: mean intensity in each SWCNT chirality window
    per subject -> Welch t-test -> Benjamini-Hochberg FDR over the 12 chiralities.
    A spatially-smoothed per-pixel -log10 p map is shown for context.
(2) For the excitations carrying significant (or, if none survive, the most-
    different) chirality regions, plot the ACTUAL emission spectra (mean ALS vs
    mean CTRL, intensity vs emission nm, mean +/- SEM), shading the region band.

Uses cached 24h EEM (dim477_region_pilot_outputs/pilot_cache.npz). Run under the
Windows `als-prs` conda python.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "dim477_region_pilot_outputs", "pilot_cache.npz")
OUT = os.path.join(ROOT, "dim477_sig_emission_outputs"); os.makedirs(OUT, exist_ok=True)
Q = 0.05
EX_HALF, EM_HALF = 12.0, 30.0     # chirality window half-width (nm) in ex / em
SMOOTH = (4.0, 1.5)               # gaussian sigma (em_px, ex_px) for the context p-map

z = np.load(CACHE, allow_pickle=True)
x24, y = z["x24"], z["y"]                       # (N,H,W) raw normalized 24h EEM ; ALS=1
ex_axis, em_axis = z["ex_axis"], z["em_axis"]
N, H, W = x24.shape
als, ctrl = x24[y == 1], x24[y == 0]
print(f"x24 {x24.shape} | ALS {len(als)} CTRL {len(ctrl)}")

# chirality peaks
chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    nm = em = ex = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): nm = ln.strip("[]")
        elif ln.startswith("Emission"): em = float(ln.split("=")[1])
        elif ln.startswith("Excitation"): ex = float(ln.split("=")[1]); chir.append((nm, ex, em))

def win(ex0, em0):
    cc = np.where(np.abs(ex_axis - ex0) <= EX_HALF)[0]
    rr = np.where(np.abs(em_axis - em0) <= EM_HALF)[0]
    return rr, cc

# ---------- (1a) region-level test per chirality ----------
rows = []
for nm, ex0, em0 in chir:
    rr, cc = win(ex0, em0)
    if len(rr) == 0 or len(cc) == 0: continue
    a = als[:, rr][:, :, cc].mean(axis=(1, 2))
    c = ctrl[:, rr][:, :, cc].mean(axis=(1, 2))
    t, p = stats.ttest_ind(a, c, equal_var=False)
    rows.append(dict(chirality=f"({nm})", ex_nm=round(ex0, 0), em_nm=round(em0, 0),
                     als_mean=round(a.mean(), 4), ctrl_mean=round(c.mean(), 4),
                     diff=round(a.mean() - c.mean(), 4), t=round(float(t), 2), p=float(p)))
reg = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
# BH-FDR over the chirality tests
mreg = len(reg); rp = reg["p"].values
bh = rp <= (np.arange(1, mreg + 1) / mreg) * Q
kmax = np.where(bh)[0].max() + 1 if bh.any() else 0
reg["p_fdr_sig"] = False
if kmax: reg.loc[:kmax-1, "p_fdr_sig"] = True
reg["p"] = reg["p"].round(4)
reg.to_csv(os.path.join(OUT, "chirality_region_tests.csv"), index=False)
print(f"\n=== chirality region tests (Welch t, BH-FDR q<{Q}) ===\n" + reg.to_string(index=False))
n_sig = int(reg["p_fdr_sig"].sum())
print(f"\nFDR-significant chiralities: {n_sig}")

# selection for emission plots: FDR-sig if any, else the 6 smallest-p regions
sel = reg[reg["p_fdr_sig"]] if n_sig else reg.head(6)
sel_status = "FDR-significant" if n_sig else "most-different (none survive FDR)"

# ---------- (1b) smoothed per-pixel context map ----------
als_s = np.stack([ndimage.gaussian_filter(im, SMOOTH) for im in als])
ctrl_s = np.stack([ndimage.gaussian_filter(im, SMOOTH) for im in ctrl])
t_s, p_s = stats.ttest_ind(als_s, ctrl_s, axis=0, equal_var=False)
mean_diff = als.mean(0) - ctrl.mean(0)
lp = -np.log10(np.where(np.isnan(p_s), 1, p_s))

extent = [ex_axis.min(), ex_axis.max(), em_axis.min(), em_axis.max()]
def peaks(ax):
    for nm, ex, em in chir: ax.plot(ex, em, "kx", ms=6, mew=1.3)

# ---------- Fig 1: difference map + smoothed significance context ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6), dpi=160)
dv = np.nanmax(np.abs(mean_diff))
im = a1.imshow(mean_diff, aspect="auto", origin="lower", extent=extent, cmap="seismic", vmin=-dv, vmax=dv)
a1.set_title("ALS - CTRL mean 24h EEM", fontsize=11)
plt.colorbar(im, ax=a1, fraction=0.046, pad=0.04, label="ALS - CTRL intensity")
im2 = a2.imshow(lp, aspect="auto", origin="lower", extent=extent, cmap="magma",
                vmin=0, vmax=max(2.0, np.nanpercentile(lp, 99.5)))
a2.contour(lp, levels=[-np.log10(0.05)], colors="lime", linewidths=1.0, extent=extent, origin="lower")
a2.set_title(f"-log10 p (smoothed pixels; green = p<0.05)", fontsize=11)
plt.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)
for ax in (a1, a2):
    peaks(ax)
    for _, s in sel.iterrows(): ax.axvline(s.ex_nm, color="cyan", lw=0.9, ls=":")
    ax.set_xlabel("excitation (nm)"); ax.set_ylabel("emission (nm)")
fig.suptitle(f"Group EEM difference & significance   (emission plots at: {sel_status})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(OUT, "fig1_significance_map.png")); plt.close(fig)

# ---------- Fig 2: actual emission spectra at selected excitations ----------
selist = list(sel.itertuples())
ncol = 3; nrow = int(np.ceil(len(selist) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2*ncol, 3.6*nrow), dpi=160, squeeze=False)
for ax in axes.ravel(): ax.axis("off")
ALS_C, CTRL_C = "crimson", "#0e8a8f"
for k, s in enumerate(selist):
    ax = axes[k // ncol][k % ncol]; ax.axis("on")
    c = int(np.abs(ex_axis - s.ex_nm).argmin())
    a_m, a_e = als[:, :, c].mean(0), als[:, :, c].std(0) / np.sqrt(len(als))
    c_m, c_e = ctrl[:, :, c].mean(0), ctrl[:, :, c].std(0) / np.sqrt(len(ctrl))
    ax.fill_between(em_axis, c_m - c_e, c_m + c_e, color=CTRL_C, alpha=0.2)
    ax.fill_between(em_axis, a_m - a_e, a_m + a_e, color=ALS_C, alpha=0.2)
    ax.plot(em_axis, c_m, color=CTRL_C, lw=1.6, label=f"CTRL (n={len(ctrl)})")
    ax.plot(em_axis, a_m, color=ALS_C, lw=1.6, label=f"ALS (n={len(als)})")
    ax.axvspan(s.em_nm - EM_HALF, s.em_nm + EM_HALF, color="gold", alpha=0.25, zorder=0)
    star = "*" if getattr(s, "p_fdr_sig", False) else ""
    ax.set_title(f"ex {s.ex_nm:.0f} nm  {s.chirality}{star}  (p={s.p:.3f})", fontsize=10)
    ax.set_xlabel("emission (nm)"); ax.set_ylabel("norm. intensity")
    if k == 0: ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
fig.suptitle(f"Actual 24h emission spectra at {sel_status} chirality excitations "
             "(gold = chirality emission window; mean +/- SEM)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(OUT, "fig2_emission_spectra.png")); plt.close(fig)

# ---------- Fig 3: ALS-CTRL difference emission curves ----------
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
for s in selist:
    c = int(np.abs(ex_axis - s.ex_nm).argmin())
    ax.plot(em_axis, mean_diff[:, c], lw=1.6, label=f"{s.ex_nm:.0f} nm {s.chirality}")
ax.axhline(0, color="k", lw=0.7)
ax.set_xlabel("emission (nm)"); ax.set_ylabel("ALS - CTRL intensity")
ax.set_title("Emission-difference spectra (ALS - CTRL) at selected chirality excitations", fontsize=11)
ax.legend(fontsize=8, title="excitation"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_difference_curves.png")); plt.close(fig)
print("\nsaved figs 1-3 + chirality_region_tests.csv to", OUT)
