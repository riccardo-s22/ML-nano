"""
3D surface comparison of the mean 24h EEM: ALS vs CTRL.
 - Panel A: both group-mean surfaces overlaid on the same axes (semi-transparent).
 - Panel B: the ALS-CTRL difference surface (where they diverge).
Two viewing azimuths saved. Uses cached 24h EEM. Run under `als-prs` python.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "dim477_region_pilot_outputs", "pilot_cache.npz")
OUT = os.path.join(ROOT, "dim477_surface_compare_outputs"); os.makedirs(OUT, exist_ok=True)
EM_LO, EM_HI = 900.0, 1360.0        # crop emission to the informative window
ALS_C, CTRL_C = "#d1495b", "#0e8a8f"

z = np.load(CACHE, allow_pickle=True)
x24, y = z["x24"], z["y"]
ex_axis, em_axis = z["ex_axis"], z["em_axis"]
mrow = (em_axis >= EM_LO) & (em_axis <= EM_HI)
em = em_axis[mrow]
als = x24[y == 1][:, mrow, :].mean(0)      # (H',W)
ctrl = x24[y == 0][:, mrow, :].mean(0)
diff = als - ctrl
Xg, Yg = np.meshgrid(ex_axis, em)           # (H',W)
print(f"ALS {int((y==1).sum())} CTRL {int((y==0).sum())} | surface {als.shape} "
      f"em {em.min():.0f}-{em.max():.0f} ex {ex_axis.min():.0f}-{ex_axis.max():.0f}")

# chirality peaks within the crop (for base-plane markers)
chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    nm = e = x = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): nm = ln.strip("[]")
        elif ln.startswith("Emission"): e = float(ln.split("=")[1])
        elif ln.startswith("Excitation"):
            x = float(ln.split("=")[1])
            if EM_LO <= e <= EM_HI: chir.append((nm, x, e))

def draw(azim, tag):
    fig = plt.figure(figsize=(17, 7.2), dpi=160)

    # ---- Panel A: overlaid group means ----
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    axA.plot_surface(Xg, Yg, ctrl, color=CTRL_C, alpha=0.55, rcount=110, ccount=71,
                     linewidth=0, antialiased=True, shade=True)
    axA.plot_surface(Xg, Yg, als, color=ALS_C, alpha=0.55, rcount=110, ccount=71,
                     linewidth=0, antialiased=True, shade=True)
    zb = min(als.min(), ctrl.min())
    for nm, x, e in chir:
        axA.plot([x, x], [e, e], [zb, zb], marker="x", color="k", ms=5, mew=1.2)
    axA.set_xlabel("excitation (nm)", labelpad=8); axA.set_ylabel("emission (nm)", labelpad=8)
    axA.set_zlabel("norm. intensity", labelpad=4)
    axA.view_init(elev=32, azim=azim)
    axA.set_title("Mean 24h EEM: ALS vs CTRL (overlaid)", fontsize=12)
    axA.legend(handles=[Patch(color=CTRL_C, label=f"CTRL (n={int((y==0).sum())})"),
                        Patch(color=ALS_C, label=f"ALS (n={int((y==1).sum())})")],
               loc="upper left", fontsize=10)

    # ---- Panel B: difference surface ----
    axB = fig.add_subplot(1, 2, 2, projection="3d")
    dv = np.abs(diff).max()
    norm = TwoSlopeNorm(vmin=-dv, vcenter=0, vmax=dv)
    fc = cm.seismic(norm(diff))
    surf = axB.plot_surface(Xg, Yg, diff, facecolors=fc, rcount=110, ccount=71,
                            linewidth=0, antialiased=True, shade=False)
    axB.contourf(Xg, Yg, diff, zdir="z", offset=diff.min(), cmap="seismic", norm=norm, alpha=0.6)
    axB.set_xlabel("excitation (nm)", labelpad=8); axB.set_ylabel("emission (nm)", labelpad=8)
    axB.set_zlabel("ALS - CTRL", labelpad=4)
    axB.view_init(elev=32, azim=azim)
    axB.set_title("Difference surface (ALS - CTRL)\nblue = ALS lower", fontsize=12)
    m = cm.ScalarMappable(cmap="seismic", norm=norm); m.set_array([])
    fig.colorbar(m, ax=axB, fraction=0.03, pad=0.08, label="ALS - CTRL intensity")

    fig.suptitle("Group-mean excitation-emission surfaces (24h)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, f"surface_compare_{tag}.png")
    fig.savefig(fp); plt.close(fig); print("saved", fp)

draw(azim=-60, tag="view1")
draw(azim=-125, tag="view2")

# also a single large overlay for a clean headline image
fig = plt.figure(figsize=(11, 9), dpi=160)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xg, Yg, ctrl, color=CTRL_C, alpha=0.5, rcount=130, ccount=71, linewidth=0, antialiased=True)
ax.plot_surface(Xg, Yg, als, color=ALS_C, alpha=0.5, rcount=130, ccount=71, linewidth=0, antialiased=True)
ax.set_xlabel("excitation (nm)", labelpad=10); ax.set_ylabel("emission (nm)", labelpad=10)
ax.set_zlabel("norm. intensity", labelpad=6); ax.view_init(elev=30, azim=-60)
ax.legend(handles=[Patch(color=CTRL_C, label=f"CTRL (n={int((y==0).sum())})"),
                   Patch(color=ALS_C, label=f"ALS (n={int((y==1).sum())})")], fontsize=11)
ax.set_title("Mean 24h EEM surface — ALS vs CTRL", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "surface_overlay_big.png")); plt.close(fig)
print("saved overlay_big")
