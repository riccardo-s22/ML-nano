"""
Edge-cropped re-render of the ALS vs CTRL 24h EEM surfaces.
Excitation cropped to [520, 820] nm (drops the ~840-850 nm boundary artifact;
keeps all chirality excitations 577-801). Emission [900, 1360].
Outputs: cropped drape, static overlay, rotating GIF. Run under `als-prs` python.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "dim477_region_pilot_outputs", "pilot_cache.npz")
OUT = os.path.join(ROOT, "dim477_surface_compare_outputs"); os.makedirs(OUT, exist_ok=True)
EX_LO, EX_HI = 520.0, 820.0        # excitation crop (removes edge artifact)
EM_LO, EM_HI = 900.0, 1360.0
ALS_C, CTRL_C = "#d1495b", "#0e8a8f"

z = np.load(CACHE, allow_pickle=True)
x24, y = z["x24"], z["y"]
ex_axis, em_axis = z["ex_axis"], z["em_axis"]
mcol = (ex_axis >= EX_LO) & (ex_axis <= EX_HI)
mrow = (em_axis >= EM_LO) & (em_axis <= EM_HI)
ex, em = ex_axis[mcol], em_axis[mrow]
sub = x24[:, mrow][:, :, mcol]
als = sub[y == 1].mean(0); ctrl = sub[y == 0].mean(0); grand = sub.mean(0)
diff = als - ctrl
Xg, Yg = np.meshgrid(ex, em)
nA, nC = int((y == 1).sum()), int((y == 0).sum())
print(f"cropped surface {als.shape} | ex {ex.min():.0f}-{ex.max():.0f} em {em.min():.0f}-{em.max():.0f} "
      f"| max|diff| {np.abs(diff).max():.4f}")

chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    nm = e = xx = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): nm = ln.strip("[]")
        elif ln.startswith("Emission"): e = float(ln.split("=")[1])
        elif ln.startswith("Excitation"):
            xx = float(ln.split("=")[1])
            if EX_LO <= xx <= EX_HI and EM_LO <= e <= EM_HI: chir.append((nm, xx, e))

# ---- drape: grand-mean surface colored by ALS-CTRL ----
dv = np.abs(diff).max()
norm = TwoSlopeNorm(vmin=-dv, vcenter=0, vmax=dv)
fig = plt.figure(figsize=(11.5, 9), dpi=160)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xg, Yg, grand, facecolors=cm.seismic(norm(diff)), rcount=grand.shape[0],
                ccount=grand.shape[1], linewidth=0, antialiased=True, shade=False)
zb = grand.min()
for nm, xx, e in chir:
    ax.plot([xx, xx], [e, e], [zb, zb], marker="x", color="k", ms=5, mew=1.2)
    ax.text(xx, e, zb, f"({nm})", fontsize=6.5)
ax.set_xlabel("excitation (nm)", labelpad=10); ax.set_ylabel("emission (nm)", labelpad=10)
ax.set_zlabel("grand-mean norm. intensity", labelpad=6)
ax.view_init(elev=34, azim=-62)
ax.set_title("Grand-mean 24h EEM (edge-cropped), colored by ALS - CTRL\n"
             "red = ALS higher, blue = ALS lower", fontsize=12)
m = cm.ScalarMappable(cmap="seismic", norm=norm); m.set_array([])
fig.colorbar(m, ax=ax, fraction=0.03, pad=0.1, label="ALS - CTRL intensity")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "surface_drape_diff_cropped.png")); plt.close(fig)
print("saved surface_drape_diff_cropped.png")

# ---- static overlay ----
fig = plt.figure(figsize=(11, 9), dpi=160)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xg, Yg, ctrl, color=CTRL_C, alpha=0.5, rcount=grand.shape[0], ccount=grand.shape[1],
                linewidth=0, antialiased=True)
ax.plot_surface(Xg, Yg, als, color=ALS_C, alpha=0.5, rcount=grand.shape[0], ccount=grand.shape[1],
                linewidth=0, antialiased=True)
ax.set_xlabel("excitation (nm)", labelpad=10); ax.set_ylabel("emission (nm)", labelpad=10)
ax.set_zlabel("norm. intensity", labelpad=6); ax.view_init(elev=30, azim=-60)
ax.legend(handles=[Patch(color=CTRL_C, label=f"CTRL (n={nC})"),
                   Patch(color=ALS_C, label=f"ALS (n={nA})")], fontsize=11)
ax.set_title("Mean 24h EEM surface (edge-cropped) — ALS vs CTRL", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "surface_overlay_big_cropped.png")); plt.close(fig)
print("saved surface_overlay_big_cropped.png")

# ---- rotating GIF (draw once, rotate camera) ----
figg = plt.figure(figsize=(7.6, 6.8), dpi=82)
axg = figg.add_subplot(111, projection="3d")
axg.plot_surface(Xg, Yg, ctrl, color=CTRL_C, alpha=0.5, rcount=grand.shape[0], ccount=grand.shape[1],
                 linewidth=0, antialiased=True)
axg.plot_surface(Xg, Yg, als, color=ALS_C, alpha=0.5, rcount=grand.shape[0], ccount=grand.shape[1],
                 linewidth=0, antialiased=True)
axg.set_xlabel("excitation (nm)", labelpad=8); axg.set_ylabel("emission (nm)", labelpad=8)
axg.set_zlabel("norm. intensity", labelpad=4)
axg.legend(handles=[Patch(color=CTRL_C, label=f"CTRL (n={nC})"),
                    Patch(color=ALS_C, label=f"ALS (n={nA})")], loc="upper left", fontsize=10)
axg.set_title("Mean 24h EEM surface (edge-cropped) — ALS vs CTRL", fontsize=12)
anim = FuncAnimation(figg, lambda a: axg.view_init(elev=28, azim=a) or [],
                     frames=list(range(-70, 290, 4)), interval=60, blit=False)
gif = os.path.join(OUT, "surface_overlay_rotating_cropped.gif")
anim.save(gif, writer=PillowWriter(fps=18)); plt.close(figg)
print("saved", gif)
