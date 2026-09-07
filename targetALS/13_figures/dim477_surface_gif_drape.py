"""
(1) Rotating GIF of the overlaid ALS vs CTRL mean 24h EEM surfaces.
(2) Drape: single grand-mean EEM surface colored by the ALS-CTRL difference.
Uses cached 24h EEM. Run under `als-prs` python.
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
EM_LO, EM_HI = 900.0, 1360.0
ALS_C, CTRL_C = "#d1495b", "#0e8a8f"

z = np.load(CACHE, allow_pickle=True)
x24, y = z["x24"], z["y"]
ex_axis, em_axis = z["ex_axis"], z["em_axis"]
mrow = (em_axis >= EM_LO) & (em_axis <= EM_HI)
em = em_axis[mrow]
als = x24[y == 1][:, mrow, :].mean(0)
ctrl = x24[y == 0][:, mrow, :].mean(0)
grand = x24[:, mrow, :].mean(0)          # mean over all 39
diff = als - ctrl
Xg, Yg = np.meshgrid(ex_axis, em)
nA, nC = int((y == 1).sum()), int((y == 0).sum())
print(f"surfaces {als.shape} | ALS {nA} CTRL {nC}")

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

# ---------- (2) drape: grand-mean surface colored by ALS-CTRL difference ----------
dv = np.abs(diff).max()
norm = TwoSlopeNorm(vmin=-dv, vcenter=0, vmax=dv)
fig = plt.figure(figsize=(11.5, 9), dpi=160)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xg, Yg, grand, facecolors=cm.seismic(norm(diff)), rcount=140, ccount=71,
                linewidth=0, antialiased=True, shade=False)
zb = grand.min()
for nm, x, e in chir:
    ax.plot([x, x], [e, e], [zb, zb], marker="x", color="k", ms=5, mew=1.2)
    ax.text(x, e, zb, f"({nm})", fontsize=6.5)
ax.set_xlabel("excitation (nm)", labelpad=10); ax.set_ylabel("emission (nm)", labelpad=10)
ax.set_zlabel("grand-mean norm. intensity", labelpad=6)
ax.view_init(elev=34, azim=-62)
ax.set_title("Grand-mean 24h EEM surface, colored by ALS - CTRL\n"
             "(height = mean signal; red = ALS higher, blue = ALS lower)", fontsize=12)
m = cm.ScalarMappable(cmap="seismic", norm=norm); m.set_array([])
fig.colorbar(m, ax=ax, fraction=0.03, pad=0.1, label="ALS - CTRL intensity")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "surface_drape_diff.png")); plt.close(fig)
print("saved surface_drape_diff.png")

# ---------- (1) rotating GIF of the overlay (draw once, rotate camera) ----------
figg = plt.figure(figsize=(7.6, 6.8), dpi=82)
axg = figg.add_subplot(111, projection="3d")
axg.plot_surface(Xg, Yg, ctrl, color=CTRL_C, alpha=0.5, rcount=95, ccount=71, linewidth=0, antialiased=True)
axg.plot_surface(Xg, Yg, als, color=ALS_C, alpha=0.5, rcount=95, ccount=71, linewidth=0, antialiased=True)
axg.set_xlabel("excitation (nm)", labelpad=8); axg.set_ylabel("emission (nm)", labelpad=8)
axg.set_zlabel("norm. intensity", labelpad=4)
axg.legend(handles=[Patch(color=CTRL_C, label=f"CTRL (n={nC})"),
                    Patch(color=ALS_C, label=f"ALS (n={nA})")], loc="upper left", fontsize=10)
axg.set_title("Mean 24h EEM surface — ALS vs CTRL", fontsize=12)

def update(az):
    axg.view_init(elev=28, azim=az)
    return []

frames = list(range(-70, 290, 4))     # 90 frames, full turn
anim = FuncAnimation(figg, update, frames=frames, interval=60, blit=False)
gif = os.path.join(OUT, "surface_overlay_rotating.gif")
anim.save(gif, writer=PillowWriter(fps=18))
plt.close(figg)
print("saved", gif)
