"""Regenerate only the edge-cropped rotating GIF, with downsampled surfaces
(the full-res version exhausted rendering memory). Run under `als-prs` python."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "dim477_region_pilot_outputs", "pilot_cache.npz")
OUT = os.path.join(ROOT, "dim477_surface_compare_outputs")
EX_LO, EX_HI, EM_LO, EM_HI = 520.0, 820.0, 900.0, 1360.0
ALS_C, CTRL_C = "#d1495b", "#0e8a8f"

z = np.load(CACHE, allow_pickle=True)
x24, y = z["x24"], z["y"]; ex_axis, em_axis = z["ex_axis"], z["em_axis"]
mcol = (ex_axis >= EX_LO) & (ex_axis <= EX_HI); mrow = (em_axis >= EM_LO) & (em_axis <= EM_HI)
ex, em = ex_axis[mcol], em_axis[mrow]
sub = x24[:, mrow][:, :, mcol]
als = sub[y == 1].mean(0); ctrl = sub[y == 0].mean(0)
Xg, Yg = np.meshgrid(ex, em)
nA, nC = int((y == 1).sum()), int((y == 0).sum())

figg = plt.figure(figsize=(7.6, 6.8), dpi=82)
axg = figg.add_subplot(111, projection="3d")
axg.plot_surface(Xg, Yg, ctrl, color=CTRL_C, alpha=0.5, rcount=95, ccount=61, linewidth=0, antialiased=True)
axg.plot_surface(Xg, Yg, als, color=ALS_C, alpha=0.5, rcount=95, ccount=61, linewidth=0, antialiased=True)
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
