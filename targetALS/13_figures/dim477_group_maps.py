"""
Group-average dim477 spectral maps: ALS vs CTRL.

Per subject, the dim477 spatial contribution map is the first-order per-pixel
contribution to z_agg[477]:
    C_i(pixel) = sum_t  [ sign * d z477 / d x_{i,t}(pixel) ] * x_{i,t}(pixel)
(the fold3 signed saliency times the subject's own EEM, summed over the 3
timepoints; attention weighting is already inside the gradient). We average C_i
across CTRL and across ALS and show the difference to localize the group signal.
Run under the Windows `als-prs` conda python (WSL has no torch).
"""
import os, importlib.util
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
PLS  = os.path.join(os.path.dirname(ROOT), "PLS")
OUT  = os.path.join(ROOT, "dim477_group_maps_outputs"); os.makedirs(OUT, exist_ok=True)
TPD  = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
DIM, DEV = 477, torch.device("cpu")
TP_NAMES = ["0h", "6h", "24h"]

spec = importlib.util.spec_from_file_location("sal", os.path.join(PLS, "dim35_chirality_saliency.py"))
sal = importlib.util.module_from_spec(spec); spec.loader.exec_module(sal)

lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
codes = lab["code"].astype(str).tolist()
groups = dict(zip(lab["code"].astype(str), lab["group"].astype(str)))
y = np.array([1 if groups[c].strip().upper() == "ALS" else 0 for c in codes])   # ALS=1
em_axis, ex_axis = sal.get_wavelength_axes(TPD[0], codes[0])
ds = sal.ExcelImageDataset(codes, [0]*len(codes), TPD)
X = torch.stack([ds[i][0] for i in range(len(codes))], 0).float()   # (N,T,1,H,W)
N, Tn, _, H, Wd = X.shape
print(f"X {tuple(X.shape)} | ALS {y.sum()} CTRL {(1-y).sum()}")

def build_model(path):
    m = sal.ConvAutoencoderWithAttention(in_channels=1, latent_dim=512, num_classes=2).to(DEV)
    with torch.no_grad(): _ = m(X[:1].to(DEV))
    m.load_state_dict(torch.load(path, map_location=DEV), strict=False); m.eval(); return m
def zagg_dim(m, dim):
    out = []
    with torch.no_grad():
        for i in range(N):
            lat = [m.encoder(X[i:i+1, t].to(DEV)) for t in range(Tn)]
            z, _ = m.attention(torch.stack(lat, 1)); out.append(z[0, dim].item())
    return np.array(out)

# fold reproducing canonical dim477
canon = pd.read_csv(os.path.join(PLS, "z_agg3.csv"))[["code", "dim477"]]
canon = dict(zip(canon["code"], canon["dim477"]))
cvec = np.array([canon.get(c, np.nan) for c in codes]); ok = ~np.isnan(cvec)
best = None
for k in range(1, 6):
    m = build_model(os.path.join(ROOT, f"5_fold_models_original/best_model_fold{k}.pth"))
    r = pearsonr(zagg_dim(m, DIM)[ok], cvec[ok])[0]
    if best is None or abs(r) > abs(best[1]): best = (k, r, m)
fold, r, model = best
sign = 1.0 if r >= 0 else -1.0
print(f"==> fold{fold} (corr {r:+.3f})")

# per-subject dim477 contribution maps: C_t = signed_saliency_t * x_t  (T,H,W); aggregate over t
Cagg = np.zeros((N, H, Wd), np.float32)          # summed over timepoints
Ctp  = np.zeros((N, Tn, H, Wd), np.float32)       # per timepoint
for i in range(N):
    x = X[i:i+1].clone().to(DEV).requires_grad_(True)
    lat = [model.encoder(x[:, t]) for t in range(Tn)]
    z, _ = model.attention(torch.stack(lat, 1))
    (z[0, DIM] * sign).backward()
    g = x.grad.detach()[0, :, 0].numpy()          # (T,H,W) signed saliency
    xe = X[i, :, 0].numpy()                        # (T,H,W) EEM
    c = g * xe                                     # (T,H,W) per-pixel contribution
    Ctp[i] = c; Cagg[i] = c.sum(0)
print("contribution maps computed")

ctrl_agg, als_agg = Cagg[y == 0].mean(0), Cagg[y == 1].mean(0)
diff_agg = als_agg - ctrl_agg
ctrl_tp = Ctp[y == 0].mean(0); als_tp = Ctp[y == 1].mean(0)
diff_tp = als_tp - ctrl_tp                          # (T,H,W)

np.savez(os.path.join(OUT, "group_maps.npz"), ctrl_agg=ctrl_agg, als_agg=als_agg,
         diff_agg=diff_agg, diff_tp=diff_tp, ex_axis=ex_axis, em_axis=em_axis)

# chirality reference peaks
chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    nm = em = ex = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): nm = ln.strip("[]")
        elif ln.startswith("Emission"): em = float(ln.split("=")[1])
        elif ln.startswith("Excitation"): ex = float(ln.split("=")[1]); chir.append((nm, ex, em))
extent = [ex_axis.min(), ex_axis.max(), em_axis.min(), em_axis.max()]
def overlay(ax):
    for nm, ex, em in chir:
        ax.plot(ex, em, "kx", ms=6, mew=1.3); ax.text(ex+3, em, f"({nm})", fontsize=6.5, va="center")

# ---------- Figure: row1 CTRL / ALS / diff ; row2 per-timepoint diff ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 9.4), dpi=160)
gv = np.abs(np.concatenate([ctrl_agg.ravel(), als_agg.ravel()])).max()   # shared scale CTRL/ALS
dv = np.abs(diff_agg).max()
for ax, M, ttl, vmax in [(axes[0,0], ctrl_agg, f"CTRL mean dim477 map (n={int((y==0).sum())})", gv),
                          (axes[0,1], als_agg,  f"ALS mean dim477 map (n={int((y==1).sum())})", gv),
                          (axes[0,2], diff_agg, "ALS - CTRL (difference)", dv)]:
    im = ax.imshow(M, aspect="auto", origin="lower", extent=extent, cmap="seismic", vmin=-vmax, vmax=vmax)
    overlay(ax); ax.set_xlabel("excitation (nm)"); ax.set_ylabel("emission (nm)"); ax.set_title(ttl, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
dvt = np.abs(diff_tp).max()
for t in range(Tn):
    ax = axes[1, t]
    im = ax.imshow(diff_tp[t], aspect="auto", origin="lower", extent=extent, cmap="seismic", vmin=-dvt, vmax=dvt)
    overlay(ax); ax.set_xlabel("excitation (nm)"); ax.set_ylabel("emission (nm)")
    ax.set_title(f"ALS - CTRL  @ {TP_NAMES[t]}", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle(f"Group-average dim477 spectral contribution maps (fold{fold})   "
             f"red = pushes dim477 up (control-ward), blue = ALS-ward", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "dim477_group_maps.png")); plt.close(fig)
print("saved", os.path.join(OUT, "dim477_group_maps.png"))
