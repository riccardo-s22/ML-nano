"""
Trace dim477 back to EEM features for the two dim477-misclassified controls
(10.3d, 37.3d).

Reuses model + loaders + saliency from PLS/dim35_chirality_saliency.py.
Steps:
  1. Load each fold model, compute z_agg[477] for all 39 subjects, pick the fold
     whose dim477 best matches PLS/z_agg3.csv (the canonical dim477).
  2. With that model, compute per-sample gradient saliency |d z_agg[477]/d x_t|
     over the EEM (excitation x emission) for the 2 target controls.
  3. Plot: their EEM, their saliency map, and saliency x (sample - control mean)
     to localize which EEM region pushes their dim477 ALS-ward.
"""
import os, sys, importlib.util
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
PLS = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/PLS"
TPD = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT = os.path.join(ROOT, "dim477_nfl_outputs"); os.makedirs(OUT, exist_ok=True)
DIM = 477
TARGETS = ["10.3d__570.opj_with_emission", "37.3d__570.opj_with_emission"]
DEV = torch.device("cpu")

spec = importlib.util.spec_from_file_location("sal", os.path.join(PLS, "dim35_chirality_saliency.py"))
sal = importlib.util.module_from_spec(spec); spec.loader.exec_module(sal)

lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
codes = lab["code"].astype(str).tolist()
groups = dict(zip(lab["code"].astype(str), lab["group"].astype(str)))
y = np.array([1 if groups[c].strip().upper() == "ALS" else 0 for c in codes])

# EEM axes
em_axis, ex_axis = sal.get_wavelength_axes(TPD[0], codes[0])

# load all EEM stacks once: [N, T, 1, H, W]
ds = sal.ExcelImageDataset(codes, [0]*len(codes), TPD)
X = torch.stack([ds[i][0] for i in range(len(codes))], 0).float()  # (N,T,1,H,W)
print("X:", tuple(X.shape), "| em", em_axis.shape, "ex", ex_axis.shape)


def build_model(path):
    m = sal.ConvAutoencoderWithAttention(in_channels=1, latent_dim=512, num_classes=2).to(DEV)
    with torch.no_grad():
        _ = m(X[:1].to(DEV))            # create lazy layers
    sd = torch.load(path, map_location=DEV)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def zagg_dim(m, dim):
    out = []
    with torch.no_grad():
        for i in range(len(codes)):
            lat = [m.encoder(X[i:i+1, t].to(DEV)) for t in range(X.shape[1])]
            z, _ = m.attention(torch.stack(lat, 1))
            out.append(z[0, dim].item())
    return np.array(out)


# ---- 1) pick fold reproducing canonical dim477 ----
canon = pd.read_csv(os.path.join(PLS, "z_agg3.csv"))[["code", "dim477"]]
canon = dict(zip(canon["code"], canon["dim477"]))
cvec = np.array([canon.get(c, np.nan) for c in codes])
best = None
for k in range(1, 6):
    m = build_model(os.path.join(ROOT, f"5_fold_models_original/best_model_fold{k}.pth"))
    zz = zagg_dim(m, DIM)
    ok = ~np.isnan(cvec)
    r = pearsonr(zz[ok], cvec[ok])[0]
    print(f"fold{k}: corr(z_agg[477], canonical dim477) = {r:+.3f}")
    if best is None or abs(r) > abs(best[1]):
        best = (k, r, m, zz)
fold, r, model, zfit = best
sign = 1.0 if r >= 0 else -1.0
print(f"\n==> using fold{fold} (corr {r:+.3f}, sign {sign:+.0f})")

# ---- 2) saliency for the 2 targets (+ control mean reference) ----
ctrl_idx = [i for i, c in enumerate(codes) if y[i] == 0]
ctrl_mean_eem = X[ctrl_idx].mean(0)            # (T,1,H,W)


def saliency(idx):
    x = X[idx:idx+1].clone().to(DEV).requires_grad_(True)
    lat = [model.encoder(x[:, t]) for t in range(x.shape[1])]
    z, attn = model.attention(torch.stack(lat, 1))
    (z[0, DIM] * sign).backward()
    g = x.grad.detach().abs()[0, :, 0].numpy()  # (T,H,W)
    return g, attn.detach()[0].numpy()


# array is (H=512 emission rows, W=71 excitation cols)
# imshow: x-axis = excitation (cols), y-axis = emission (rows)
ex_lo, ex_hi = ex_axis.min(), ex_axis.max()
em_lo, em_hi = em_axis.min(), em_axis.max()
extent = [ex_lo, ex_hi, em_lo, em_hi]
tp_names = ["0h", "6h", "24h"]

summary = []
for code in TARGETS:
    idx = codes.index(code)
    g, attn = saliency(idx)                     # (T,H,W)
    g_agg = (g * attn[:, None, None]).sum(0)    # attention-weighted over timepoints -> (H,W)
    eem = X[idx, :, 0].numpy()                  # (T,H,W)
    eem_agg = (eem * attn[:, None, None]).sum(0)
    dev_agg = ((X[idx] - ctrl_mean_eem)[:, 0].numpy() * attn[:, None, None]).sum(0)
    contrib = g_agg * dev_agg                   # saliency x deviation-from-controls

    # locate peak contribution (where this sample's EEM deviates AND dim477 is sensitive)
    # contrib is (H=emission rows, W=excitation cols)
    fy, fx = np.unravel_index(np.argmax(np.abs(contrib)), contrib.shape)
    peak_em = em_axis[fy]      # row index -> emission
    peak_ex = ex_axis[fx]      # col index -> excitation
    summary.append(dict(code=code.split('.')[0], attn=np.round(attn, 3).tolist(),
                        peak_ex_nm=round(float(peak_ex), 0), peak_em_nm=round(float(peak_em), 0),
                        peak_dev=round(float(dev_agg[fy, fx]), 4),
                        peak_sal=round(float(g_agg[fy, fx]), 6)))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, M, ttl, cm in zip(
        ax, [eem_agg, g_agg, contrib],
        [f"{code.split('.')[0]} — EEM (attn-agg)",
         f"|d z[477]/d x| saliency",
         "saliency x (sample - CTRL mean)"],
        ["viridis", "magma", "seismic"]):
        vmax = np.abs(M).max()
        im = a.imshow(M, aspect="auto", origin="lower", extent=extent, cmap=cm,
                      vmin=(-vmax if cm == "seismic" else None),
                      vmax=(vmax if cm == "seismic" else None))
        a.set_xlabel("excitation (nm)"); a.set_ylabel("emission (nm)"); a.set_title(ttl)
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    ax[2].plot(peak_ex, peak_em, "kx", ms=12, mew=2)
    fig.suptitle(f"dim477 trace — {code.split('.')[0]} "
                 f"(CTRL; dim477 ALS-side) | attn 0h/6h/24h = {np.round(attn,2)}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, f"dim477_trace_{code.split('.')[0]}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)

print("\n=== peak dim477-driving EEM location (where sample deviates from controls) ===")
print(pd.DataFrame(summary).to_string(index=False))
pd.DataFrame(summary).to_csv(os.path.join(OUT, "dim477_trace_summary.csv"), index=False)
