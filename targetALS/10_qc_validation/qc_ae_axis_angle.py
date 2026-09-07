"""
How far off is the PBS->FBS axis from the CTRL->ALS axis in the conv-AE latent?

Both are difference-of-means directions in z_agg:
    u_corona  = mean(z FBS wells)  - mean(z PBS wells)      (n = 3 vs 4)
    u_disease = mean(z ALS)        - mean(z CTRL)           (n = 20 vs 19)
and we report cos(u_corona, u_disease) -- 0 = orthogonal/independent,
|1| = the model encodes protein-presence and disease along the same direction.

Three things make a raw cosine uninterpretable on its own, so each is measured:

  NULL     In 512-d two arbitrary directions are near-orthogonal by default, so a
           small cosine proves nothing. Exact permutation over all C(7,3)=35
           control relabellings gives the null for this geometry.
  CEILING  u_corona is estimated from 7 wells and is therefore noisy; a cosine
           against a noisy axis is attenuated toward 0. Leave-one-out replicates
           of each axis give its self-consistency -- the largest cosine the data
           could support even for two identical directions.
  SCALING  Per-image min-max normalisation (deployed) hides the real PBS/FBS
           contrast, so under it u_corona is mostly noise. Everything is
           recomputed under global scaling, where that contrast survives.

Also computed: a BRIGHTNESS axis in latent space (z regressed on raw mean EEM
intensity over all 46 samples). Since the PBS/FBS contrast is a brightness
effect, cos(u_disease, u_brightness) is the direct confound test -- does the
disease direction carry an absolute-intensity component?

Inference only. Outputs: qc_ae_axis_angle.csv, FigSX_qc_ae_axis_angle.png/.pdf
"""
import os, re, glob, importlib.util, itertools
import numpy as np, pandas as pd
import torch
from openpyxl import load_workbook
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
HERE = os.path.dirname(os.path.abspath(__file__))
TC   = os.path.join(ROOT, "tech_controls")
MODEL_DIR = os.path.join(ROOT, "5_fold_models_original")
TP_DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
DEVICE, LATENT_DIM, SEED = torch.device("cpu"), 512, 42
BLUE, VERM, GREEN, GREY, INK, MUT = "#0072B2", "#D55E00", "#009E73", "#9aa3ad", "#222", "#8a8a8a"
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)

spec = importlib.util.spec_from_file_location(
    "cae", os.path.join(ROOT, "conv_autoencoder_detailed.py"))
cae = importlib.util.module_from_spec(spec); spec.loader.exec_module(cae)
Model = cae.ConvAutoencoderWithAttention

def load_raw(fp):
    ws = load_workbook(fp, data_only=True).active
    data = []
    for ri, row in enumerate(ws.iter_rows(values_only=True)):
        if ri == 0:
            continue
        rd = [float(c) if isinstance(c, (int, float)) and c is not None else 0.0
              for ci, c in enumerate(row) if ci != 0]
        if rd:
            data.append(rd)
    return np.array(data, dtype=np.float32)

def control_map():
    m = {}
    for tp, pats in [("0h", [os.path.join(TC, "*.xlsx"), os.path.join(TC, "0h", "*.xlsx")]),
                     ("6h", [os.path.join(TC, "6h", "*.xlsx")]),
                     ("24h", [os.path.join(TC, "24h", "*.xlsx")])]:
        for pat in pats:
            for fp in glob.glob(pat):
                mm = re.match(r"^([FP]\d)", os.path.basename(fp))
                if mm:
                    m.setdefault(mm.group(1), {})[tp] = fp
    return {w: d for w, d in sorted(m.items()) if len(d) == 3}

CM = control_map(); WELLS = sorted(CM, key=lambda w: (w[0], w))
CGRP = np.array(["PBS" if w[0] == "P" else "FBS" for w in WELLS])
lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
codes = lab["code"].astype(str).tolist()
PGRP = np.array(["ALS" if str(g).strip().lower() == "als" else "CTRL" for g in lab["group"]])
print(f"controls {len(WELLS)} ({(CGRP=='PBS').sum()} PBS/{(CGRP=='FBS').sum()} FBS)   "
      f"patients {len(codes)} ({(PGRP=='ALS').sum()} ALS/{(PGRP=='CTRL').sum()} CTRL)")

RC, RP = os.path.join(HERE, "raw_controls.npy"), os.path.join(HERE, "raw_patients.npy")
RAWC = np.load(RC) if os.path.exists(RC) else np.stack(
    [np.stack([load_raw(CM[w][tp]) for tp in ["0h", "6h", "24h"]]) for w in WELLS])
if not os.path.exists(RC): np.save(RC, RAWC)
if os.path.exists(RP):
    RAWP = np.load(RP)
else:
    print("loading 39 patient EEMs (cached after this run)...")
    RAWP = np.stack([np.stack([load_raw(os.path.join(d, f"{c}.xlsx")) for d in TP_DIRS])
                     for c in codes])
    np.save(RP, RAWP)
print(f"raw arrays: controls {RAWC.shape}  patients {RAWP.shape}")

ALLRAW = np.concatenate([RAWP, RAWC], 0)                  # 46 samples
GMAX = float(ALLRAW.max())
def norm_perimage(a):
    mn, mx = a.min(), a.max()
    return (a-mn)/(mx-mn) if mx > mn else np.zeros_like(a)
def norm_global(a):
    return a/GMAX
BRIGHT = np.array([ALLRAW[i].mean() for i in range(len(ALLRAW))])   # raw mean intensity

def encode_all(X):
    T = torch.from_numpy(X.astype(np.float32)).unsqueeze(2)
    out = []
    for k in range(1, 6):
        model = Model(in_channels=1, latent_dim=LATENT_DIM, num_classes=2).to(DEVICE)
        with torch.no_grad():
            _ = model(T[:1])
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"best_model_fold{k}.pth"),
                                         map_location=DEVICE), strict=True)
        model.eval()
        with torch.no_grad():
            zs = torch.stack([model.encoder(T[:, t]) for t in range(3)], dim=1)
            za, _ = model.attention(zs)
        out.append(za.cpu().numpy())
    return out

def unit(v):
    n = np.linalg.norm(v)
    return v/n if n > 0 else v
def cos(a, b):
    return float(np.dot(unit(a), unit(b)))
def axis(Z, g, pos, neg):
    return Z[g == pos].mean(0) - Z[g == neg].mean(0)

COMBOS = list(itertools.combinations(range(7), 3))
rng = np.random.RandomState(SEED)

rows = []
for tag, fn in [("per-image (deployed)", norm_perimage), ("global scale", norm_global)]:
    X = np.stack([[fn(ALLRAW[i, t]) for t in range(3)] for i in range(len(ALLRAW))])
    Zs = encode_all(X)
    for k, Z in enumerate(Zs, start=1):
        Zp, Zc = Z[:len(codes)], Z[len(codes):]
        u_cor = axis(Zc, CGRP, "FBS", "PBS")
        u_dis = axis(Zp, PGRP, "ALS", "CTRL")
        c_obs = cos(u_cor, u_dis)

        # brightness axis: least-squares direction of z on raw mean intensity
        b = BRIGHT - BRIGHT.mean(); Zcen = Z - Z.mean(0)
        u_bri = (b @ Zcen)/(b @ b)
        c_dis_bri = cos(u_dis, u_bri); c_cor_bri = cos(u_cor, u_bri)

        # NULL: exact permutation of the control labels
        null = []
        for cb in COMBOS:
            gg = np.array(["PBS"]*7, dtype=object); gg[list(cb)] = "FBS"
            null.append(cos(axis(Zc, gg, "FBS", "PBS"), u_dis))
        null = np.array(null)
        p_perm = (np.abs(null) >= abs(c_obs)).mean()

        # CEILING: leave-one-out self-consistency of each axis
        loo_c = [unit(axis(np.delete(Zc, i, 0), np.delete(CGRP, i), "FBS", "PBS"))
                 for i in range(7)]
        rel_cor = float(np.mean([loo_c[i] @ loo_c[j]
                                 for i in range(7) for j in range(i+1, 7)]))
        idx = rng.permutation(len(codes))
        h1, h2 = idx[:len(idx)//2], idx[len(idx)//2:]
        rel_dis = cos(axis(Zp[h1], PGRP[h1], "ALS", "CTRL"),
                      axis(Zp[h2], PGRP[h2], "ALS", "CTRL"))

        rows.append(dict(scaling=tag, fold=k, cos_cor_dis=c_obs,
                         angle_deg=np.degrees(np.arccos(np.clip(abs(c_obs), 0, 1))),
                         perm_p=p_perm, null_sd=null.std(),
                         reliab_corona=rel_cor, reliab_disease=rel_dis,
                         cos_disease_bright=c_dis_bri, cos_corona_bright=c_cor_bri))
        print(f"  {tag:22s} fold{k}: cos={c_obs:+.3f} ({rows[-1]['angle_deg']:.0f}deg) "
              f"p={p_perm:.3f} | reliab cor={rel_cor:+.2f} dis={rel_dis:+.2f} "
              f"| cos(dis,bright)={c_dis_bri:+.3f}")

R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "qc_ae_axis_angle.csv"), index=False)

print("\n" + "="*80)
print("PBS->FBS  vs  CTRL->ALS  axis alignment in z_agg")
print("="*80)
for tag in R.scaling.unique():
    s = R[R.scaling == tag]
    print(f"\n  {tag}")
    print(f"    cos(corona, disease) = {s.cos_cor_dis.mean():+.3f} "
          f"(range {s.cos_cor_dis.min():+.3f} to {s.cos_cor_dis.max():+.3f})  "
          f"-> {s.angle_deg.mean():.0f} deg from orthogonal-90")
    print(f"    exact permutation p  = {s.perm_p.min():.3f}-{s.perm_p.max():.3f}   "
          f"(null sd {s.null_sd.mean():.3f})")
    print(f"    RELIABILITY ceiling  : corona axis {s.reliab_corona.mean():+.2f}, "
          f"disease axis {s.reliab_disease.mean():+.2f}")
    print(f"    cos(disease, brightness) = {s.cos_disease_bright.mean():+.3f}   "
          f"cos(corona, brightness) = {s.cos_corona_bright.mean():+.3f}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axs = plt.subplots(1, 4, figsize=(14.6, 3.5))
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.80, wspace=0.42)
TAGS = ["per-image (deployed)", "global scale"]; COLS = [GREY, BLUE]; w = 0.36

a = axs[0]
for i, (tag, col) in enumerate(zip(TAGS, COLS)):
    s = R[R.scaling == tag]
    a.bar(np.arange(5)+(i-0.5)*w, s.cos_cor_dis, w, color=col, alpha=.9, label=tag)
a.axhline(0, color=INK, lw=.9)
for i, (tag, col) in enumerate(zip(TAGS, COLS)):
    sd = R[R.scaling == tag].null_sd.mean()
    a.axhline(2*sd, color=col, ls=":", lw=1); a.axhline(-2*sd, color=col, ls=":", lw=1)
a.set_xticks(range(5)); a.set_xticklabels([f"f{k}" for k in range(1, 6)])
a.set_xlabel("fold model"); a.set_ylabel("cos(corona, disease)")
a.set_title("(a) Axis alignment\n dotted = ±2 SD of null", loc="left")
a.legend(frameon=False, fontsize=6.5)

b = axs[1]
for i, (tag, col) in enumerate(zip(TAGS, COLS)):
    s = R[R.scaling == tag]
    b.bar(np.arange(5)+(i-0.5)*w, s.perm_p, w, color=col, alpha=.9)
b.axhline(0.05, color=VERM, ls="--", lw=1)
b.set_xticks(range(5)); b.set_xticklabels([f"f{k}" for k in range(1, 6)])
b.set_xlabel("fold model"); b.set_ylabel("exact permutation p"); b.set_ylim(0, 1.05)
b.set_title("(b) Is the alignment\n more than chance?", loc="left")

c = axs[2]
lbl = ["corona\naxis", "disease\naxis"]
for i, (tag, col) in enumerate(zip(TAGS, COLS)):
    s = R[R.scaling == tag]
    c.bar(np.arange(2)+(i-0.5)*w, [s.reliab_corona.mean(), s.reliab_disease.mean()], w,
          yerr=[s.reliab_corona.std(), s.reliab_disease.std()], capsize=3,
          color=col, alpha=.9, label=tag)
c.axhline(0, color=INK, lw=.9)
c.set_xticks(range(2)); c.set_xticklabels(lbl, fontsize=8)
c.set_ylabel("self-consistency (cos)")
c.set_title("(c) Reliability ceiling\n how well is each axis estimated?", loc="left")

d = axs[3]
for i, (tag, col) in enumerate(zip(TAGS, COLS)):
    s = R[R.scaling == tag]
    d.bar(np.arange(2)+(i-0.5)*w, [s.cos_disease_bright.mean(), s.cos_corona_bright.mean()], w,
          yerr=[s.cos_disease_bright.std(), s.cos_corona_bright.std()], capsize=3,
          color=col, alpha=.9, label=tag)
d.axhline(0, color=INK, lw=.9)
d.set_xticks(range(2)); d.set_xticklabels(["disease vs\nbrightness", "corona vs\nbrightness"],
                                          fontsize=8)
d.set_ylabel("cosine")
d.set_title("(d) Confound test:\n brightness component", loc="left")
d.legend(frameon=False, fontsize=6.5)

fig.savefig(os.path.join(HERE, "FigSX_qc_ae_axis_angle.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "FigSX_qc_ae_axis_angle.pdf"), bbox_inches="tight")
print("\nSaved FigSX_qc_ae_axis_angle.png/.pdf + qc_ae_axis_angle.csv")
