"""
Tier-3 QC in the autoencoder's own feature space.

Motivation: the peak-based QC (qc_metrics.py) validates the 12-chirality
relative-share space -- but only ~17% of the classifier's disease score is
explained by that space (triangulation2_ae.py). So the existing QC certifies a
representation the model largely does not use. This adds the missing check:

    Do the two technical controls (PBS = no protein, FBS = reference corona)
    remain distinguishable in the features the conv-AE actually computes?

If yes, the plate is sound *in the representation the classifier consumes*.

Runs the 5 saved fold models (5_fold_models_original/) over the 7 exp5 control
wells (4 PBS, 3 FBS) x 3 timepoints, using the identical preprocessing as
eval_5fold_metrics.py, and extracts z_agg (512-d attention-aggregated latent)
plus the classifier probability.

Reported, per fold model:
  A) separation of PBS vs FBS in z_agg     -- multivariate effect size,
     leave-one-out projected SSMD, exact permutation p (C(7,3)=35)
  B) the same statistics in the 12-peak share space, on the SAME wells,
     as a head-to-head comparison
  C) where the controls sit relative to the 39 patients -- tests whether the
     separation is trivial out-of-distribution behaviour
  D) what the classifier says about a protein-free well (model negative control)

Outputs: qc_ae_controls_summary.csv, qc_ae_controls_perwell.csv,
         FigSX_qc_ae_controls.png/.pdf
"""
import os, re, glob, importlib.util, itertools
import numpy as np, pandas as pd
from scipy import stats
import torch
from openpyxl import load_workbook
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
HERE = os.path.dirname(os.path.abspath(__file__))
TC   = os.path.join(ROOT, "tech_controls")
MODEL_DIR = os.path.join(ROOT, "5_fold_models_original")
TP_DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
DEVICE, LATENT_DIM = torch.device("cpu"), 512
BLUE, VERM, GREY, INK, MUT = "#0072B2", "#D55E00", "#8a8a8a", "#222", "#8a8a8a"
torch.manual_seed(42); np.random.seed(42)

spec = importlib.util.spec_from_file_location(
    "cae", os.path.join(ROOT, "conv_autoencoder_detailed.py"))
cae = importlib.util.module_from_spec(spec); spec.loader.exec_module(cae)
Model = cae.ConvAutoencoderWithAttention


def load_excel(fp):
    """Identical preprocessing to eval_5fold_metrics.py: strip header row and
    index column, per-image min-max normalisation."""
    ws = load_workbook(fp, data_only=True).active
    data = []
    for ri, row in enumerate(ws.iter_rows(values_only=True)):
        if ri == 0:
            continue
        rd = [float(c) if isinstance(c, (int, float)) and c is not None else 0.0
              for ci, c in enumerate(row) if ci != 0]
        if rd:
            data.append(rd)
    arr = np.array(data, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    return (arr-mn)/(mx-mn) if mx > mn else np.zeros_like(arr)


# ---------------------------------------------------------------- control files
def control_map():
    """well -> {tp: path}. 0h lives partly at the tech_controls root."""
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

CM = control_map()
CWELLS = sorted(CM, key=lambda w: (w[0], w))
CGRP = np.array(["PBS" if w[0] == "P" else "FBS" for w in CWELLS])
print(f"control wells with all 3 timepoints: {CWELLS}")
print(f"  {(CGRP=='PBS').sum()} PBS / {(CGRP=='FBS').sum()} FBS\n")

# ---------------------------------------------------------------- load stacks
lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
codes = lab["code"].astype(str).tolist()
y_pat = np.array([1 if str(g).strip().lower() == "als" else 0 for g in lab["group"]])

print("Loading EEMs...")
Xp = np.stack([np.stack([load_excel(os.path.join(d, f"{c}.xlsx")) for d in TP_DIRS])
               for c in codes]).astype(np.float32)
Xc = np.stack([np.stack([load_excel(CM[w][tp]) for tp in ["0h", "6h", "24h"]])
               for w in CWELLS]).astype(np.float32)
print(f"  patients {Xp.shape}   controls {Xc.shape}")
Tp = torch.from_numpy(Xp).unsqueeze(2); Tc = torch.from_numpy(Xc).unsqueeze(2)

# ---------------------------------------------------------------- inference
def embed(model, X):
    """z_agg (attention-aggregated latent) and P(class=1)."""
    with torch.no_grad():
        zs = torch.stack([model.encoder(X[:, t]) for t in range(X.shape[1])], dim=1)
        z_agg, _ = model.attention(zs)
        logits = model.classifier(z_agg)
        return z_agg.cpu().numpy(), torch.softmax(logits, 1)[:, 1].cpu().numpy()

# ---------------------------------------------------------------- statistics
def mv_effect(A, B):
    """multivariate effect size: ||mu_A - mu_B|| / sqrt(pooled mean within-group var)."""
    pooled = np.sqrt(((A.var(0, ddof=1) + B.var(0, ddof=1))/2).mean())
    return np.linalg.norm(A.mean(0)-B.mean(0))/pooled if pooled > 0 else np.nan

def loo_ssmd(Z, g):
    """leave-one-out projection onto the between-group axis, then SSMD.
    The axis excludes the projected well, so the statistic is not circular."""
    proj = np.empty(len(Z))
    for i in range(len(Z)):
        keep = np.ones(len(Z), bool); keep[i] = False
        a = Z[keep & (g == "FBS")].mean(0) - Z[keep & (g == "PBS")].mean(0)
        n = np.linalg.norm(a)
        proj[i] = Z[i] @ (a/n) if n > 0 else 0.0
    f, p = proj[g == "FBS"], proj[g == "PBS"]
    d = np.sqrt(f.var(ddof=1)+p.var(ddof=1))
    return (f.mean()-p.mean())/d if d > 0 else np.nan, proj

def perm_p(Z, g, statfn):
    """exact permutation p over all C(7,3) relabellings (two-sided)."""
    obs = abs(statfn(Z, g))
    idx = np.arange(len(g)); nF = (g == "FBS").sum()
    null = []
    for combo in itertools.combinations(idx, nF):
        gg = np.array(["PBS"]*len(g), dtype=object); gg[list(combo)] = "FBS"
        null.append(abs(statfn(Z, gg)))
    null = np.array(null)
    return (null >= obs).mean(), len(null)

stat_mv  = lambda Z, g: mv_effect(Z[g == "FBS"], Z[g == "PBS"])
stat_loo = lambda Z, g: loo_ssmd(Z, g)[0]          # cross-validated -> better powered

def perm_p_1d(v, g):
    """exact permutation p on a scalar per-well readout (two-sided, |mean diff|)."""
    obs = abs(v[g == "FBS"].mean() - v[g == "PBS"].mean())
    idx = np.arange(len(g)); nF = (g == "FBS").sum(); null = []
    for combo in itertools.combinations(idx, nF):
        m = np.zeros(len(g), bool); m[list(combo)] = True
        null.append(abs(v[m].mean() - v[~m].mean()))
    return (np.array(null) >= obs).mean()

# ---------------------------------------------------------------- run folds
rows, perwell = [], []
for k in range(1, 6):
    model = Model(in_channels=1, latent_dim=LATENT_DIM, num_classes=2).to(DEVICE)
    with torch.no_grad():
        _ = model(Tc[:1])                       # lazy-init decoder
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"best_model_fold{k}.pth"),
                                     map_location=DEVICE), strict=True)
    model.eval()
    Zc, Pc = embed(model, Tc)
    Zp, Pp = embed(model, Tp)

    mv = stat_mv(Zc, CGRP)
    ss, proj = loo_ssmd(Zc, CGRP)
    pv, nperm = perm_p(Zc, CGRP, stat_mv)
    pv_loo, _ = perm_p(Zc, CGRP, stat_loo)         # cross-validated statistic
    pv_pals = perm_p_1d(Pc, CGRP)                  # classifier axis, fit on patients only

    # C) are the controls simply out-of-distribution?
    cen = Zp.mean(0); scat = np.linalg.norm(Zp-cen, axis=1)
    dP = np.linalg.norm(Zc[CGRP == "PBS"]-cen, axis=1).mean()
    dF = np.linalg.norm(Zc[CGRP == "FBS"]-cen, axis=1).mean()

    rows.append(dict(fold=k, mv_effect=mv, loo_ssmd=ss, perm_p=pv,
                     perm_p_loo=pv_loo, perm_p_PALS=pv_pals,
                     dist_PBS_z=(dP-scat.mean())/scat.std(ddof=1),
                     dist_FBS_z=(dF-scat.mean())/scat.std(ddof=1),
                     P_ALS_PBS=Pc[CGRP == "PBS"].mean(), P_ALS_FBS=Pc[CGRP == "FBS"].mean(),
                     P_ALS_patients=Pp.mean()))
    for i, w in enumerate(CWELLS):
        perwell.append(dict(fold=k, well=w, group=CGRP[i], proj=proj[i], P_ALS=Pc[i]))
    print(f"fold{k}: mv={mv:5.2f} (p={pv:.3f})  LOO-SSMD={ss:+6.2f} (p={pv_loo:.3f})  "
          f"P_ALS PBS={Pc[CGRP=='PBS'].mean():.3f} FBS={Pc[CGRP=='FBS'].mean():.3f} "
          f"(p={pv_pals:.3f})")

R = pd.DataFrame(rows); PW = pd.DataFrame(perwell)
R.to_csv(os.path.join(HERE, "qc_ae_controls_summary.csv"), index=False)
PW.to_csv(os.path.join(HERE, "qc_ae_controls_perwell.csv"), index=False)

# ---------------------------------------------------------------- B) peak-space head-to-head
L = pd.read_csv(os.path.join(HERE, "controls_features_long.csv"), dtype={'chirality': str})
CHIRS = ['6.5','7.5','7.6','8.3','8.4','8.6','8.7','9.4','9.5','10.2','10.3','10.5']
def peak_vec(w, feat='gauss_max'):
    v = []
    for tp in ['0h', '6h', '24h']:
        s = L[(L['sample'] == w) & (L.tp == tp)].set_index('chirality').reindex(CHIRS)[feat]
        a = s.to_numpy(float); v.append(a/np.nanmean(a))
    return np.concatenate(v)
Zpk = np.array([peak_vec(w) for w in CWELLS])
mv_pk = stat_mv(Zpk, CGRP)
ss_pk, proj_pk = loo_ssmd(Zpk, CGRP)
pv_pk, _ = perm_p(Zpk, CGRP, stat_mv)

print("\n" + "="*74)
print("A) AE feature space (z_agg, 512-d), mean over 5 fold models")
print("="*74)
print(f"  multivariate effect ||mu_F-mu_P||/sigma_within = {R.mv_effect.mean():.2f} "
      f"(range {R.mv_effect.min():.2f}-{R.mv_effect.max():.2f})")
print(f"  LOO-projected SSMD                             = {R.loo_ssmd.mean():+.2f} "
      f"(range {R.loo_ssmd.min():+.2f}-{R.loo_ssmd.max():+.2f})")
print(f"  exact permutation p, in-sample stat             = "
      f"{R.perm_p.min():.3f}-{R.perm_p.max():.3f}   [floor = {1/nperm:.3f}, n={nperm}]")
print(f"  exact permutation p, LOO (cross-validated)      = "
      f"{R.perm_p_loo.min():.3f}-{R.perm_p_loo.max():.3f}")
print(f"  exact permutation p, on P_ALS (classifier axis) = "
      f"{R.perm_p_PALS.min():.3f}-{R.perm_p_PALS.max():.3f}")
print("  NOTE the multivariate effect size is inflated by construction: with n=7 in")
print("  512 dimensions every relabelling yields a large value, which is why the")
print("  permutation test -- not the effect size -- is the number to read.")
print(f"\nB) 12-peak relative-share space, SAME wells  (head-to-head)")
print(f"  multivariate effect = {mv_pk:.2f}   LOO-SSMD = {ss_pk:+.2f}   perm p = {pv_pk:.3f}")
print(f"\nC) are the controls merely out-of-distribution?")
print(f"  distance from the 39-patient centroid, in units of patient scatter:")
print(f"     PBS wells  z = {R.dist_PBS_z.mean():+.2f}      FBS wells  z = {R.dist_FBS_z.mean():+.2f}")
print("     (|z| < ~2 means the controls sit inside the patient cloud, so the")
print("      separation is not trivial OOD behaviour)")
print(f"\nD) model negative control -- what does the classifier say about")
print(f"   a well containing no protein at all?")
print(f"     P_ALS  PBS = {R.P_ALS_PBS.mean():.3f}   FBS = {R.P_ALS_FBS.mean():.3f}   "
      f"patients = {R.P_ALS_patients.mean():.3f}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
fig, axs = plt.subplots(1, 4, figsize=(14.2, 3.4))
fig.subplots_adjust(left=0.05, right=0.995, bottom=0.19, top=0.83, wspace=0.36)

a = axs[0]
for gname, col in [("PBS", BLUE), ("FBS", VERM)]:
    s = PW[PW.group == gname]
    a.scatter(s.fold + np.where(s.group == "PBS", -0.13, 0.13), s.proj, s=26, color=col,
              alpha=.9, zorder=3, label=gname)
a.set_xticks(range(1, 6)); a.set_xlabel("fold model"); a.set_ylabel("LOO projection on PBS→FBS axis")
a.set_title("(a) Controls separate in\n AE feature space", loc="left")
a.legend(frameon=False, fontsize=7.5)

b = axs[1]
x = np.arange(2); vals = [R.mv_effect.mean(), mv_pk]
b.bar(x, vals, yerr=[R.mv_effect.std(ddof=1), 0], capsize=3, color=[BLUE, GREY], alpha=.45)
b.set_xticks(x); b.set_xticklabels(["AE z_agg\n(512-d)", "12-peak\nshare"])
b.set_ylabel("||μ$_F$−μ$_P$|| / σ$_{within}$")
b.set_title("(b) Effect size is inflated —\n n=7 in 512-d. Do not read this.", loc="left")
for xi, vi in zip(x, vals):
    b.text(xi, vi*0.5, "n.s.", ha="center", fontsize=11, fontweight="bold", color=VERM)

c = axs[2]
w = 0.26
c.bar(np.arange(5)-w, R.perm_p, w, color=GREY, alpha=.85, label="in-sample")
c.bar(np.arange(5), R.perm_p_loo, w, color=BLUE, alpha=.85, label="LOO")
c.bar(np.arange(5)+w, R.perm_p_PALS, w, color=VERM, alpha=.85, label="P$_{ALS}$ axis")
c.axhline(0.05, color=INK, ls='--', lw=1)
c.axhline(1/nperm, color=MUT, ls=':', lw=1)
c.text(-0.45, 1/nperm*1.6, f"floor {1/nperm:.3f}", fontsize=6.5, color=MUT)
c.set_xticks(range(5)); c.set_xticklabels([f"f{k}" for k in range(1, 6)])
c.set_xlabel("fold model"); c.set_ylabel("exact permutation p"); c.set_ylim(0, 1.05)
c.set_title(f"(c) No separation on any\n statistic ({nperm} relabellings)", loc="left")
c.legend(frameon=False, fontsize=6.8, ncol=3, loc="lower center")

d = axs[3]
for i, (gname, col) in enumerate([("PBS", BLUE), ("FBS", VERM)]):
    v = PW[PW.group == gname]['P_ALS']
    d.scatter(np.full(len(v), i)+np.random.RandomState(0).uniform(-.09, .09, len(v)),
              v, s=22, color=col, alpha=.85, zorder=3)
d.scatter(np.full(5, 2)+np.random.RandomState(1).uniform(-.09, .09, 5), R.P_ALS_patients,
          s=22, color=GREY, alpha=.85, zorder=3)
d.axhline(0.5, color=INK, lw=0.9, ls='--')
d.set_xticks(range(3)); d.set_xticklabels(["PBS\n(no protein)", "FBS", "patients\n(mean)"])
d.set_ylabel("classifier P(ALS)"); d.set_ylim(-0.05, 1.05)
d.set_title("(d) Model negative control", loc="left")

fig.savefig(os.path.join(HERE, "FigSX_qc_ae_controls.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(HERE, "FigSX_qc_ae_controls.pdf"), bbox_inches="tight")
print("\nSaved FigSX_qc_ae_controls.png/.pdf + qc_ae_controls_{summary,perwell}.csv")
