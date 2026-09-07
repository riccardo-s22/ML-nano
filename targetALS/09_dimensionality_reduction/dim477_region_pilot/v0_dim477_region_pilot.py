"""
PILOT: chirality-region reduction of the 24h dim477 reconstruction map.

Demonstrates the grant aim on exp5 data:
  1. Build the *24h component* of the dim477 reconstruction map:
     W(pixel) = mean_train signed d z_agg[477] / d x_24h   (fold3, label-free).
  2. Tile the EEM into contiguous excitation x emission regions.
  3. Rank regions by |reconstructed dim477 contribution| = |sum_R W * mean_x24h|.
  4. Add regions cumulatively; score each participant's normalized 24h spectrum
     projected onto the locked regions with signed W as weights -> regional scores.
  5. Evaluate held-out (LOO) AUROC vs #regions; stop when within tol of full map.
  6. Retain only regions whose signed score has consistent direction within groups.
  7. Label each locked region by nearest known SWCNT chirality.

Honest-eval notes: W and region ranking are LABEL-FREE (encoder + mean spectrum
only), so computing them once is not leakage. The only label-touching step -- the
logistic classifier on regional scores -- is leave-one-out with train-only
standardization. The AUROC-vs-k curve is shown in full so the stopping choice is
transparent (the real study will nest k-selection in an inner CV loop).
"""
import os, importlib.util
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PLS  = os.path.join(os.path.dirname(ROOT), "PLS")
TPD  = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT  = os.path.join(ROOT, "dim477_region_pilot_outputs"); os.makedirs(OUT, exist_ok=True)
DIM, DEV = 477, torch.device("cpu")
T24 = 2                      # index of 24h timepoint in TPD
N_EX_TILES, N_EM_TILES = 6, 8
AUROC_TOL = 0.02
SIGN_CONSIST = 0.70         # >=70% of a group's subjects share the group-mean sign
np.random.seed(0); torch.manual_seed(0)

spec = importlib.util.spec_from_file_location("sal", os.path.join(PLS, "dim35_chirality_saliency.py"))
sal = importlib.util.module_from_spec(spec); spec.loader.exec_module(sal)

lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
codes  = lab["code"].astype(str).tolist()
groups = dict(zip(lab["code"].astype(str), lab["group"].astype(str)))
y = np.array([1 if groups[c].strip().upper() == "ALS" else 0 for c in codes])   # ALS=1
em_axis, ex_axis = sal.get_wavelength_axes(TPD[0], codes[0])

ds = sal.ExcelImageDataset(codes, [0]*len(codes), TPD)
X = torch.stack([ds[i][0] for i in range(len(codes))], 0).float()   # (N,T,1,H,W)
N, Tn, _, H, Wd = X.shape
print(f"X {tuple(X.shape)} | em {em_axis.shape} ex {ex_axis.shape} | ALS {y.sum()} CTRL {(1-y).sum()}")

# ---------- model: pick fold reproducing canonical dim477 ----------
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

canon = pd.read_csv(os.path.join(PLS, "z_agg3.csv"))[["code", "dim477"]]
canon = dict(zip(canon["code"], canon["dim477"]))
cvec = np.array([canon.get(c, np.nan) for c in codes]); ok = ~np.isnan(cvec)
best = None
for k in range(1, 6):
    m = build_model(os.path.join(ROOT, f"5_fold_models_original/best_model_fold{k}.pth"))
    r = pearsonr(zagg_dim(m, DIM)[ok], cvec[ok])[0]
    print(f"  fold{k}: corr(z_agg[477], canonical) = {r:+.3f}")
    if best is None or abs(r) > abs(best[1]): best = (k, r, m)
fold, r, model = best
sign = 1.0 if r >= 0 else -1.0
print(f"==> fold{fold} (corr {r:+.3f})")

# ---------- 24h component of dim477 reconstruction map: W = mean signed d z[477]/d x_24h ----------
def saliency_24h(idx):
    x = X[idx:idx+1].clone().to(DEV).requires_grad_(True)
    lat = [model.encoder(x[:, t]) for t in range(Tn)]
    z, _ = model.attention(torch.stack(lat, 1))
    (z[0, DIM] * sign).backward()
    return x.grad.detach()[0, T24, 0].numpy()          # signed (H,W), 24h slot

Wsal = np.stack([saliency_24h(i) for i in range(N)], 0)  # (N,H,W) signed
Wmap = Wsal.mean(0)                                       # label-free weight map (H,W)
x24  = X[:, T24, 0].numpy()                              # (N,H,W) normalized 24h spectra
mean_x24 = x24.mean(0)

# ---------- contiguous EEM tiling ----------
ex_edges = np.linspace(0, Wd, N_EX_TILES + 1).astype(int)
em_edges = np.linspace(0, H,  N_EM_TILES + 1).astype(int)
tiles = []
for ei in range(N_EX_TILES):
    for mj in range(N_EM_TILES):
        c0, c1 = ex_edges[ei], ex_edges[ei+1]
        r0, r1 = em_edges[mj], em_edges[mj+1]
        contrib = float(np.sum(Wmap[r0:r1, c0:c1] * mean_x24[r0:r1, c0:c1]))  # signed dim477 share
        tiles.append(dict(ei=ei, mj=mj, r0=r0, r1=r1, c0=c0, c1=c1,
                          contrib=contrib, absc=abs(contrib)))
tiles = pd.DataFrame(tiles).sort_values("absc", ascending=False).reset_index(drop=True)

# regional score matrix: S[i, t] = sum_{pixel in tile} W * x24_i   (projection, signed weights)
def tile_score(i, t):
    return float(np.sum(Wmap[t.r0:t.r1, t.c0:t.c1] * x24[i, t.r0:t.r1, t.c0:t.c1]))
S = np.array([[tile_score(i, t) for t in tiles.itertuples()] for i in range(N)])  # (N, n_tiles)
s_full = (Wmap[None] * x24).reshape(N, -1).sum(1)                                 # full-map dim477-24h score

# ---------- sign-consistency filter (direction consistent within ALS and within CTRL) ----------
def sign_consistent(col):
    keep = True
    for g in (0, 1):
        v = col[y == g]
        mu = np.sign(v.mean()) if v.mean() != 0 else 1
        frac = np.mean(np.sign(v) == mu)
        keep &= (frac >= SIGN_CONSIST)
    return keep
tiles["sign_ok"] = [sign_consistent(S[:, j]) for j in range(len(tiles))]

# ---------- LOO AUROC vs cumulative #regions ----------
def loo_auc(feat):                       # feat: (N, k)
    if feat.ndim == 1: feat = feat[:, None]
    p = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(C=0.5, max_iter=1000).fit(sc.transform(feat[tr]), y[tr])
        p[i] = clf.predict_proba(sc.transform(feat[i:i+1]))[0, 1]
    return roc_auc_score(y, p), p

order = tiles.index.tolist()             # already ranked by |contrib|
auc_full, p_full = loo_auc(s_full)
aucs = []
for k in range(1, len(order) + 1):
    cols = [order[j] for j in range(k)]
    aucs.append(loo_auc(S[:, cols])[0])
aucs = np.array(aucs)

# smallest k within tol of full map
within = np.where(aucs >= auc_full - AUROC_TOL)[0]
k_star = int(within[0] + 1) if len(within) else int(np.argmax(aucs) + 1)
locked_ranks = order[:k_star]
# apply sign-consistency retention among the locked set
locked = [j for j in locked_ranks if tiles.loc[j, "sign_ok"]]
if not locked: locked = locked_ranks
auc_locked, p_locked = loo_auc(S[:, locked])
print(f"\nfull-map LOO AUROC={auc_full:.3f} | k*={k_star} (within {AUROC_TOL}) "
      f"AUROC={aucs[k_star-1]:.3f} | sign-locked {len(locked)} regions AUROC={auc_locked:.3f}")

# ---------- label locked regions by nearest chirality ----------
chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    name = em = ex = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): name = ln.strip("[]")
        elif ln.startswith("Emission"): em = float(ln.split("=")[1])
        elif ln.startswith("Excitation"):
            ex = float(ln.split("=")[1]); chir.append((name, ex, em))
def label_tile(t):
    ex_lo, ex_hi = ex_axis[t.c0], ex_axis[min(t.c1, Wd)-1]
    em_lo, em_hi = em_axis[t.r0], em_axis[min(t.r1, H)-1]
    lo_ex, hi_ex = min(ex_lo, ex_hi), max(ex_lo, ex_hi)
    lo_em, hi_em = min(em_lo, em_hi), max(em_lo, em_hi)
    inside = [nm for nm, ex, em in chir if lo_ex <= ex <= hi_ex and lo_em <= em <= hi_em]
    return ",".join(f"({n})" for n in inside) if inside else "-"

rows = []
for rank_pos, j in enumerate(locked_ranks):
    t = tiles.loc[j]
    ex_lo, ex_hi = sorted([ex_axis[t.c0], ex_axis[min(int(t.c1), Wd)-1]])
    em_lo, em_hi = sorted([em_axis[t.r0], em_axis[min(int(t.r1), H)-1]])
    rows.append(dict(rank=rank_pos+1,
                     ex_nm=f"{ex_lo:.0f}-{ex_hi:.0f}", em_nm=f"{em_lo:.0f}-{em_hi:.0f}",
                     chirality=label_tile(t), contrib=round(t.contrib, 4),
                     sign="+" if t.contrib > 0 else "-",
                     sign_ok=bool(t.sign_ok),
                     als_mean=round(S[y==1, j].mean(), 4), ctrl_mean=round(S[y==0, j].mean(), 4)))
locked_df = pd.DataFrame(rows)
locked_df.to_csv(os.path.join(OUT, "locked_regions.csv"), index=False)
print("\n=== locked regions ===\n" + locked_df.to_string(index=False))

# ==================== FIGURES ====================
FOLD_C, POOL_C = "#4b5bd4", "#0e8a8f"
ex_lo, ex_hi = ex_axis.min(), ex_axis.max()
em_lo, em_hi = em_axis.min(), em_axis.max()
extent = [ex_lo, ex_hi, em_lo, em_hi]

# Fig 1: weight map + tiles + chirality peaks + locked highlight
fig, ax = plt.subplots(figsize=(7.2, 6), dpi=170)
vmax = np.abs(Wmap).max()
im = ax.imshow(Wmap, aspect="auto", origin="lower", extent=extent, cmap="seismic",
               vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed dim477 weight  (24h)")
for e in ex_edges: ax.axvline(ex_axis[min(e, Wd-1)], color="k", lw=0.4, alpha=0.3)
for e in em_edges: ax.axhline(em_axis[min(e, H-1)], color="k", lw=0.4, alpha=0.3)
for nm, ex, em in chir:
    ax.plot(ex, em, "kx", ms=7, mew=1.6)
    ax.text(ex+3, em, f"({nm})", fontsize=7, color="k", va="center")
for j in locked:
    t = tiles.loc[j]
    ax.add_patch(Rectangle((ex_axis[t.c0], em_axis[t.r0]),
                           ex_axis[min(int(t.c1),Wd)-1]-ex_axis[t.c0],
                           em_axis[min(int(t.r1),H)-1]-em_axis[t.r0],
                           fill=False, edgecolor="lime", lw=2.2))
ax.set_xlabel("excitation (nm)"); ax.set_ylabel("emission (nm)")
ax.set_title(f"24h dim477 reconstruction weights\n(green = {len(locked)} locked chirality regions)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_weightmap_regions.png")); plt.close(fig)

# Fig 2: AUROC vs cumulative #regions
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
ax.plot(np.arange(1, len(aucs)+1), aucs, "-o", color=FOLD_C, ms=4, label="cumulative regions (LOO)")
ax.axhline(auc_full, color=POOL_C, lw=2, label=f"full 24h map = {auc_full:.3f}")
ax.axhspan(auc_full-AUROC_TOL, auc_full, color=POOL_C, alpha=0.12, label=f"±{AUROC_TOL} tolerance")
ax.axvline(k_star, color="grey", ls="--", lw=1)
ax.plot(k_star, aucs[k_star-1], "*", color="crimson", ms=15, zorder=5,
        label=f"k* = {k_star} regions ({aucs[k_star-1]:.3f})")
ax.set_xlabel("number of regions (ranked by |dim477 contribution|)")
ax.set_ylabel("held-out AUROC"); ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.25)
ax.set_title("Region-reduced vs complete 24h map", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_auroc_vs_regions.png")); plt.close(fig)

# Fig 3: regional-score fingerprint (subjects x locked regions) + group separation
order_subj = np.argsort(y)
Slock = S[:, locked][order_subj]
Sz = (Slock - Slock.mean(0)) / (Slock.std(0) + 1e-9)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=170,
                             gridspec_kw={"width_ratios": [2.4, 1]})
im = a1.imshow(Sz.T, aspect="auto", cmap="coolwarm", vmin=-2.2, vmax=2.2)
a1.set_xlabel("participant (sorted: CTRL | ALS)"); a1.set_ylabel("locked region")
a1.set_yticks(range(len(locked)))
a1.set_yticklabels([f"{locked_df.loc[i,'chirality']} {locked_df.loc[i,'em_nm']}nm"
                    if locked_df.loc[i,'chirality']!='-' else locked_df.loc[i,'em_nm']+'nm'
                    for i in range(len(locked))], fontsize=7)
a1.axvline((y==0).sum()-0.5, color="k", lw=1.5)
plt.colorbar(im, ax=a1, fraction=0.046, pad=0.04, label="regional score (z)")
a1.set_title("Regional spectral-score fingerprint", fontsize=11)

score_sum = S[:, locked].sum(1)
for g, c, lbl in [(0, POOL_C, "CTRL"), (1, "crimson", "ALS")]:
    a2.scatter(np.full((y==g).sum(), g) + np.random.uniform(-0.08, 0.08, (y==g).sum()),
               score_sum[y==g], color=c, s=28, alpha=0.8, label=lbl)
a2.set_xticks([0, 1]); a2.set_xticklabels(["CTRL", "ALS"])
a2.set_ylabel("summed locked-region score")
a2.set_title(f"locked AUROC = {auc_locked:.3f}", fontsize=11)
a2.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_regional_scores.png")); plt.close(fig)

pd.DataFrame(dict(k=np.arange(1, len(aucs)+1), loo_auroc=aucs)).to_csv(
    os.path.join(OUT, "auroc_vs_k.csv"), index=False)
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write(f"fold{fold} corr {r:+.3f}\n")
    f.write(f"full-map LOO AUROC = {auc_full:.3f}\n")
    f.write(f"k* within {AUROC_TOL} = {k_star} regions, AUROC = {aucs[k_star-1]:.3f}\n")
    f.write(f"sign-consistent locked regions = {len(locked)}, AUROC = {auc_locked:.3f}\n")
print("\nsaved figures + tables to", OUT)
