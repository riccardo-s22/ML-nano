"""
PILOT v2 — region reduction of the 24h dim477 component, revised criteria.

Implements the three grant revisions:
 (1) DROP the "within 0.02 AUROC" stopping rule. Replace with BOTH:
      1a. Primary: rank/stop by cumulative fraction of the *interindividual
          variance of the 24h dim477 component* that the regions reconstruct
          (label-free; this is literally "explain variation in dim477").
      1b. Secondary QC: paired-bootstrap NON-INFERIORITY of held-out AUROC
          (reduced vs complete 24h map), margin = 0.05, 95% CI.
 (2) Report the region count EMPIRICALLY at 80/90/95% reconstruction; do not
     promise a small contiguous set.
 (3) Quantify FIDELITY of the linear signed-weight proxy to the true (nonlinear,
     all-timepoint) dim477 = z_agg[477]. Report R^2; classification is QC only.

Expensive arrays (EEM stacks, saliency map, true dim477) are cached to npz so the
analysis/figures iterate fast.
"""
import os, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PLS  = os.path.join(os.path.dirname(ROOT), "PLS")
OUT  = os.path.join(ROOT, "dim477_region_pilot_outputs"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "pilot_cache.npz")
DIM, T24 = 477, 2
N_EX_TILES, N_EM_TILES = 6, 8
VAR_THRESHOLDS = [0.80, 0.90, 0.95]
NI_MARGIN = 0.05          # non-inferiority margin on AUROC
B_BOOT = 3000
rng = np.random.default_rng(0)

# ---------------- load or build cache ----------------
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    Wmap, x24, z477 = z["Wmap"], z["x24"], z["z477"]
    y, ex_axis, em_axis = z["y"], z["ex_axis"], z["em_axis"]
    codes = list(z["codes"]); fold = int(z["fold"]); rcorr = float(z["rcorr"])
    print(f"loaded cache | fold{fold} corr {rcorr:+.3f} | x24 {x24.shape}")
else:
    import torch
    from scipy.stats import pearsonr
    DEV = torch.device("cpu")
    TPD = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
    spec = importlib.util.spec_from_file_location("sal", os.path.join(PLS, "dim35_chirality_saliency.py"))
    sal = importlib.util.module_from_spec(spec); spec.loader.exec_module(sal)
    lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
    codes = lab["code"].astype(str).tolist()
    groups = dict(zip(lab["code"].astype(str), lab["group"].astype(str)))
    y = np.array([1 if groups[c].strip().upper() == "ALS" else 0 for c in codes])
    em_axis, ex_axis = sal.get_wavelength_axes(TPD[0], codes[0])
    ds = sal.ExcelImageDataset(codes, [0]*len(codes), TPD)
    X = torch.stack([ds[i][0] for i in range(len(codes))], 0).float()
    N, Tn, _, H, Wd = X.shape
    def build_model(path):
        m = sal.ConvAutoencoderWithAttention(in_channels=1, latent_dim=512, num_classes=2).to(DEV)
        with torch.no_grad(): _ = m(X[:1].to(DEV))
        m.load_state_dict(torch.load(path, map_location=DEV), strict=False); m.eval(); return m
    def zagg_dim(m, dim):
        out = []
        with torch.no_grad():
            for i in range(N):
                lat = [m.encoder(X[i:i+1, t].to(DEV)) for t in range(Tn)]
                zz, _ = m.attention(torch.stack(lat, 1)); out.append(zz[0, dim].item())
        return np.array(out)
    canon = pd.read_csv(os.path.join(PLS, "z_agg3.csv"))[["code", "dim477"]]
    canon = dict(zip(canon["code"], canon["dim477"]))
    cvec = np.array([canon.get(c, np.nan) for c in codes]); ok = ~np.isnan(cvec)
    best = None
    for k in range(1, 6):
        m = build_model(os.path.join(ROOT, f"5_fold_models_original/best_model_fold{k}.pth"))
        zz = zagg_dim(m, DIM); rr = pearsonr(zz[ok], cvec[ok])[0]
        print(f"  fold{k}: corr {rr:+.3f}")
        if best is None or abs(rr) > abs(best[1]): best = (k, rr, m, zz)
    fold, rcorr, model, z477 = best
    sign = 1.0 if rcorr >= 0 else -1.0
    def saliency_24h(idx):
        x = X[idx:idx+1].clone().to(DEV).requires_grad_(True)
        lat = [model.encoder(x[:, t]) for t in range(Tn)]
        zz, _ = model.attention(torch.stack(lat, 1)); (zz[0, DIM] * sign).backward()
        return x.grad.detach()[0, T24, 0].numpy()
    Wmap = np.stack([saliency_24h(i) for i in range(N)], 0).mean(0)
    x24 = X[:, T24, 0].numpy()
    np.savez(CACHE, Wmap=Wmap, x24=x24, z477=np.asarray(z477), y=y,
             ex_axis=ex_axis, em_axis=em_axis, codes=np.array(codes, object),
             fold=fold, rcorr=rcorr)
    print(f"built + cached | fold{fold} corr {rcorr:+.3f}")

N, H, Wd = x24.shape
mean_x24 = x24.mean(0)

# ---------------- tiling + regional scores ----------------
ex_edges = np.linspace(0, Wd, N_EX_TILES + 1).astype(int)
em_edges = np.linspace(0, H,  N_EM_TILES + 1).astype(int)
tiles = []
for ei in range(N_EX_TILES):
    for mj in range(N_EM_TILES):
        tiles.append(dict(r0=em_edges[mj], r1=em_edges[mj+1], c0=ex_edges[ei], c1=ex_edges[ei+1]))
tiles = pd.DataFrame(tiles)
nT = len(tiles)
# S[i,j] = projection of participant i's normalized 24h spectrum onto tile j, signed-W weights
S = np.zeros((N, nT))
for j, t in tiles.iterrows():
    S[:, j] = (Wmap[t.r0:t.r1, t.c0:t.c1] * x24[:, t.r0:t.r1, t.c0:t.c1]).reshape(N, -1).sum(1)
s_full = S.sum(1)                                   # complete 24h dim477 linear map (per participant)

# ---------------- (1a) rank by variance contribution to the 24h component ----------------
sf_c = s_full - s_full.mean()
cov_j = (S - S.mean(0)) .T @ sf_c / (N - 1)         # Cov(tile_j, s_full)
order = np.argsort(-cov_j)                          # descending variance contribution
var_full = sf_c.var(ddof=1)
# cumulative reconstruction R^2 of the 24h component: 1 - Var(residual)/Var(full)
R2_recon = np.array([1 - (s_full - S[:, order[:k]].sum(1)).var(ddof=1) / var_full
                     for k in range(1, nT + 1)])
k_at = {th: int(np.argmax(R2_recon >= th) + 1) for th in VAR_THRESHOLDS}
print("\n(2) regions to reconstruct the 24h dim477 component:")
for th in VAR_THRESHOLDS:
    print(f"    >={int(th*100)}% : {k_at[th]} of {nT} regions")

# ---------------- (3) fidelity of linear 24h proxy to true nonlinear dim477 ----------------
def r2(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b)**2 / ((a @ a) * (b @ b)))
fid_full = r2(s_full, z477)                         # how much of true dim477 the linear 24h proxy explains
# cumulative fidelity of top-k regions to true dim477
fid_k = np.array([r2(S[:, order[:k]].sum(1), z477) for k in range(1, nT + 1)])
print(f"\n(3) linear 24h signed-weight proxy vs TRUE dim477 (nonlinear, all TP): R^2 = {fid_full:.3f}")

# ---------------- classification QC (secondary): LOO AUROC ----------------
def loo_prob(feat):
    if feat.ndim == 1: feat = feat[:, None]
    p = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(C=0.5, max_iter=1000).fit(sc.transform(feat[tr]), y[tr])
        p[i] = clf.predict_proba(sc.transform(feat[i:i+1]))[0, 1]
    return p
p_full = loo_prob(s_full)
auc_dim477 = roc_auc_score(y, z477 if roc_auc_score(y, z477) >= 0.5 else -z477)  # nonlinear ceiling
auc_full = roc_auc_score(y, p_full)
p_k = [loo_prob(S[:, order[:k]]) for k in range(1, nT + 1)]
auc_k = np.array([roc_auc_score(y, p) for p in p_k])

# ---------------- (1b) paired-bootstrap non-inferiority of AUROC (reduced vs full) ----------------
als_idx = np.where(y == 1)[0]; ctrl_idx = np.where(y == 0)[0]
def boot_delta(pk):
    d = np.empty(B_BOOT)
    for b in range(B_BOOT):
        ii = np.concatenate([rng.choice(als_idx, als_idx.size, True),
                             rng.choice(ctrl_idx, ctrl_idx.size, True)])
        yy = y[ii]
        d[b] = roc_auc_score(yy, pk[ii]) - roc_auc_score(yy, p_full[ii])
    return d
d_lo = np.empty(nT); d_hi = np.empty(nT); d_med = np.empty(nT)
for k in range(nT):
    d = boot_delta(p_k[k])
    d_lo[k], d_med[k], d_hi[k] = np.percentile(d, [2.5, 50, 97.5])
non_inf = np.where(d_lo > -NI_MARGIN)[0]
k_ni = int(non_inf[0] + 1) if len(non_inf) else nT
print(f"\n(1b) paired-bootstrap non-inferiority (margin {NI_MARGIN}): first k = {k_ni} regions")

# ---------------- locked set at 90% reconstruction; sign-consistency retention ----------------
k_lock = k_at[0.90]
locked = list(order[:k_lock])
def sign_consistent(col, thr=0.70):
    ok = True
    for g in (0, 1):
        v = col[y == g]; mu = np.sign(v.mean()) or 1
        ok &= (np.mean(np.sign(v) == mu) >= thr)
    return ok
sign_ok = {j: sign_consistent(S[:, j]) for j in locked}
locked_signed = [j for j in locked if sign_ok[j]] or locked

# chirality labels
chir = []
with open(os.path.join(ROOT, "Coordinates_DNA.txt")) as f:
    nm = em = ex = None
    for ln in f:
        ln = ln.strip()
        if ln.startswith("["): nm = ln.strip("[]")
        elif ln.startswith("Emission"): em = float(ln.split("=")[1])
        elif ln.startswith("Excitation"): ex = float(ln.split("=")[1]); chir.append((nm, ex, em))
def label(j):
    t = tiles.loc[j]
    lo_ex, hi_ex = sorted([ex_axis[t.c0], ex_axis[min(int(t.c1), Wd)-1]])
    lo_em, hi_em = sorted([em_axis[t.r0], em_axis[min(int(t.r1), H)-1]])
    ins = [n for n, ex, em in chir if lo_ex <= ex <= hi_ex and lo_em <= em <= hi_em]
    return ",".join(f"({n})" for n in ins) if ins else "-", (lo_ex, hi_ex, lo_em, hi_em)

rows = []
for rank_pos, j in enumerate(order[:k_lock]):
    lab_s, (lex, hex_, lem, hem) = label(j)
    rows.append(dict(rank=rank_pos+1, ex_nm=f"{lex:.0f}-{hex_:.0f}", em_nm=f"{lem:.0f}-{hem:.0f}",
                     chirality=lab_s, var_share=round(cov_j[j]/var_full, 4),
                     cum_recon=round(R2_recon[rank_pos], 3),
                     sign="+" if cov_j[j] > 0 else "-", sign_ok=bool(sign_ok[j])))
locked_df = pd.DataFrame(rows)
locked_df.to_csv(os.path.join(OUT, "locked_regions_v2.csv"), index=False)
print("\n=== locked regions (90% reconstruction) ===\n" + locked_df.to_string(index=False))

# ==================== FIGURES ====================
FOLD_C, POOL_C, ACC = "#4b5bd4", "#0e8a8f", "crimson"
extent = [ex_axis.min(), ex_axis.max(), em_axis.min(), em_axis.max()]
kk = np.arange(1, nT + 1)

# Fig A: primary selection — cumulative reconstruction of 24h dim477 variance
fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=170)
ax.plot(kk, R2_recon, "-o", color=FOLD_C, ms=4)
for th in VAR_THRESHOLDS:
    ax.axhline(th, color="grey", ls=":", lw=1)
    ax.plot(k_at[th], R2_recon[k_at[th]-1], "s", color=ACC, ms=9)
    ax.annotate(f"{int(th*100)}% → {k_at[th]} regions", (k_at[th], th),
                textcoords="offset points", xytext=(8, -12), fontsize=9, color=ACC)
ax.set_xlabel("number of regions (ranked by variance contribution)")
ax.set_ylabel("cumulative variance of 24h dim477\ncomponent reconstructed  (R²)")
ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
ax.set_title("(1a/2) Primary criterion: reconstruct the 24h dim477 component", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figA_variance_reconstruction.png")); plt.close(fig)

# Fig B: (3) fidelity of linear proxy to true dim477
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=170, gridspec_kw={"width_ratios":[1,1.15]})
b1.scatter(z477, s_full, c=np.where(y==1, ACC, POOL_C), s=34, alpha=0.85)
b1.set_xlabel("true dim477  (z_agg[477], nonlinear, all timepoints)")
b1.set_ylabel("linear 24h signed-weight proxy  (s_full)")
b1.set_title(f"(3) proxy fidelity:  R² = {fid_full:.2f}", fontsize=11); b1.grid(alpha=0.25)
b1.scatter([], [], c=ACC, label="ALS"); b1.scatter([], [], c=POOL_C, label="CTRL"); b1.legend(fontsize=8)
b2.plot(kk, fid_k, "-o", color=FOLD_C, ms=4, label="cumulative regions → true dim477")
b2.axhline(fid_full, color=POOL_C, lw=2, ls="--", label=f"full 24h proxy ceiling R²={fid_full:.2f}")
b2.set_xlabel("number of regions"); b2.set_ylabel("R² vs true dim477")
b2.set_ylim(0, max(0.6, fid_full*1.15)); b2.grid(alpha=0.25); b2.legend(fontsize=8, loc="lower right")
b2.set_title("fidelity is capped — linear/24h loses signal", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_fidelity.png")); plt.close(fig)

# Fig C: (1b) paired-bootstrap non-inferiority of AUROC (secondary QC)
fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=170)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(-NI_MARGIN, color=ACC, ls="--", lw=1.5, label=f"non-inferiority margin −{NI_MARGIN}")
ax.fill_between(kk, d_lo, d_hi, color=FOLD_C, alpha=0.2, label="95% paired-bootstrap CI")
ax.plot(kk, d_med, "-o", color=FOLD_C, ms=3.5, label="ΔAUROC (reduced − full)")
if k_ni <= nT:
    ax.axvline(k_ni, color="grey", ls=":", lw=1.2)
    ax.annotate(f"non-inferior from k={k_ni}", (k_ni, -NI_MARGIN),
                textcoords="offset points", xytext=(6, 14), fontsize=9, color="grey")
ax.set_xlabel("number of regions"); ax.set_ylabel("held-out ΔAUROC vs complete 24h map")
ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="lower right")
ax.set_title(f"(1b) Secondary QC: AUROC non-inferiority  "
             f"(full map={auc_full:.2f}, dim477 ceiling={auc_dim477:.2f})", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figC_noninferiority.png")); plt.close(fig)

# Fig D: weight map + locked (sign-consistent) regions
fig, ax = plt.subplots(figsize=(7.2, 6), dpi=170)
vmax = np.abs(Wmap).max()
im = ax.imshow(Wmap, aspect="auto", origin="lower", extent=extent, cmap="seismic", vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed dim477 weight (24h)")
for nm, ex, em in chir:
    ax.plot(ex, em, "kx", ms=7, mew=1.6); ax.text(ex+3, em, f"({nm})", fontsize=7, va="center")
for j in locked_signed:
    t = tiles.loc[j]
    ax.add_patch(Rectangle((ex_axis[t.c0], em_axis[t.r0]),
                 ex_axis[min(int(t.c1),Wd)-1]-ex_axis[t.c0], em_axis[min(int(t.r1),H)-1]-em_axis[t.r0],
                 fill=False, edgecolor="lime", lw=2.4))
ax.set_xlabel("excitation (nm)"); ax.set_ylabel("emission (nm)")
ax.set_title(f"Locked regions: {len(locked_signed)} sign-consistent tiles\n"
             f"(90% of 24h dim477 variance)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figD_weightmap_locked.png")); plt.close(fig)

with open(os.path.join(OUT, "summary_v2.txt"), "w") as f:
    f.write(f"fold{fold} corr {rcorr:+.3f}\n")
    f.write(f"(2) regions to reconstruct 24h dim477 component: "
            + ", ".join(f"{int(th*100)}%={k_at[th]}" for th in VAR_THRESHOLDS) + f" of {nT}\n")
    f.write(f"(3) linear 24h proxy fidelity to true dim477: R^2={fid_full:.3f}\n")
    f.write(f"(1b) AUROC non-inferiority (margin {NI_MARGIN}) from k={k_ni}; "
            f"full-map AUROC={auc_full:.3f}, nonlinear dim477 ceiling={auc_dim477:.3f}\n")
    f.write(f"locked (90%, sign-consistent) = {len(locked_signed)} regions\n")
np.savez(os.path.join(OUT, "curves_v2.npz"), R2_recon=R2_recon, fid_k=fid_k,
         auc_k=auc_k, d_lo=d_lo, d_med=d_med, d_hi=d_hi)
print("\nsaved figures A-D + tables to", OUT)
