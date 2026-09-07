"""
WHY does fold3 (uniquely) flag C9 control 10 as ALS-like?
For control 10 across all 5 folds (+ control 30 and a typical control as refs):
  1) attention weights over timepoints (0h/6h/24h)
  2) latent nearest-neighbours (is 10 among ALS in fold3?)
  3) fold3 per-dim classifier attribution of the ALS margin (grad x (z-ctrl_mean)):
     which latent dims push 10 toward ALS, is dim477 dominant, is 10 an outlier there
  4) input saliency per timepoint (which EEM/timepoint drives the ALS call), fold3 vs
     a fold where 10 is control-like
"""
import os, numpy as np, pandas as pd, torch
import importlib.util
ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
spec = importlib.util.spec_from_file_location("rf", os.path.join(ROOT, "rerun_faithful.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
DEV = torch.device("cpu"); DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT = os.path.join(ROOT, "all_confounders_outputs"); MODELS = os.path.join(ROOT, "5_fold_models_original")
TPS = ["0h", "6h", "24h"]

lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
lab["id"] = lab["code"].str.extract(r"^(\d+)").astype(int)
ids = lab["id"].values; y = (lab.group == "ALS").astype(int).values
cache = {int(i): torch.from_numpy(np.stack(
    [rf.load_excel_image(os.path.join(d, f"{c}.xlsx")) for d in DIRS], axis=0))
    for c, i in zip(lab["code"].values, ids)}
idx = {int(i): k for k, i in enumerate(ids)}

def encode(m, x):  # per-timepoint latents (3,512), z_agg (512), attn (3)
    zs = torch.stack([m.encoder(x[:, t]) for t in range(x.shape[1])], dim=1)  # (1,3,512)
    za, aw = m.attention(zs)
    return zs.squeeze(0).detach().cpu().numpy(), za.squeeze(0).detach().cpu().numpy(), aw.squeeze(0).detach().cpu().numpy()

def margin_from_z(m, z_agg_t):  # ALS margin logit0-logit1 from z_agg tensor
    lg = m.classifier(z_agg_t); return lg[..., 0] - lg[..., 1]

REF = 10  # main subject
# pick a typical control (median existing OOF P_ALS) as comparison
typ = 12
print("subjects: control 10 (C9+, the fold3-ALS one), control 30 (C9+), control 12 (typical)")

Zagg = {}; ATTN = {}; PALS = {}
for f in range(1, 6):
    m = rf.ConvAutoencoderWithAttention(1, 512, 2).to(DEV)
    with torch.no_grad():
        m.eval(); _ = m(cache[ids[0]].unsqueeze(0).to(DEV))
    m.load_state_dict(torch.load(os.path.join(MODELS, f"best_model_fold{f}.pth"), map_location=DEV), strict=True)
    m.eval()
    Za = np.zeros((len(ids), 512)); At = {}
    with torch.no_grad():
        for i in ids:
            _, za, aw = encode(m, cache[int(i)].unsqueeze(0).to(DEV))
            Za[idx[int(i)]] = za; At[int(i)] = aw
    Zagg[f] = Za; ATTN[f] = At
    with torch.no_grad():
        _, lg, _ = m(cache[REF].unsqueeze(0).to(DEV))
    PALS[f] = (1 - torch.softmax(lg, 1)[0, 1]).item()

print("\n" + "="*70)
print("1) ATTENTION over timepoints (0h/6h/24h)  + P_ALS, per fold")
print("="*70)
for f in range(1, 6):
    a10 = ATTN[f][10]; a30 = ATTN[f][30]; a12 = ATTN[f][typ]
    tag = "  <-- ALS call" if PALS[f] >= 0.5 else ""
    print(f" fold{f}: P_ALS(10)={PALS[f]:.2f}{tag}")
    print(f"    attn 10:[{a10[0]:.2f} {a10[1]:.2f} {a10[2]:.2f}]  "
          f"30:[{a30[0]:.2f} {a30[1]:.2f} {a30[2]:.2f}]  "
          f"12:[{a12[0]:.2f} {a12[1]:.2f} {a12[2]:.2f}]  (0h/6h/24h)")

print("\n" + "="*70)
print("2) LATENT NEAREST NEIGHBOURS of control 10 (euclid on z-scored z_agg)")
print("="*70)
for f in range(1, 6):
    Z = Zagg[f]; zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    d = np.linalg.norm(zz - zz[idx[REF]], axis=1); order = np.argsort(d)
    nn = [o for o in order if o != idx[REF]][:5]
    labs = [("ALS" if y[o] else "ctrl") + f"{ids[o]}" for o in nn]
    nals = sum(y[o] for o in nn)
    print(f" fold{f}: {nals}/5 ALS neighbours -> {labs}")

print("\n" + "="*70)
print("3) FOLD3 per-dim attribution of the ALS margin for control 10")
print("   contribution_dim = grad(margin)/dz * (z - control_mean);  + => pushes ALS")
print("="*70)
m = rf.ConvAutoencoderWithAttention(1, 512, 2).to(DEV)
with torch.no_grad():
    m.eval(); _ = m(cache[ids[0]].unsqueeze(0).to(DEV))
m.load_state_dict(torch.load(os.path.join(MODELS, "best_model_fold3.pth"), map_location=DEV), strict=True); m.eval()
Z3 = Zagg[3]; ctrl_mean = Z3[y == 0].mean(0)
z10 = torch.tensor(Z3[idx[REF]], dtype=torch.float32, requires_grad=True)
margin = margin_from_z(m, z10.unsqueeze(0))[0]; margin.backward()
grad = z10.grad.numpy(); contrib = grad * (Z3[idx[REF]] - ctrl_mean)
top = np.argsort(-contrib)[:10]
print(f" fold3 ALS margin(10)={margin.item():+.2f}; total attribution={contrib.sum():+.2f}; "
      f"dim477 contribution={contrib[477]:+.3f} (rank {int((contrib> contrib[477]).sum())+1}/512)")
print(f" top-10 ALS-pushing dims:")
for dcol in top:
    ctl_vals = Z3[y == 0][:, dcol]; pct = (Z3[y == 0][:, dcol] < Z3[idx[REF], dcol]).mean() * 100
    als_side = "ALS-side" if abs(Z3[idx[REF], dcol] - Z3[y==1][:, dcol].mean()) < abs(Z3[idx[REF], dcol] - ctrl_mean[dcol]) else "ctrl-side"
    print(f"   dim{dcol:3d}: contrib={contrib[dcol]:+.3f}  z10={Z3[idx[REF],dcol]:+.2f}  "
          f"(ctrl {pct:.0f}th pct; {als_side}){'  <-- dim477' if dcol==477 else ''}")
share = contrib[top].sum() / contrib[contrib > 0].sum() * 100
print(f" top-10 dims carry {share:.0f}% of all positive (ALS-pushing) attribution; "
      f"dim477 alone = {contrib[477]/contrib[contrib>0].sum()*100:.0f}%")

print("\n" + "="*70)
print("4) INPUT SALIENCY per timepoint: |d margin / d x_t| (fold3 vs fold1)")
print("="*70)
for f in [3, 1]:
    m = rf.ConvAutoencoderWithAttention(1, 512, 2).to(DEV)
    with torch.no_grad():
        m.eval(); _ = m(cache[ids[0]].unsqueeze(0).to(DEV))
    m.load_state_dict(torch.load(os.path.join(MODELS, f"best_model_fold{f}.pth"), map_location=DEV), strict=True); m.eval()
    x = cache[REF].clone().unsqueeze(0).to(DEV).requires_grad_(True)
    zs = torch.stack([m.encoder(x[:, t]) for t in range(3)], dim=1)
    za, _ = m.attention(zs); mg = margin_from_z(m, za)[0]; mg.backward()
    g = x.grad.abs().squeeze(0).squeeze(1).numpy()  # (3,H,W)
    per_tp = g.reshape(3, -1).sum(1); per_tp = per_tp / per_tp.sum()
    # locate peak EEM pixel per timepoint (rows=emission 512, cols=excitation 71)
    print(f" fold{f} (P_ALS={PALS[f]:.2f}): timepoint saliency 0h/6h/24h = "
          f"[{per_tp[0]:.2f} {per_tp[1]:.2f} {per_tp[2]:.2f}]")
    dom = per_tp.argmax(); peak = np.unravel_index(g[dom].argmax(), g[dom].shape)
    print(f"    dominant tp={TPS[dom]}, peak at emission-row {peak[0]}, excitation-col {peak[1]}")

np.savez(os.path.join(OUT, "c9_fold3_why.npz"), **{f"zagg_f{f}": Zagg[f] for f in range(1, 6)}, y=y, ids=ids)
print("\nsaved -> all_confounders_outputs/c9_fold3_why.npz")
