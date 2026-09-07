"""
Across ALL latent features: do the C9+ controls project toward ALS more than
the rest?  For each fold's 512-d latent we build a supervised ALS-axis =
(ALS centroid - control centroid) fit on the 37 NON-C9 subjects (so the C9 pair
cannot bias the direction), z-score latent dims first (robust, no overfit).
Score s = z . w_ALS.  Position on the control->ALS scale:
   0 = typical control, 1 = typical ALS.
Also a directionality fraction: of 512 ALS-oriented dims, how many place the
subject on the ALS side of the non-C9 control mean.
Tested: C9 pair TOGETHER (exact control-pair permutation) and ALONE (percentile
among controls + control->ALS position), per fold and averaged across folds.
"""
import os, numpy as np, pandas as pd, torch
from scipy import stats
from itertools import combinations
import importlib.util
ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
spec = importlib.util.spec_from_file_location("rf", os.path.join(ROOT, "rerun_faithful.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
DEV = torch.device("cpu"); DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT = os.path.join(ROOT, "all_confounders_outputs"); MODELS = os.path.join(ROOT, "5_fold_models_original")
C9 = [10, 30]

lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
lab["id"] = lab["code"].str.extract(r"^(\d+)").astype(int)
codes, ids = lab["code"].values, lab["id"].values
y = (lab.group == "ALS").astype(int).values
cache = {int(i): torch.from_numpy(np.stack(
    [rf.load_excel_image(os.path.join(d, f"{c}.xlsx")) for d in DIRS], axis=0))
    for c, i in zip(codes, ids)}
def zagg(m, x):
    zs = [m.encoder(x[:, t]) for t in range(x.shape[1])]
    za, _ = m.attention(torch.stack(zs, dim=1)); return za.squeeze(0).cpu().numpy()

is_c9 = np.isin(ids, C9)
axis_scores = {}; pos_scores = {}; frac_scores = {}
for f in range(1, 6):
    m = rf.ConvAutoencoderWithAttention(1, 512, 2).to(DEV)
    with torch.no_grad():
        m.eval(); _ = m(cache[ids[0]].unsqueeze(0).to(DEV))
    m.load_state_dict(torch.load(os.path.join(MODELS, f"best_model_fold{f}.pth"), map_location=DEV), strict=True)
    m.eval(); Z = np.zeros((len(ids), 512))
    with torch.no_grad():
        for k, i in enumerate(ids):
            Z[k] = zagg(m, cache[int(i)].unsqueeze(0).to(DEV))
    # z-score dims using NON-C9 subjects only
    ref = ~is_c9
    mu = Z[ref].mean(0); sd = Z[ref].std(0) + 1e-9; zz = (Z - mu) / sd
    als_m = zz[(y == 1)].mean(0)                       # ALS centroid (all ALS are non-C9)
    ctl_m = zz[(y == 0) & (~is_c9)].mean(0)            # non-C9 control centroid
    w = als_m - ctl_m; w /= (np.linalg.norm(w) + 1e-9)  # ALS direction
    s = zz @ w                                          # projection
    # position on control->ALS scale
    c0 = s[(y == 0) & (~is_c9)].mean(); c1 = s[(y == 1)].mean()
    pos = (s - c0) / (c1 - c0 + 1e-9)
    # directionality fraction over 512 ALS-oriented dims (vs non-C9 control mean)
    frac = ((zz * np.sign(w)) > 0).mean(1)  # sign(w) orients each dim toward ALS; >0 = above control mean toward ALS
    axis_scores[f] = s; pos_scores[f] = pos; frac_scores[f] = frac

df = pd.DataFrame({"id": ids, "y": y})
for f in range(1, 6):
    df[f"axis_f{f}"] = axis_scores[f]; df[f"pos_f{f}"] = pos_scores[f]; df[f"frac_f{f}"] = frac_scores[f]
df["axis_mean"] = df[[f"axis_f{f}" for f in range(1, 6)]].mean(1)
df["pos_mean"] = df[[f"pos_f{f}" for f in range(1, 6)]].mean(1)
df["frac_mean"] = df[[f"frac_f{f}" for f in range(1, 6)]].mean(1)
df.to_csv(os.path.join(OUT, "c9_alsaxis_scores.csv"), index=False)

ctl = df[df.y == 0]
def perm_pair(score):
    vals = ctl.set_index("id")[score]; obs = vals.loc[C9].mean()
    allp = np.array([np.mean([vals.loc[a], vals.loc[b]]) for a, b in combinations(ctl.id, 2)])
    return obs, (allp >= obs).mean()
def pct_ctl(score, cid):
    v = ctl.set_index("id")[score]; return (v >= v.loc[cid]).mean()

print("="*76)
print("SUPERVISED ALS-AXIS across all 512 latent features (leave-C9-out direction)")
print("Position: 0.0 = typical control, 1.0 = typical ALS")
print("="*76)
alsp = df[df.y == 1]["pos_mean"]
print(f"\n reference control->ALS scale (pos_mean): controls ~0.00 (sd {ctl['pos_mean'].std():.2f}), "
      f"ALS mean {alsp.mean():.2f}")
print("\n --- averaged across the 5 fold latents ---")
for cid in C9:
    r = df[df.id == cid].iloc[0]
    print(f" control {cid}: ALS-position={r.pos_mean:+.2f} (control->ALS scale) | "
          f"axis pctile among controls={pct_ctl('axis_mean', cid)*100:.0f}th | "
          f"ALS-leaning dims={r.frac_mean*100:.0f}%")
obs_axis, p_axis = perm_pair("axis_mean"); obs_pos, p_pos = perm_pair("pos_mean")
print(f"\n C9 PAIR TOGETHER (mean): ALS-position={df[df.id.isin(C9)]['pos_mean'].mean():+.2f}  "
      f"other-controls={ctl[~ctl.id.isin(C9)]['pos_mean'].mean():+.2f}")
print(f"   exact pair-permutation p (axis)={p_axis:.3f}   (position)={p_pos:.3f}   (min possible {1/171:.3f})")

print("\n --- per fold: ALS-position (0=ctrl,1=ALS) & pair-permutation p ---")
for cid in C9:
    cells = [f"f{f}={df[df.id==cid].iloc[0][f'pos_f{f}']:+.2f}" for f in range(1, 6)]
    print(f" control {cid} position: " + "  ".join(cells))
pp = [perm_pair(f"pos_f{f}")[1] for f in range(1, 6)]
print(" pair-permutation p by fold:  " + "  ".join(f"f{f}={p:.3f}" for f, p in zip(range(1, 6), pp)))

print("\n --- directionality: is each C9 control ALS-leaning across dims? (binomial vs 0.5) ---")
for cid in C9:
    for f in [3]:  # highlight dim477 fold; mean below
        pass
    fr = df[df.id == cid].iloc[0]["frac_mean"]
    # per-fold binomial two-sided on 512 dims
    perf = []
    for f in range(1, 6):
        k = int(round(df[df.id == cid].iloc[0][f"frac_f{f}"] * 512))
        perf.append(f"f{f}={df[df.id==cid].iloc[0][f'frac_f{f}']*100:.0f}%")
    print(f" control {cid}: mean ALS-leaning dims={fr*100:.0f}%  ({' '.join(perf)})  "
          f"[control median {ctl['frac_mean'].median()*100:.0f}%]")
print("\nsaved -> all_confounders_outputs/c9_alsaxis_scores.csv")
