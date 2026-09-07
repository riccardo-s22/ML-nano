"""
Are the C9orf72+ controls (10, 30) significantly more ALS-like than other
controls, ACROSS latent features?

Two ALS-ness readouts per fold latent representation (5 folds):
  - P_ALS       : the fold classifier's ALS probability (operates on z_agg)
  - latent_comp : ALS-composite = mean over top-50 group-discriminative latent
                  dims of the z-scored, ALS-oriented latent
Tested as a GROUP (2 C9 vs 17 other controls) and INDIVIDUALLY, per fold and
pooled across folds. Group test uses an exact permutation over control pairs
(n=2 is tiny -> report exact p, not asymptotic).
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

recs = {"id": ids, "y": y}
for f in range(1, 6):
    m = rf.ConvAutoencoderWithAttention(1, 512, 2).to(DEV)
    with torch.no_grad():
        m.eval(); _ = m(cache[ids[0]].unsqueeze(0).to(DEV))
    m.load_state_dict(torch.load(os.path.join(MODELS, f"best_model_fold{f}.pth"), map_location=DEV), strict=True)
    m.eval(); P = np.zeros(len(ids)); Zf = np.zeros((len(ids), 512))
    with torch.no_grad():
        for k, i in enumerate(ids):
            x = cache[int(i)].unsqueeze(0).to(DEV)
            _, lg, _ = m(x); P[k] = 1.0 - torch.softmax(lg, 1)[0, 1].item(); Zf[k] = zagg(m, x)
    r = np.array([stats.pointbiserialr(y, Zf[:, d])[0] for d in range(512)])
    orient = np.sign(r); orient[orient == 0] = 1
    top = np.argsort(-np.abs(r))[:50]
    zz = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-9)
    comp = (zz[:, top] * orient[top]).mean(1)
    recs[f"P_ALS_f{f}"] = P; recs[f"comp_f{f}"] = comp
df = pd.DataFrame(recs)
df["P_ALS_mean"] = df[[f"P_ALS_f{f}" for f in range(1, 6)]].mean(1)
df["comp_mean"] = df[[f"comp_f{f}" for f in range(1, 6)]].mean(1)
df.to_csv(os.path.join(OUT, "c9_significance_scores.csv"), index=False)

ctl = df[df.y == 0].reset_index(drop=True)
oth = ctl[~ctl.id.isin(C9)]; c9 = ctl[ctl.id.isin(C9)]

def perm_group(score):
    """exact p: P(mean of a random control-pair >= mean of the 2 C9 controls)."""
    vals = ctl.set_index("id")[score]
    obs = vals.loc[C9].mean()
    allpairs = [np.mean([vals.loc[a], vals.loc[b]]) for a, b in combinations(ctl.id, 2)]
    allpairs = np.array(allpairs)
    return obs, (allpairs >= obs).mean()

def perm_indiv(score, cid):
    """P(a random control's score >= this control's)."""
    v = ctl.set_index("id")[score]
    return (v >= v.loc[cid]).mean()

print("="*74)
print("GROUP TEST: 2 C9 controls vs 17 other controls  (higher = more ALS-like)")
print("="*74)
for score, name in [("P_ALS_mean", "classifier P_ALS (mean over 5 folds)"),
                    ("comp_mean", "latent ALS-composite (mean over 5 folds)")]:
    mwu = stats.mannwhitneyu(c9[score], oth[score], alternative="greater").pvalue
    obs, pp = perm_group(score)
    print(f"\n {name}:")
    print(f"   C9 mean={c9[score].mean():+.3f}  other-controls mean={oth[score].mean():+.3f}  ALS mean={df[df.y==1][score].mean():+.3f}")
    print(f"   MWU one-sided p={mwu:.3f}   exact pair-permutation p={pp:.3f}  (min possible p={1/171:.3f})")

print("\n per-fold GROUP permutation p (C9 pair vs random control pairs):")
for score_base, nm in [("P_ALS", "P_ALS"), ("comp", "latent-composite")]:
    ps = []
    for f in range(1, 6):
        _, pp = perm_group(f"{score_base}_f{f}"); ps.append(pp)
    print(f"   {nm:16s}: " + "  ".join(f"f{f}={p:.3f}" for f, p in zip(range(1, 6), ps)))

print("\n" + "="*74)
print("INDIVIDUAL: each C9 control vs the 19-control distribution")
print("="*74)
for cid in C9:
    print(f"\n control {cid}:")
    for score_base, nm in [("P_ALS", "P_ALS"), ("comp", "latent-composite")]:
        cells = []
        for f in range(1, 6):
            p = perm_indiv(f"{score_base}_f{f}", cid); cells.append(f"f{f}p={p:.2f}")
        pm = perm_indiv(f"{score_base}_mean", cid)
        print(f"   {nm:16s}: " + " ".join(cells) + f"  | mean-across-folds p={pm:.3f}")
print("\nsaved -> all_confounders_outputs/c9_significance_scores.csv")
