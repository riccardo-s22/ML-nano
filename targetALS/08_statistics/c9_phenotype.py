"""
Do the two C9orf72+ controls (codes 10, 30) show an ALS phenotype in the models?
Check ALS-probability ACROSS the 5 fold classifiers, and ALS-position ACROSS
latent features (not just dim477).
"""
import os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy import stats
import importlib.util
ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
spec = importlib.util.spec_from_file_location("rf", os.path.join(ROOT, "rerun_faithful.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
DEV = torch.device("cpu"); DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT = os.path.join(ROOT, "all_confounders_outputs"); os.makedirs(OUT, exist_ok=True)
MODELS = os.path.join(ROOT, "5_fold_models_original")
C9 = [10, 30]

lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
lab["id"] = lab["code"].str.extract(r"^(\d+)").astype(int)
codes = lab["code"].values; ids = lab["id"].values
y_als = (lab.group == "ALS").astype(int).values  # 1=ALS
# which fold held out each subject (from existing OOF)
oof = pd.read_csv(os.path.join(ROOT, "model_metrics_eval", "existing_models_oof_predictions.csv"))
oof["id"] = oof["code"].str.extract(r"^(\d+)").astype(int)
held = dict(zip(oof.id, oof.held_out_fold))

# preload
cache = {int(i): torch.from_numpy(np.stack(
    [rf.load_excel_image(os.path.join(d, f"{c}.xlsx")) for d in DIRS], axis=0))
    for c, i in zip(codes, ids)}

def zagg(model, x):
    zs = [model.encoder(x[:, t]) for t in range(x.shape[1])]
    za, _ = model.attention(torch.stack(zs, dim=1)); return za.squeeze(0).cpu().numpy()

# ---- P_ALS from every fold model for every subject ----
Pmat = np.zeros((len(ids), 5)); Z = {}   # Z[fold] = (n,512)
for f in range(1, 6):
    m = rf.ConvAutoencoderWithAttention(1, 512, 2).to(DEV)
    with torch.no_grad():
        m.eval(); _ = m(cache[ids[0]].unsqueeze(0).to(DEV))
    m.load_state_dict(torch.load(os.path.join(MODELS, f"best_model_fold{f}.pth"), map_location=DEV), strict=True)
    m.eval(); zf = np.zeros((len(ids), 512))
    with torch.no_grad():
        for k, i in enumerate(ids):
            x = cache[int(i)].unsqueeze(0).to(DEV)
            _, logits, _ = m(x)
            Pmat[k, f - 1] = 1.0 - torch.softmax(logits, 1)[0, 1].item()  # P(ALS)
            zf[k] = zagg(m, x)
    Z[f] = zf
df = pd.DataFrame({"id": ids, "y": y_als});
for f in range(5): df[f"P_ALS_fold{f+1}"] = Pmat[:, f]
df["P_ALS_mean5"] = Pmat.mean(1)

print("="*72)
print("ACROSS MODELS: P(ALS) from all 5 fold classifiers  (marking OOF fold *)")
print("="*72)
ctl = df[df.y == 0]
for cc in C9:
    r = df[df.id == cc].iloc[0]
    hv = held.get(cc)
    cells = []
    for f in range(1, 6):
        star = "*" if f == hv else " "
        cells.append(f"f{f}={r[f'P_ALS_fold{f}']:.2f}{star}")
    # percentile of mean5 among controls
    pct = (ctl.P_ALS_mean5 < r.P_ALS_mean5).mean() * 100
    print(f" control {cc}: " + " ".join(cells) +
          f"  | mean={r.P_ALS_mean5:.2f} ({pct:.0f}th pct of controls)  (*=held-out/OOF)")
print(f"\n control-group P(ALS) mean5: median={ctl.P_ALS_mean5.median():.2f}  "
      f"90th pct={ctl.P_ALS_mean5.quantile(.9):.2f} ;  ALS-group median={df[df.y==1].P_ALS_mean5.median():.2f}")
n_als_calls = {cc: int((df[df.id==cc].iloc[0][[f'P_ALS_fold{f}' for f in range(1,6)]] >= 0.5).sum()) for cc in C9}
print(" #folds calling them ALS (P>=0.5):", n_als_calls)

print("\n" + "="*72)
print("ACROSS LATENT FEATURES: position on ALS-discriminative dims (per fold)")
print("="*72)
# For each fold latent: build composite ALS-score = mean over top-K discriminative
# dims of the z-scored, ALS-oriented latent; report control percentile.
for f in range(1, 6):
    zf = Z[f]; y = y_als
    mu = zf.mean(0); sd = zf.std(0) + 1e-9
    zz = (zf - mu) / sd
    # discrimination per dim (point-biserial), orient so + = ALS
    r = np.array([stats.pointbiserialr(y, zf[:, k])[0] for k in range(512)])
    orient = np.sign(r); orient[orient == 0] = 1
    K = 50; top = np.argsort(-np.abs(r))[:K]
    comp = (zz[:, top] * orient[top]).mean(1)  # composite ALS-ness per subject
    cser = pd.Series(comp)
    cmask = y == 0
    for cc in C9:
        idx = np.where(ids == cc)[0][0]
        pct = (comp[cmask] < comp[idx]).mean() * 100
        # also: fraction of top-K dims where subject is on ALS side of control mean
        ctl_mean = zz[cmask][:, top].mean(0)
        on_als = ((zz[idx, top] * orient[top]) > (ctl_mean * orient[top])).mean() * 100
        tag = " [dim477 space]" if f == 3 else ""
        print(f" fold{f}{tag} control {cc}: latent ALS-composite {pct:5.0f}th pct of controls | "
              f"on ALS-side of {on_als:.0f}% of top-{K} dims")
    if f == 3: print("   (fold3 = canonical dim477 latent)")

df.to_csv(os.path.join(OUT, "c9_across_models.csv"), index=False)
print("\nsaved -> all_confounders_outputs/c9_across_models.csv")
