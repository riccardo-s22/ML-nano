"""
AGE-MATCHED RETRAIN  -- the rigorous age-leakage test.

Retrain the SAME encoder+attention+classifier (faithful original procedure)
using ONLY the 1:1 nearest-age matched subset (12 ALS + 12 CTRL, age balanced
by construction, MWU p=0.91). If the AE still separates ALS/CTRL out-of-fold
and still recovers a dim477-like discriminative axis, then age imbalance cannot
be what the representation is exploiting -- leakage is excluded by TRAINING-SET
design, not just by post-hoc tests.

Procedure is byte-faithful to rerun_faithful.py (BATCH=4, 100 epochs, no early
stop, best-val-acc checkpoint, encoder.fc & decoder frozen at random init,
StratifiedKFold(5, shuffle, seed42)). Same procedure as the original 39-subject
run, so the ONLY changed variable is the age-balanced training set.
"""
import os, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats

# reuse the faithful harness pieces
import importlib.util
ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
spec = importlib.util.spec_from_file_location("rf", os.path.join(ROOT, "rerun_faithful.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
load_excel_image = rf.load_excel_image
CachedDataset = rf.CachedDataset
train_epoch = rf.train_epoch
validate = rf.validate
ConvAutoencoderWithAttention = rf.ConvAutoencoderWithAttention

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device("cpu"); LATENT_DIM = 512; BATCH_SIZE = 4
NUM_EPOCHS = 100; LR = 1e-3; ALPHA = BETA = 1.0; N_SPLITS = 5
TIMEPOINT_DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT = os.path.join(ROOT, "age_matched_retrain_outputs"); os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "log.txt"), "w")
def p(*a):
    m = " ".join(str(x) for x in a); print(m); log.write(m + "\n"); log.flush()

# ---- reconstruct the exact 1:1 age-matched subset (same greedy caliper-5 rule) ----
meta = pd.read_csv(os.path.join(ROOT, "age_sex_confound_outputs", "merged_dim477_age_sex.csv"))
als = meta[meta.y == 1].sort_values("age"); ctl = meta[meta.y == 0]; used = set(); pairs = []
for _, a in als.iterrows():
    cand = ctl[~ctl.code.isin(used)].copy(); cand["d"] = (cand.age - a.age).abs()
    cand = cand[cand.d <= 5]
    if len(cand):
        c = cand.sort_values("d").iloc[0]; used.add(c.code)
        pairs.append((int(a.code), int(c.code)))
matched_ids = sorted(set([x for pr in pairs for x in pr]))
p(f"matched pairs={len(pairs)}  subjects={len(matched_ids)}  ids={matched_ids}")

# map integer id -> sample_labels code string + label (ALS=0, CTRL=1 as original)
lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
lab["id"] = lab["code"].str.extract(r"^(\d+)").astype(int)
lab = lab[lab.id.isin(matched_ids)].copy()
lab["enc"] = (lab.group == "CTRL").astype(int)  # ALS=0, CTRL=1
codes = lab["code"].values; labels = lab["enc"].values
mage = meta.set_index("code")
lab["age"] = lab.id.map(mage.age); lab["d477"] = lab.id.map(mage.dim477)
p(f"  ALS={int((labels==0).sum())} CTRL={int((labels==1).sum())}  "
  f"age ALS med={lab.age[lab.enc==0].median():.0f} CTRL med={lab.age[lab.enc==1].median():.0f} "
  f"(MWU p={stats.mannwhitneyu(lab.age[lab.enc==0],lab.age[lab.enc==1]).pvalue:.2f})")

# ---- preload EEMs ----
p("preloading EEMs ...")
cache = {}
for c in codes:
    tps = [load_excel_image(os.path.join(d, f"{c}.xlsx")) for d in TIMEPOINT_DIRS]
    cache[c] = torch.from_numpy(np.stack(tps, axis=0))  # (3,1,H,W)

def get_zagg(model, x):  # x: (1,3,1,H,W) -> (512,)
    zs = [model.encoder(x[:, t]) for t in range(x.shape[1])]
    z_agg, _ = model.attention(torch.stack(zs, dim=1))
    return z_agg.squeeze(0).cpu().numpy()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_pal = np.full(len(codes), np.nan); oof_fold = np.full(len(codes), -1, int)
oof_zagg = np.full((len(codes), LATENT_DIM), np.nan, np.float32)
fold_rows = []

for fold, (tr, va) in enumerate(skf.split(codes, labels)):
    p(f"\n=== FOLD {fold+1}/{N_SPLITS}  train={len(tr)} val={len(va)} ===")
    trc, trl = codes[tr], labels[tr]; vac, val_ = codes[va], labels[va]
    train_ds = CachedDataset([cache[c] for c in trc], trl)
    val_ds = CachedDataset([cache[c] for c in vac], val_)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvAutoencoderWithAttention(1, LATENT_DIM, 2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)  # before lazy fc/decoder exist
    with torch.no_grad():
        model.eval(); _ = model(train_ds[0][0].unsqueeze(0).to(DEVICE))
    for prm in model.encoder.fc.parameters(): prm.requires_grad_(False)
    for prm in model.decoder.parameters(): prm.requires_grad_(False)
    model.train()

    best_acc = -1.0; ckpt = os.path.join(OUT, f"best_fold{fold+1}.pth"); t0 = time.time()
    for ep in range(NUM_EPOCHS):
        train_epoch(model, train_loader, optimizer, DEVICE, ALPHA, BETA)
        _, _, _, vaa = validate(model, val_loader, DEVICE, ALPHA, BETA)
        if vaa > best_acc:
            best_acc = vaa; torch.save(model.state_dict(), ckpt)
        if (ep + 1) % 25 == 0:
            p(f"   ep{ep+1:3d} best_va_acc={best_acc:.1f} ({time.time()-t0:.0f}s)")

    # reload best-val-acc checkpoint, score held-out fold
    m2 = ConvAutoencoderWithAttention(1, LATENT_DIM, 2).to(DEVICE)
    with torch.no_grad():
        m2.eval(); _ = m2(next(iter(val_loader))[0].to(DEVICE))
    m2.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True); m2.eval()
    with torch.no_grad():
        for j, gi in enumerate(va):
            x = cache[codes[gi]].unsqueeze(0).to(DEVICE)
            _, logits, _ = m2(x)
            oof_pal[gi] = 1.0 - torch.softmax(logits, 1)[0, 1].item()  # P(ALS)
            oof_fold[gi] = fold + 1
            oof_zagg[gi] = get_zagg(m2, x)
    y_als = 1 - val_; pal = oof_pal[va]
    auc = roc_auc_score(y_als, pal) if len(set(y_als)) > 1 else float("nan")
    acc = accuracy_score(y_als, (pal >= 0.5).astype(int))
    fold_rows.append(dict(fold=fold + 1, n=len(va), best_val_acc=best_acc, acc=acc, auc=auc))
    p(f"  fold{fold+1}: acc={acc:.3f} auc={auc:.3f}")

# ---- OOF summary (positive = ALS) ----
y_als_full = 1 - labels
oof_pred = (oof_pal >= 0.5).astype(int)
oof_auc = roc_auc_score(y_als_full, oof_pal); oof_acc = accuracy_score(y_als_full, oof_pred)
p("\n" + "="*66)
p(f"AGE-MATCHED OOF:  AUC={oof_auc:.3f}  acc={oof_acc:.3f}  (n={len(codes)}, age-balanced p=0.91)")
p(f"  (original 39-subject imbalanced OOF AUC was 0.96)")

# age-invariance of the NEW score, within group
for g, gv in [("ALS", 0), ("CTRL", 1)]:
    m = labels == gv
    r, pv = stats.spearmanr(lab.age.values[m], oof_pal[m])
    p(f"  new P_ALS vs age within {g}: rho={r:+.2f} p={pv:.2f}")

# did the balanced model recover the SAME axis as the original dim477?
r_axis, p_axis = stats.spearmanr(lab.d477.values, oof_pal)
p(f"  new balanced-model P_ALS vs ORIGINAL dim477 (n=24): Spearman rho={r_axis:+.2f} p={p_axis:.1e}")
# best single latent dim (report as sanity; selected in-sample so descriptive only)
gb = np.array([abs(stats.pointbiserialr(y_als_full, oof_zagg[:, k])[0]) for k in range(LATENT_DIM)])
p(f"  top OOF latent dim |r_group| = {gb.max():.2f} at dim{int(gb.argmax())} "
  f"(#dims with |r|>0.5: {(gb>0.5).sum()})")

pd.DataFrame({"code": codes, "id": lab.id.values, "group": lab.group.values,
              "age": lab.age.values, "fold": oof_fold, "P_ALS_new": np.round(oof_pal, 4),
              "orig_dim477": lab.d477.values}).to_csv(
    os.path.join(OUT, "age_matched_oof.csv"), index=False)
pd.DataFrame(fold_rows).to_csv(os.path.join(OUT, "fold_metrics.csv"), index=False)
np.save(os.path.join(OUT, "oof_zagg.npy"), oof_zagg)
p("\nsaved ->", OUT); log.close()
