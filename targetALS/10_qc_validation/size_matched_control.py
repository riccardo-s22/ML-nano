"""
SIZE-MATCHED CONTROL for the age-matched retrain.

Disambiguates the AUC drop (0.96 full-cohort -> 0.78 age-matched): was it caused
by removing the age imbalance, or merely by halving the data (19 train subj/fold)?

This retrains the SAME faithful procedure on RANDOM 12-ALS + 12-CTRL subsets --
same n and class balance as the age-matched run, but age left IMBALANCED. Run
several random draws; report their OOF-AUC distribution. If these land near 0.78
too, the drop is a sample-size effect and age removal costs ~nothing.
"""
import os, time, json
import numpy as np, pandas as pd, torch, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats
import importlib.util

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
spec = importlib.util.spec_from_file_location("rf", os.path.join(ROOT, "rerun_faithful.py"))
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)

DEVICE = torch.device("cpu"); LATENT_DIM = 512; BATCH_SIZE = 4
NUM_EPOCHS = 100; LR = 1e-3; ALPHA = BETA = 1.0; N_SPLITS = 5
TIMEPOINT_DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
OUT = os.path.join(ROOT, "age_matched_retrain_outputs"); os.makedirs(OUT, exist_ok=True)
N_DRAWS = 3
log = open(os.path.join(OUT, "size_control_log.txt"), "w")
def p(*a):
    m = " ".join(str(x) for x in a); print(m); log.write(m + "\n"); log.flush()

meta = pd.read_csv(os.path.join(ROOT, "age_sex_confound_outputs", "merged_dim477_age_sex.csv"))
lab = pd.read_csv(os.path.join(ROOT, "sample_labels.csv"))
lab["id"] = lab["code"].str.extract(r"^(\d+)").astype(int)
als_ids = lab.id[lab.group == "ALS"].values
ctl_ids = lab.id[lab.group == "CTRL"].values
age = meta.set_index("code").age

# preload all EEMs once
p("preloading all EEMs ...")
cache = {}
for _, r in lab.iterrows():
    tps = [rf.load_excel_image(os.path.join(d, f"{r.code}.xlsx")) for d in TIMEPOINT_DIRS]
    cache[int(r.id)] = torch.from_numpy(np.stack(tps, axis=0))
id2code = dict(zip(lab.id, lab.code))

def run_subset(ids, tag, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    sub = lab[lab.id.isin(ids)].copy()
    codes = sub.code.values
    labels = (sub.group == "CTRL").astype(int).values  # ALS=0, CTRL=1
    ages = sub.id.map(age).values
    ap = stats.mannwhitneyu(ages[labels == 0], ages[labels == 1]).pvalue
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_pal = np.full(len(codes), np.nan)
    for fold, (tr, va) in enumerate(skf.split(codes, labels)):
        trc, trl = codes[tr], labels[tr]; vac = codes[va]
        ids_tr = sub.id.values[tr]; ids_va = sub.id.values[va]
        train_ds = rf.CachedDataset([cache[i] for i in ids_tr], trl)
        val_ds = rf.CachedDataset([cache[i] for i in ids_va], labels[va])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        model = rf.ConvAutoencoderWithAttention(1, LATENT_DIM, 2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        with torch.no_grad():
            model.eval(); _ = model(train_ds[0][0].unsqueeze(0).to(DEVICE))
        for prm in model.encoder.fc.parameters(): prm.requires_grad_(False)
        for prm in model.decoder.parameters(): prm.requires_grad_(False)
        model.train()
        best = -1.0; ck = os.path.join(OUT, f"_tmp_{tag}_f{fold}.pth")
        for ep in range(NUM_EPOCHS):
            rf.train_epoch(model, train_loader, optimizer, DEVICE, ALPHA, BETA)
            _, _, _, vaa = rf.validate(model, val_loader, DEVICE, ALPHA, BETA)
            if vaa > best: best = vaa; torch.save(model.state_dict(), ck)
        m2 = rf.ConvAutoencoderWithAttention(1, LATENT_DIM, 2).to(DEVICE)
        with torch.no_grad():
            m2.eval(); _ = m2(next(iter(val_loader))[0].to(DEVICE))
        m2.load_state_dict(torch.load(ck, map_location=DEVICE), strict=True); m2.eval()
        with torch.no_grad():
            for j, gi in enumerate(va):
                _, logits, _ = m2(cache[ids_va[j]].unsqueeze(0).to(DEVICE))
                oof_pal[gi] = 1.0 - torch.softmax(logits, 1)[0, 1].item()
        os.remove(ck)
    y_als = 1 - labels
    auc = roc_auc_score(y_als, oof_pal); acc = accuracy_score(y_als, (oof_pal >= 0.5).astype(int))
    p(f"  [{tag}] OOF AUC={auc:.3f} acc={acc:.3f}  (age MWU p={ap:.2f}; "
      f"ALS age med={np.median(ages[labels==0]):.0f} CTRL={np.median(ages[labels==1]):.0f})")
    return auc, acc, ap

results = []
for k in range(N_DRAWS):
    rng = np.random.default_rng(100 + k)
    ids = np.concatenate([rng.choice(als_ids, 12, replace=False),
                          rng.choice(ctl_ids, 12, replace=False)])
    p(f"\n=== RANDOM DRAW {k+1}/{N_DRAWS} (seed {100+k}) ===")
    auc, acc, ap = run_subset(ids, f"rand{k}", seed=42)
    results.append(dict(draw=k + 1, auc=auc, acc=acc, age_p=ap))

R = pd.DataFrame(results); R.to_csv(os.path.join(OUT, "size_control_metrics.csv"), index=False)
p("\n" + "=" * 60)
p(f"RANDOM size-matched (n=24, age-IMBALANCED) OOF AUC: "
  f"mean={R.auc.mean():.3f}  range={R.auc.min():.3f}-{R.auc.max():.3f}  (n_draws={N_DRAWS})")
p(f"age-matched (n=24, age-BALANCED) OOF AUC = 0.778 for comparison")
p("If random draws ~0.78 -> drop is sample size, not age.")
log.close()
