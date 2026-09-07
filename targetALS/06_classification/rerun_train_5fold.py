"""
Part 2 — Controlled re-run of the 5-fold autoencoder+classifier training.

Uses the model classes from conv_autoencoder_detailed.py, the out_0h/6h/24h
data and sample_labels.csv, and the SAME canonical seed-42
StratifiedKFold(5, shuffle=True) split that was validated against the existing
models. Records per-epoch learning curves and per-fold / OOF validation metrics.

Positive class = ALS (encoded as 1 here; the partition is identical regardless
of label polarity).
"""
import os, sys, time, importlib.util, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from openpyxl import load_workbook
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
TP_DIRS = [os.path.join(ROOT, d) for d in ["out_0h", "out_6h", "out_24h"]]
LABELS_CSV = os.path.join(ROOT, "sample_labels.csv")
OUT = os.path.join(ROOT, "model_metrics_rerun")
MDL = os.path.join(OUT, "models")
os.makedirs(MDL, exist_ok=True)

DEVICE = torch.device("cpu")
LATENT_DIM = 512
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-4
LR = 1e-3
BATCH = 8
ALPHA = 1.0   # recon weight
BETA = 1.0    # class weight
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))

spec = importlib.util.spec_from_file_location(
    "cae", os.path.join(ROOT, "conv_autoencoder_detailed.py"))
cae = importlib.util.module_from_spec(spec); spec.loader.exec_module(cae)
Model = cae.ConvAutoencoderWithAttention


def load_excel(fp):
    wb = load_workbook(fp, data_only=True); ws = wb.active
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
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)


class DS(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]))


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    rc, cc = nn.MSELoss(), nn.CrossEntropyLoss()
    tot = tr = tc = 0.0
    correct = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(DEVICE); labels = labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            recon, logits, _ = model(images)
            rl = rc(recon, images); cl = cc(logits, labels)
            loss = ALPHA * rl + BETA * cl
            if train:
                loss.backward(); optimizer.step()
            tot += loss.item(); tr += rl.item(); tc += cl.item()
            correct += (logits.argmax(1) == labels).sum().item(); n += labels.size(0)
    nb = len(loader)
    return tot / nb, tr / nb, tc / nb, 100.0 * correct / n


def main():
    log = open(os.path.join(OUT, "train_log.txt"), "w")
    def p(*a):
        msg = " ".join(str(x) for x in a)
        print(msg); log.write(msg + "\n"); log.flush()

    lab = pd.read_csv(LABELS_CSV)
    codes = lab["code"].astype(str).tolist()
    y = np.array([1 if str(g).strip().lower() == "als" else 0 for g in lab["group"]], dtype=int)

    p(f"Loading {len(codes)} subjects x 3 timepoints ...")
    X = np.stack([np.stack([load_excel(os.path.join(d, f"{c}.xlsx"))[None] for d in TP_DIRS], 0)
                  for c in codes], 0).astype(np.float32)  # [N,3,1,H,W]
    p("X shape:", X.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(len(codes)), y))

    curves = {}
    oof_p = np.full(len(codes), np.nan)
    oof_fold = np.full(len(codes), -1, dtype=int)
    rows = []

    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        p(f"\n===== Fold {k}/5 | train={len(tr_idx)} val={len(va_idx)} =====")
        torch.manual_seed(SEED + k); np.random.seed(SEED + k)
        g = torch.Generator(); g.manual_seed(SEED + k)
        tl = DataLoader(DS(X[tr_idx], y[tr_idx]), batch_size=BATCH, shuffle=True,
                        num_workers=0, generator=g)
        vl = DataLoader(DS(X[va_idx], y[va_idx]), batch_size=BATCH, shuffle=False,
                        num_workers=0)

        model = Model(in_channels=1, latent_dim=LATENT_DIM, num_classes=2).to(DEVICE)
        # initialize lazy layers (encoder.fc, decoder) before building optimizer
        with torch.no_grad():
            model.eval(); _ = model(torch.from_numpy(X[tr_idx[:1]]).to(DEVICE))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        hist = {kk: [] for kk in ["tr_loss", "tr_recon", "tr_class", "tr_acc",
                                   "va_loss", "va_recon", "va_class", "va_acc"]}
        best_val = float("inf"); best_state = None; best_ep = -1; bad = 0
        t0 = time.time()
        for ep in range(EPOCHS):
            trl, trr, trc, tra = run_epoch(model, tl, optimizer)
            val, var, vac, vaa = run_epoch(model, vl, None)
            for kk, vv in zip(hist, [trl, trr, trc, tra, val, var, vac, vaa]):
                hist[kk].append(vv)
            if ep == 0 or (ep + 1) % 5 == 0:
                p(f"  ep{ep+1:3d}: tr_loss={trl:.4f} tr_acc={tra:5.1f}  "
                  f"va_loss={val:.4f} va_acc={vaa:5.1f}  ({time.time()-t0:.0f}s)")
            if val < best_val - MIN_DELTA:
                best_val = val; best_ep = ep; bad = 0
                best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    p(f"  early stop at ep{ep+1} (best ep{best_ep+1}, val={best_val:.4f})")
                    break
        curves[f"fold{k}"] = hist

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(MDL, f"best_model_fold{k}.pth"))

        # eval best on val
        model.eval()
        with torch.no_grad():
            _, logits, _ = model(torch.from_numpy(X[va_idx]).to(DEVICE))
            pal = torch.softmax(logits, 1)[:, 1].cpu().numpy()  # P(class1)=P(ALS) here
        for j, i in enumerate(va_idx):
            oof_p[i] = pal[j]; oof_fold[i] = k
        yk = y[va_idx]; prk = (pal >= 0.5).astype(int)
        try:
            auc = roc_auc_score(yk, pal)
        except ValueError:
            auc = np.nan
        rows.append(dict(fold=k, n=len(va_idx), best_epoch=best_ep + 1,
                         best_val_loss=best_val,
                         accuracy=accuracy_score(yk, prk), auc=auc,
                         precision=precision_score(yk, prk, zero_division=0),
                         recall=recall_score(yk, prk, zero_division=0),
                         f1=f1_score(yk, prk, zero_division=0)))
        p(f"  [fold{k}] acc={rows[-1]['accuracy']:.3f} auc={auc:.3f}")

    # save caches + metrics
    np.savez(os.path.join(OUT, "rerun_cache.npz"),
             oof_p=oof_p, oof_fold=oof_fold, y=y, codes=np.array(codes))
    with open(os.path.join(OUT, "curves.json"), "w") as f:
        json.dump(curves, f)

    fold_df = pd.DataFrame(rows)
    oof_pred = (oof_p >= 0.5).astype(int)
    summ = pd.concat([fold_df, pd.DataFrame([
        dict(fold="mean", **{m: fold_df[m].mean() for m in
             ["accuracy", "auc", "precision", "recall", "f1"]}),
        dict(fold="std", **{m: fold_df[m].std() for m in
             ["accuracy", "auc", "precision", "recall", "f1"]}),
        dict(fold="OOF", n=len(y), accuracy=accuracy_score(y, oof_pred),
             auc=roc_auc_score(y, oof_p),
             precision=precision_score(y, oof_pred, zero_division=0),
             recall=recall_score(y, oof_pred, zero_division=0),
             f1=f1_score(y, oof_pred, zero_division=0))])], ignore_index=True)
    summ.to_csv(os.path.join(OUT, "rerun_metrics.csv"), index=False)
    pd.DataFrame({"code": codes,
                  "true_label": ["ALS" if v else "CTRL" for v in y],
                  "held_out_fold": oof_fold, "P_ALS": np.round(oof_p, 4),
                  "pred": ["ALS" if v else "CTRL" for v in oof_pred],
                  "correct": (oof_pred == y)}).to_csv(
        os.path.join(OUT, "rerun_oof_predictions.csv"), index=False)
    p("\n" + summ.round(3).to_string(index=False))
    p("\nDone. Outputs in", OUT)
    log.close()


if __name__ == "__main__":
    main()
