"""
PLS-DA (and companion linear chemometric) baselines for ALS-vs-CTRL from EEM,
evaluated on the SAME inputs and the SAME canonical 5-fold OOF split as the
conv-autoencoder, so the comparison is apples-to-apples.

Design goals (to be fair / steelman the simple baseline):
  * identical inputs: per-file min-max EEM, 3 timepoints stacked, flattened.
  * identical outer folds: canonical held_out_fold from the AE evaluation.
  * leak-free nested CV: all preprocessing + component/hyperparam selection is
    fit inside the training partition of each outer fold only.
  * PLS-DA hyperparameter (n_components) and preprocessing (mean-center vs
    autoscale) chosen by inner stratified CV maximizing AUC.
  * also report an *optimistic* OOF-AUC-vs-components curve so the reader sees
    the method's ceiling, not just the honestly-tuned operating point.

Outputs: console table, results CSV, ROC overlay, PLS score plot, ncomp curve.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT = "/mnt/c/Users/riccardo-s/Documents/CNT/targetALS/exp5"
CACHE = os.path.join(ROOT, "plsda_baseline", "eem_cache.npz")
AE_OOF = os.path.join(ROOT, "model_metrics_eval", "existing_models_oof_predictions.csv")
OUT = os.path.join(ROOT, "plsda_baseline")
os.makedirs(OUT, exist_ok=True)
RNG = 42


# ----------------------------- preprocessing -----------------------------
def preprocess_fit(Xtr, mode):
    """Return (transform_fn, ) fit on training rows only."""
    if mode == "center":
        mu = Xtr.mean(axis=0, keepdims=True)
        return lambda A: A - mu
    elif mode == "autoscale":
        sc = StandardScaler().fit(Xtr)
        return lambda A: sc.transform(A)
    else:
        raise ValueError(mode)


# ----------------------------- PLS-DA core -----------------------------
def plsda_scores(Xtr, ytr, Xte, ncomp, mode):
    """Fit PLS-DA on train, return continuous decision score for test rows."""
    tf = preprocess_fit(Xtr, mode)
    Xtr2, Xte2 = tf(Xtr), tf(Xte)
    ncomp = min(ncomp, min(Xtr2.shape) - 1)
    pls = PLSRegression(n_components=ncomp, scale=False)
    pls.fit(Xtr2, ytr.astype(float))
    return pls.predict(Xte2).ravel(), ncomp


def inner_select(Xtr, ytr, comp_grid, modes, n_splits=4):
    """Pick (ncomp, mode) by inner stratified-CV mean AUC on the training set."""
    best, best_auc = None, -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    for mode in modes:
        for nc in comp_grid:
            scores, labels = [], []
            ok = True
            for itr, ite in skf.split(Xtr, ytr):
                if len(np.unique(ytr[itr])) < 2:
                    ok = False
                    break
                s, _ = plsda_scores(Xtr[itr], ytr[itr], Xtr[ite], nc, mode)
                scores.append(s)
                labels.append(ytr[ite])
            if not ok:
                continue
            s = np.concatenate(scores)
            l = np.concatenate(labels)
            if len(np.unique(l)) < 2:
                continue
            auc = roc_auc_score(l, s)
            if auc > best_auc:
                best_auc, best = auc, (nc, mode)
    return best, best_auc


# ----------------------------- generic OOF runner -----------------------------
def oof_predict(X, y, fold, score_fn):
    """score_fn(Xtr,ytr,Xte)->test scores. Returns OOF score vector aligned to rows."""
    oof = np.full(len(y), np.nan)
    for k in np.unique(fold):
        te = fold == k
        tr = ~te
        oof[te] = score_fn(X[tr], y[tr], X[te])
    return oof


def metrics(y, score, thr=0.5):
    pred = (score >= thr).astype(int)
    return dict(
        auc=roc_auc_score(y, score),
        acc=accuracy_score(y, pred),
        f1=f1_score(y, pred),
    )


def main():
    d = np.load(CACHE, allow_pickle=True)
    X4, y, fold, codes = d["X"], d["y"], d["fold"], d["codes"].astype(str)
    N = X4.shape[0]
    X = X4.reshape(N, -1)  # [39, 109056]
    print(f"X flat: {X.shape}, ALS={y.sum()}/{N}, folds={np.bincount(fold)[1:]}")

    comp_grid = list(range(1, 16))
    modes = ["center", "autoscale"]
    rows = []

    # ---------- 1. PLS-DA, nested (honest) ----------
    oof = np.full(N, np.nan)
    picks = []
    for k in np.unique(fold):
        te = fold == k
        tr = ~te
        (nc, mode), iauc = inner_select(X[tr], y[tr], comp_grid, modes)
        s, nc_used = plsda_scores(X[tr], y[tr], X[te], nc, mode)
        oof[te] = s
        picks.append((int(k), nc, mode, round(iauc, 3)))
    m = metrics(y, oof)
    print("\nPLS-DA nested picks (fold, ncomp, preproc, innerAUC):")
    for p in picks:
        print("  ", p)
    print(f"PLS-DA (nested)   AUC={m['auc']:.3f} acc={m['acc']:.3f} f1={m['f1']:.3f}")
    rows.append(("PLS-DA (nested CV, tuned)", m["auc"], m["acc"], m["f1"]))
    plsda_oof = oof.copy()

    # ---------- 2. PLS-DA, optimistic OOF-AUC vs ncomp (fixed comp, autoscale) ----------
    curve = []
    for nc in comp_grid:
        oof_c = oof_predict(X, y, fold,
                            lambda a, b, c, nc=nc: plsda_scores(a, b, c, nc, "autoscale")[0])
        curve.append((nc, roc_auc_score(y, oof_c), accuracy_score(y, (oof_c >= 0.5).astype(int))))
    curve = np.array(curve)
    best_nc = int(curve[np.argmax(curve[:, 1]), 0])
    best_auc = curve[:, 1].max()
    print(f"\nPLS-DA optimistic ceiling (best fixed ncomp={best_nc}): AUC={best_auc:.3f}")
    rows.append((f"PLS-DA (optimistic, fixed k={best_nc})", best_auc,
                 curve[np.argmax(curve[:, 1]), 2], np.nan))

    # ---------- 3. Companion linear baselines (nested-ish, leak-free) ----------
    def pca_lda_score(Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        nc = min(10, min(Xtr2.shape) - 1)
        pca = PCA(n_components=nc, random_state=RNG).fit(Xtr2)
        Ztr, Zte = pca.transform(Xtr2), pca.transform(Xte2)
        lda = LinearDiscriminantAnalysis().fit(Ztr, ytr)
        return lda.predict_proba(Zte)[:, 1]

    def logreg_score(Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
        lr = LogisticRegression(penalty="l2", C=0.01, max_iter=5000).fit(Xtr2, ytr)
        return lr.predict_proba(Xte2)[:, 1]

    for name, fn in [("PCA(10)+LDA", pca_lda_score), ("L2-LogReg (C=0.01)", logreg_score)]:
        oof_b = oof_predict(X, y, fold, fn)
        mb = metrics(y, oof_b)
        print(f"{name:22s} AUC={mb['auc']:.3f} acc={mb['acc']:.3f} f1={mb['f1']:.3f}")
        rows.append((name, mb["auc"], mb["acc"], mb["f1"]))

    # ---------- 4. AE reference (same folds) ----------
    ae = pd.read_csv(AE_OOF)
    ae = ae.set_index("code").loc[codes]
    ae_score = ae["P_ALS"].values
    ae_y = (ae["true_label"].str.lower() == "als").astype(int).values
    assert np.array_equal(ae_y, y), "AE label alignment mismatch"
    ma = metrics(y, ae_score)
    print(f"\nConv-AE (same folds) AUC={ma['auc']:.3f} acc={ma['acc']:.3f} f1={ma['f1']:.3f}")
    rows.append(("Conv-autoencoder + classifier", ma["auc"], ma["acc"], ma["f1"]))

    # ---------- save results table ----------
    res = pd.DataFrame(rows, columns=["method", "OOF_AUC", "OOF_acc", "OOF_f1"])
    res.to_csv(os.path.join(OUT, "results_table.csv"), index=False)
    print("\n==== RESULTS TABLE ====")
    print(res.to_string(index=False))

    # ---------- permutation test: is AE > PLS-DA by chance on this split? ----------
    # paired label-permutation on AUC difference
    rs = np.random.RandomState(RNG)
    obs = ma["auc"] - m["auc"]
    perm = []
    for _ in range(2000):
        yp = rs.permutation(y)
        perm.append(roc_auc_score(yp, ae_score) - roc_auc_score(yp, plsda_oof))
    perm = np.array(perm)
    p_two = (np.abs(perm) >= abs(obs)).mean()
    print(f"\nAUC(AE)-AUC(PLSDA nested) = {obs:+.3f}; label-perm two-sided p={p_two:.3f}")

    # ---------- figures ----------
    # ROC overlay
    plt.figure(figsize=(5, 5))
    for lab, sc, col in [("Conv-AE", ae_score, "C3"),
                         (f"PLS-DA nested (AUC {m['auc']:.2f})", plsda_oof, "C0"),
                         (f"PLS-DA best k={best_nc} (AUC {best_auc:.2f})",
                          oof_predict(X, y, fold, lambda a, b, c: plsda_scores(a, b, c, best_nc, "autoscale")[0]), "C0")]:
        fpr, tpr, _ = roc_curve(y, sc)
        style = "-" if "AE" in lab or "nested" in lab else "--"
        plt.plot(fpr, tpr, style, color=col, label=f"{lab}")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("OOF ROC: Conv-AE vs PLS-DA (same folds)")
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_overlay.png"), dpi=150)
    plt.close()

    # ncomp curve
    plt.figure(figsize=(5, 4))
    plt.plot(curve[:, 0], curve[:, 1], "o-", label="OOF AUC")
    plt.axhline(ma["auc"], color="C3", ls="--", label=f"Conv-AE AUC {ma['auc']:.2f}")
    plt.xlabel("PLS latent variables (n_components)")
    plt.ylabel("OOF AUC (optimistic, fixed k)")
    plt.title("PLS-DA performance vs model complexity")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "plsda_ncomp_curve.png"), dpi=150)
    plt.close()

    # PLS score plot (fit on ALL data, illustrative only)
    sc_all = StandardScaler().fit(X)
    pls_all = PLSRegression(n_components=2, scale=False).fit(sc_all.transform(X), y.astype(float))
    T = pls_all.x_scores_
    plt.figure(figsize=(5, 4.2))
    for cls, col, name in [(1, "C3", "ALS"), (0, "C0", "CTRL")]:
        plt.scatter(T[y == cls, 0], T[y == cls, 1], c=col, label=name, s=40, edgecolor="k", lw=0.4)
    plt.xlabel("PLS LV1"); plt.ylabel("PLS LV2")
    plt.title("PLS-DA scores (full-data, illustrative)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "plsda_score_plot.png"), dpi=150)
    plt.close()

    print("\nWrote figures + results_table.csv to", OUT)


if __name__ == "__main__":
    main()
