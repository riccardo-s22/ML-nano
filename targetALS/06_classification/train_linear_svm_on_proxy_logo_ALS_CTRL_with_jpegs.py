#!/usr/bin/env python3
"""
LOOCV Linear SVM (LinearSVC) on proxy latents (proxy_latents.csv) with:

- C grid search (by pooled AUC, tie-break by balanced accuracy at default thr=0)
- Optional threshold sweep on pooled LOOCV decision scores
- Saves CSV/JSON metrics artifacts for reproducibility
- Saves JPEG figures:
    - confusion_matrix_default.jpeg
    - confusion_matrix_operating.jpeg (if threshold sweep)
    - roc_curve.jpeg (if threshold sweep)
    - pr_curve.jpeg (if threshold sweep)
    - metrics_summary.jpeg

Expected proxy CSV columns:
  - subject (or code) [optional but recommended]
  - group (e.g., CTRL/ALS)
  - proxy_dim_* feature columns

Example:
  python train_linear_svm_on_proxy_logo_ALS_CTRL_with_jpegs.py ^
    --proxy_csv proxy_latents.csv ^
    --positive_group ALS ^
    --c_grid 0.01,0.1,1,10 ^
    --sweep_thresholds ^
    --choose_metric youden ^
    --outdir results_svm

Notes:
- For your dataset, use: CTRL=0, ALS=1 by setting --positive_group ALS
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut


def _parse_c_grid(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    if not out:
        raise ValueError("--c_grid must contain at least one value")
    return out


def _load_proxy_latents(proxy_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(proxy_csv)
    df.columns = [c.strip() for c in df.columns]

    # tolerate either "subject" or "code"
    if "subject" not in df.columns and "code" in df.columns:
        df = df.rename(columns={"code": "subject"})

    required = {"group"}
    if not required.issubset(df.columns):
        raise ValueError(f"proxy_latents.csv must contain column(s) {sorted(required)}. Found: {list(df.columns)}")

    feat_cols = [c for c in df.columns if c.startswith("proxy_dim_")]
    if not feat_cols:
        raise ValueError("No proxy features found (expected columns starting with 'proxy_dim_').")

    return df


def _binary_labels(groups: List[str], positive_group: str | None) -> Tuple[np.ndarray, Dict[str, int], str]:
    uniq = sorted(list(set(groups)))
    if len(uniq) != 2:
        raise ValueError(f"Expected exactly 2 groups, found: {uniq}")

    if positive_group is None:
        # default behavior: if ALS exists, make it positive, else fall back to the second sorted label
        pos = "ALS" if "ALS" in uniq else uniq[1]
    else:
        pos = positive_group
        if pos not in uniq:
            raise ValueError(f"--positive_group '{pos}' not found in groups: {uniq}")

    neg = [g for g in uniq if g != pos][0]
    y = np.array([1 if g == pos else 0 for g in groups], dtype=int)
    mapping = {neg: 0, pos: 1}
    return y, mapping, pos


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    return float(tn / denom) if denom > 0 else float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(_compute_specificity(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, mapping: Dict[str, int], title: str, out_jpeg: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # labels in order 0,1, but display group names if possible
    inv = {v: k for k, v in mapping.items()}
    xlabels = [f"Pred {inv.get(0,'0')}", f"Pred {inv.get(1,'1')}"]
    ylabels = [f"True {inv.get(0,'0')}", f"True {inv.get(1,'1')}"]

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, xlabels, rotation=45, ha="right")
    plt.yticks(tick_marks, ylabels)

    # annotate cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_jpeg, dpi=200)
    plt.close()

    # also return counts for caller if needed (not used directly here)
    _ = (tn, fp, fn, tp)


@dataclass
class PooledResult:
    y_true: np.ndarray
    scores: np.ndarray
    y_pred_default: np.ndarray
    auc: float
    cm_default: Tuple[int, int, int, int]  # TN, FP, FN, TP
    metrics_default: Dict[str, float]


def _loocv_scores(X: np.ndarray, y: np.ndarray, C: float, class_weight: str | None = "balanced") -> PooledResult:
    loo = LeaveOneOut()
    scores = np.zeros_like(y, dtype=float)
    y_pred_default = np.zeros_like(y, dtype=int)

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]

        clf = LinearSVC(C=C, class_weight=class_weight, dual="auto", random_state=0)
        clf.fit(Xtr, ytr)

        s = float(clf.decision_function(Xte)[0])
        scores[test_idx[0]] = s
        y_pred_default[test_idx[0]] = 1 if s >= 0.0 else 0

    auc = float(roc_auc_score(y, scores))

    cm = confusion_matrix(y, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = _compute_metrics(y, y_pred_default)

    return PooledResult(
        y_true=y,
        scores=scores,
        y_pred_default=y_pred_default,
        auc=auc,
        cm_default=(tn, fp, fn, tp),
        metrics_default=metrics,
    )


def _threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    uniq = np.unique(scores)
    thresholds = np.concatenate(([np.min(uniq) - 1e-12], uniq, [np.max(uniq) + 1e-12]))

    rows = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        acc = float(accuracy_score(y_true, y_pred))
        bal = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        spec = float(_compute_specificity(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        youden = rec + spec - 1.0 if not np.isnan(spec) else float("nan")

        rows.append(
            {
                "threshold": float(thr),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "accuracy": acc,
                "balanced_accuracy": bal,
                "precision": prec,
                "recall": rec,
                "specificity": spec,
                "f1": f1,
                "youden_J": float(youden),
            }
        )

    return pd.DataFrame(rows)


def _best_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity", "youden_J"]
    out: Dict[str, Dict[str, float]] = {}

    for m in metrics:
        sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[m]).copy()
        if len(sub) == 0:
            continue
        best_val = float(sub[m].max())
        cand = sub[sub[m] == best_val].copy()
        cand["abs_thr"] = np.abs(cand["threshold"])
        cand = cand.sort_values("abs_thr", ascending=True)
        best = cand.iloc[0].to_dict()
        out[m] = {
            "threshold": float(best["threshold"]),
            "value": float(best[m]),
            "accuracy": float(best["accuracy"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "specificity": float(best["specificity"]),
            "f1": float(best["f1"]),
            "TN": int(best["TN"]),
            "FP": int(best["FP"]),
            "FN": int(best["FN"]),
            "TP": int(best["TP"]),
        }
    return out


def _point_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tpr

    return {
        "threshold": float(thr),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "precision": float(prec),
        "recall": float(rec),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def _plot_roc(y_true: np.ndarray, scores: np.ndarray, op_thr: float, out_jpeg: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    op = _point_at_threshold(y_true, scores, op_thr)

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.scatter([op["FPR"]], [op["TPR"]], marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={auc:.4f}), operating threshold={op_thr:.6g}")
    plt.tight_layout()
    plt.savefig(out_jpeg, dpi=200)
    plt.close()
    return auc


def _plot_pr(y_true: np.ndarray, scores: np.ndarray, op_thr: float, out_jpeg: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    op = _point_at_threshold(y_true, scores, op_thr)

    plt.figure()
    plt.plot(rec, prec, linewidth=2)
    plt.scatter([op["recall"]], [op["precision"]], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall, operating threshold={op_thr:.6g}")
    plt.tight_layout()
    plt.savefig(out_jpeg, dpi=200)
    plt.close()


def _write_metrics_summary_jpeg(summary: Dict, out_jpeg: Path) -> None:
    # Render a simple text-only summary image
    lines = []
    lines.append(f"Best C: {summary['best_C']}")
    lines.append(f"Pooled LOOCV AUC: {summary['best_auc']:.4f}")
    lines.append("")
    lines.append("Default threshold (0.0):")
    cm = summary["default_confusion"]
    lines.append(f"  Confusion (TN,FP,FN,TP): ({cm['TN']}, {cm['FP']}, {cm['FN']}, {cm['TP']})")
    for k, v in summary["default_metrics"].items():
        lines.append(f"  {k}: {v:.4f}")
    if "operating_point" in summary:
        lines.append("")
        lines.append(f"Operating threshold ({summary['choose_metric']}): {summary['operating_point']['threshold']:.6g}")
        cm2 = summary["operating_point"]["confusion"]
        lines.append(f"  Confusion (TN,FP,FN,TP): ({cm2['TN']}, {cm2['FP']}, {cm2['FN']}, {cm2['TP']})")
        for k, v in summary["operating_point"]["metrics"].items():
            lines.append(f"  {k}: {v:.4f}")

    text = "\n".join(lines)

    plt.figure(figsize=(9, 6))
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", ha="left", family="monospace")
    plt.tight_layout()
    plt.savefig(out_jpeg, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy_csv", type=Path, default=Path("proxy_latents.csv"))
    ap.add_argument("--c_grid", type=str, default="0.01,0.1,1,10")
    ap.add_argument("--positive_group", type=str, default="ALS", help="Set to 'ALS' to enforce CTRL=0, ALS=1.")
    ap.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "none"])
    ap.add_argument(
        "--sweep_thresholds",
        action="store_true",
        help="Sweep thresholds on pooled LOOCV scores for the best-C model, and write ROC/PR + operating confusion matrix.",
    )
    ap.add_argument(
        "--choose_metric",
        type=str,
        default="youden",
        choices=["youden", "balanced_accuracy", "f1", "accuracy", "precision", "recall", "specificity"],
        help="Metric used to select the operating threshold for plots.",
    )
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--img_ext", type=str, default="jpeg", choices=["jpeg", "jpg", "png"])
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = _load_proxy_latents(args.proxy_csv)
    feat_cols = [c for c in df.columns if c.startswith("proxy_dim_")]

    groups = df["group"].astype(str).tolist()
    y, mapping, pos = _binary_labels(groups, args.positive_group)
    X = df[feat_cols].to_numpy(dtype=float)

    class_weight = None if args.class_weight == "none" else "balanced"
    c_grid = _parse_c_grid(args.c_grid)

    print(f"Rows: total={len(df)}")
    print(f"Positive class: {pos} (mapping: {mapping})")

    pooled_by_c: Dict[float, PooledResult] = {}
    best_c = None
    best_auc = -1.0
    best_bal = -1.0

    for C in c_grid:
        res = _loocv_scores(X, y, C=C, class_weight=class_weight)
        pooled_by_c[C] = res

        print(f"\n=== C={C} ===")
        print(f"AUC (pooled): {res.auc:.4f}")
        m = res.metrics_default
        print(
            f"Default thr=0.0: acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
            f"sens={m['recall']:.3f} spec={m['specificity']:.3f} f1={m['f1']:.3f}"
        )

        if res.auc > best_auc + 1e-12 or (abs(res.auc - best_auc) <= 1e-12 and m["balanced_accuracy"] > best_bal):
            best_auc = res.auc
            best_bal = m["balanced_accuracy"]
            best_c = C

    assert best_c is not None
    best_res = pooled_by_c[best_c]

    # Save default confusion matrix image
    ext = args.img_ext
    cm_default_path = args.outdir / f"confusion_matrix_default.{ext}"
    _plot_confusion_matrix(best_res.y_true, best_res.y_pred_default, mapping, "Confusion Matrix (default thr=0.0)", cm_default_path)

    # Save summary JSON/CSV
    tn, fp, fn, tp = best_res.cm_default
    summary = {
        "best_C": float(best_c),
        "best_auc": float(best_res.auc),
        "mapping": mapping,
        "default_threshold": 0.0,
        "default_confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "default_metrics": best_res.metrics_default,
    }

    print("\n==============================")
    print("BEST SETTING (by pooled AUC)")
    print("==============================")
    print(f"Best C: {best_c}")
    print(f"Best pooled AUC: {best_res.auc:.4f}")
    print(f"Saved: {cm_default_path}")
    print("Using default threshold (0.0):")
    print(f"Confusion (TN,FP,FN,TP): ({tn}, {fp}, {fn}, {tp})")
    for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"]:
        print(f"{k.replace('_',' ').title()}: {best_res.metrics_default[k]:.4f}")

    # Threshold sweep + ROC/PR + operating confusion matrix
    if args.sweep_thresholds:
        sweep = _threshold_sweep(best_res.y_true, best_res.scores)
        sweep_path = args.outdir / "threshold_sweep.csv"
        sweep.to_csv(sweep_path, index=False)

        bests = _best_thresholds(sweep)

        chosen_key = "youden_J" if args.choose_metric == "youden" else args.choose_metric
        if chosen_key not in bests:
            raise ValueError(f"Chosen metric '{args.choose_metric}' not available in sweep results.")

        op_thr = float(bests[chosen_key]["threshold"])

        # operating y_pred and metrics
        y_pred_op = (best_res.scores >= op_thr).astype(int)
        op_metrics = _compute_metrics(best_res.y_true, y_pred_op)
        cm_op = confusion_matrix(best_res.y_true, y_pred_op, labels=[0, 1])
        op_tn, op_fp, op_fn, op_tp = int(cm_op[0, 0]), int(cm_op[0, 1]), int(cm_op[1, 0]), int(cm_op[1, 1])

        # plots
        roc_jpeg = args.outdir / f"roc_curve.{ext}"
        pr_jpeg = args.outdir / f"pr_curve.{ext}"
        _plot_roc(best_res.y_true, best_res.scores, op_thr, roc_jpeg)
        _plot_pr(best_res.y_true, best_res.scores, op_thr, pr_jpeg)

        cm_op_path = args.outdir / f"confusion_matrix_operating.{ext}"
        _plot_confusion_matrix(best_res.y_true, y_pred_op, mapping, f"Confusion Matrix (thr={op_thr:.6g})", cm_op_path)

        # console summary
        print("\n==============================")
        print("THRESHOLD SWEEP")
        print("==============================")
        print(f"Saved: {sweep_path}")
        print(f"Saved: {roc_jpeg}")
        print(f"Saved: {pr_jpeg}")
        print(f"Saved: {cm_op_path}")
        print(f"Chosen operating metric: {args.choose_metric}")
        print(f"Operating threshold: {op_thr:.6g}")
        print(f"Operating confusion (TN,FP,FN,TP): ({op_tn}, {op_fp}, {op_fn}, {op_tp})")
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"]:
            print(f"{k.replace('_',' ').title()} (op): {op_metrics[k]:.4f}")

        summary["choose_metric"] = args.choose_metric
        summary["operating_point"] = {
            "threshold": float(op_thr),
            "confusion": {"TN": op_tn, "FP": op_fp, "FN": op_fn, "TP": op_tp},
            "metrics": op_metrics,
        }

        # Also write a single-row operating metrics CSV for convenience
        op_row = {
            "best_C": float(best_c),
            "auc": float(best_res.auc),
            "threshold": float(op_thr),
            "TN": op_tn,
            "FP": op_fp,
            "FN": op_fn,
            "TP": op_tp,
            **op_metrics,
        }
        pd.DataFrame([op_row]).to_csv(args.outdir / "operating_point_metrics.csv", index=False)

    # Save summary artifacts + a metrics summary JPEG
    with open(args.outdir / "svm_loocv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_metrics_summary_jpeg(summary, args.outdir / f"metrics_summary.{ext}")

    print(f"Saved: {args.outdir / 'svm_loocv_summary.json'}")
    print(f"Saved: {args.outdir / f'metrics_summary.{ext}'}")


if __name__ == "__main__":
    main()
