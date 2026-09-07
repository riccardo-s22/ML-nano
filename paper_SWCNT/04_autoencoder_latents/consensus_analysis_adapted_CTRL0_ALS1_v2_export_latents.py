
"""
consensus_analysis_adapted.py

Drop-in analysis script for your joint ConvAutoencoder+Classifier models trained across folds.

Outputs consensus heatmaps (mean across folds/models):
  1) Reconstruction error maps by class (ALS vs CTRL (or any 2-group labels)) + difference
  2) Gradient-based input importance for discriminative latent features
  3) Linear probe importance (pixels -> teacher margin)

Supports both single-timepoint and multi-timepoint inputs.
Handles your Excel structure: first row = excitation labels, first column = emission labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import glob
from typing import List, Tuple, Optional, Dict

import numpy as np

# Optional imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception as e:
    torch = None
    nn = None
    DataLoader = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from openpyxl import load_workbook
except Exception:
    load_workbook = None

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
except Exception:
    LogisticRegression = None
    Ridge = None
    StandardScaler = None


# -----------------------------
# Excel parsing helpers (axes-aware)
# -----------------------------
def _parse_excitation_label(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace("Excitation_", "").replace("excitation_", "")
    try:
        return float(s)
    except Exception:
        import re as _re
        nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(nums[-1]) if nums else float("nan")


def _parse_emission_label(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("Emission_", "").replace("emission_", "")
    try:
        return float(s)
    except Exception:
        import re as _re
        nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(nums[-1]) if nums else float("nan")


def load_excel_with_axes(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an .xlsx into (data_matrix, emission_axis, excitation_axis).

    Expected file structure:
      - First row (excluding [0,0]) is excitation axis labels
      - First col (excluding [0,0]) is emission axis labels
      - Remaining block is numeric intensity
    """
    if load_workbook is None:
        raise ImportError("openpyxl is required to read .xlsx files: pip install openpyxl")
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"Empty worksheet: {filepath}")

    header = rows[0]
    excitation = np.array([_parse_excitation_label(v) for v in header[1:]], dtype=float)
    emission = np.array([_parse_emission_label(r[0]) for r in rows[1:]], dtype=float)

    H = len(rows) - 1
    W = max(0, len(header) - 1)
    data = np.full((H, W), np.nan, dtype=float)
    for i, row in enumerate(rows[1:]):
        for j in range(W):
            val = row[j + 1] if (j + 1) < len(row) else None
            if val is None:
                v = np.nan
            elif isinstance(val, (int, float, np.number)):
                v = float(val)
            else:
                try:
                    v = float(str(val).strip())
                except Exception:
                    v = np.nan
            data[i, j] = v

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, emission, excitation


def resolve_excel_path(code: str, tp_dir: str) -> str:
    # code can be numeric or string; training used f"{code}.xlsx"
    return os.path.join(tp_dir, f"{code}.xlsx")


# -----------------------------
# Dynamic import for model + dataset
# -----------------------------
def import_arch(arch_file: str):
    spec = importlib.util.spec_from_file_location("archmod", arch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import arch file: {arch_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -----------------------------
# Data assembly from embeddings or labels CSV
# -----------------------------
def load_codes_and_groups(
    emb_glob: Optional[str],
    labels_csv: Optional[str],
) -> Tuple[List[str], List[str]]:
    """
    Returns (codes, groups) where codes are strings and groups are e.g. 'TG'/'WT'.
    Priority:
      1) labels_csv if provided
      2) first embeddings csv that matches emb_glob
    """
    if pd is None:
        raise ImportError("pandas is required: pip install pandas")

    if labels_csv:
        df = pd.read_csv(labels_csv)
        cols = {c.lower(): c for c in df.columns}
        # try common column names
        code_col = cols.get("code") or cols.get("id") or cols.get("sample") or cols.get("subject")
        group_col = cols.get("group") or cols.get("label") or cols.get("class")
        if code_col is None or group_col is None:
            raise ValueError(
                f"Could not infer code/group columns from {labels_csv}. "
                f"Need something like columns: code, group. Found: {list(df.columns)}"
            )
        codes = df[code_col].astype(str).tolist()
        groups = df[group_col].astype(str).tolist()
        return codes, groups

    if not emb_glob:
        raise ValueError("Need either --labels_csv or --emb_glob")
    emb_files = sorted(glob.glob(emb_glob))
    if not emb_files:
        raise FileNotFoundError(f"No embeddings matched: {emb_glob}")
    df = pd.read_csv(emb_files[0])
    if "code" not in df.columns or "group" not in df.columns:
        raise ValueError(f"Embeddings file must have columns code, group. Found: {list(df.columns)[:20]}")
    codes = df["code"].astype(str).tolist()
    groups = df["group"].astype(str).tolist()
    return codes, groups


def make_label_mapping(groups: List[str], preferred_order: Optional[List[str]] = None) -> Dict[str, int]:
    uniq = sorted(set(groups))
    if preferred_order:
        ordered = [g for g in preferred_order if g in uniq] + [g for g in uniq if g not in preferred_order]
    else:
        ordered = uniq
    return {g: i for i, g in enumerate(ordered)}


# -----------------------------
# Plotting
# -----------------------------
def plot_map(
    data: np.ndarray,
    title: str,
    out_png: str,
    extent: Optional[List[float]] = None,
    origin: str = "upper",
    xlabel: str = "Excitation index",
    ylabel: str = "Emission index",
):
    if plt is None:
        raise ImportError("matplotlib is required: pip install matplotlib")

    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto", origin=origin, extent=extent)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def infer_axis_extent(tp_dir: str, code: str) -> Tuple[Optional[List[float]], str]:
    """
    Returns (extent, origin) for imshow in excitation/emission units.
    Also flips vertically to put emission increasing upward.
    """
    try:
        arr, emission, excitation = load_excel_with_axes(resolve_excel_path(code, tp_dir))
    except Exception:
        return None, "upper"

    if len(emission) and len(excitation) and np.all(np.isfinite(emission)) and np.all(np.isfinite(excitation)):
        # We'll flip data when plotting, so origin='lower' and extent uses flipped emission axis
        em_disp = emission[::-1]
        extent = [float(excitation[0]), float(excitation[-1]), float(em_disp[0]), float(em_disp[-1])]
        return extent, "lower"
    return None, "upper"


# -----------------------------
# Core computations
# -----------------------------
def compute_reconstruction_error_maps(
    models,
    dataloader,
    device,
    label_names: Dict[int, str],
):
    """
    Pixel-wise MSE maps for each class, averaged across subjects and models.
    """
    criterion = nn.MSELoss(reduction="none")
    # Determine H,W from first batch
    images0, labels0 = next(iter(dataloader))
    _, T, C, H, W = images0.shape
    class_accum = {c: np.zeros((H, W), dtype=np.float64) for c in label_names.keys()}
    class_count = {c: 0 for c in label_names.keys()}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # average error across models for each subject
            per_model_maps = []
            for model in models:
                recons, logits, attn = model(images)  # recons: (B,T,C,H,W)
                loss = criterion(recons, images)      # (B,T,C,H,W)
                loss_map = loss.mean(dim=1).mean(dim=1)  # (B,H,W) average time + channel
                per_model_maps.append(loss_map)

            mean_loss_map = torch.stack(per_model_maps, dim=0).mean(dim=0)  # (B,H,W)
            mean_loss_np = mean_loss_map.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                c = int(labels_np[i])
                if c in class_accum:
                    class_accum[c] += mean_loss_np[i]
                    class_count[c] += 1

    class_mean = {c: class_accum[c] / max(class_count[c], 1) for c in class_accum}
    return class_mean


def compute_gradient_importance(models, dataloader, device, top_k_dims: int = 10):
    """
    Gradient-based input importance: d(sum of top discriminative latent dims)/d(input)
    Returns HxW map averaged across models and batches.
    """
    if LogisticRegression is None or StandardScaler is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    images0, _ = next(iter(dataloader))
    _, _, _, H, W = images0.shape

    grad_accum = np.zeros((H, W), dtype=np.float64)
    count = 0

    for model in models:
        # Fit LR on aggregated latents to select important dims
        z_list, y_list = [], []
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs = imgs.to(device)
                lbls = lbls.cpu().numpy()
                # Compute z_agg via encoder per timepoint + attention
                latents = []
                for t in range(imgs.shape[1]):
                    latents.append(model.encoder(imgs[:, t]))
                z_agg, _ = model.attention(torch.stack(latents, dim=1))
                z_list.append(z_agg.cpu().numpy())
                y_list.append(lbls)

        X_lat = np.concatenate(z_list, axis=0)
        y_lat = np.concatenate(y_list, axis=0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_lat)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")
        clf.fit(Xs, y_lat)
        w = clf.coef_[0]
        top_dims = np.argsort(np.abs(w))[-top_k_dims:]

        # Now compute gradients to increase those dims (weighted by sign)
        sign = np.sign(w[top_dims])
        sign[sign == 0] = 1.0

        for imgs, _ in dataloader:
            imgs = imgs.to(device).requires_grad_(True)

            latents = []
            for t in range(imgs.shape[1]):
                latents.append(model.encoder(imgs[:, t]))
            z_agg, _ = model.attention(torch.stack(latents, dim=1))

            # maximize signed activation of selected dims
            target = (z_agg[:, top_dims] * torch.tensor(sign, device=device, dtype=z_agg.dtype)).sum()
            model.zero_grad(set_to_none=True)
            if imgs.grad is not None:
                imgs.grad.zero_()
            target.backward()

            grads = imgs.grad.detach().abs().mean(dim=(0, 1, 2)).cpu().numpy()  # HxW
            grad_accum += grads
            count += 1

    return grad_accum / max(count, 1)


def compute_linear_probe_importance(models, dataloader, device, pos_class: int = 0):
    """
    Ridge: flattened pixels -> teacher margin (logit_pos - logit_other).
    Returns HxW abs(coef) averaged across models.
    """
    if Ridge is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    images0, _ = next(iter(dataloader))
    _, _, _, H, W = images0.shape

    coef_accum = np.zeros((H * W,), dtype=np.float64)
    n_models_ok = 0

    for model in models:
        X_list = []
        y_margin = []

        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                # Teacher logits from model
                recons, logits, attn = model(imgs)
                # margin
                if logits.ndim == 2 and logits.shape[1] == 2:
                    pos = logits[:, pos_class]
                    neg = logits[:, 1 - pos_class]
                    margin = (pos - neg).cpu().numpy()
                else:
                    # Fallback: treat as single logit
                    margin = logits.view(-1).cpu().numpy()

                # Flatten mean-over-time pixels (B,T,1,H,W) -> (B,H*W)
                flat = imgs.mean(dim=1).squeeze(1).reshape(imgs.shape[0], -1).cpu().numpy()
                X_list.append(flat)
                y_margin.append(margin)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_margin, axis=0)

        probe = Ridge(alpha=100.0)
        probe.fit(X, y)
        coef_accum += np.abs(probe.coef_)
        n_models_ok += 1

    coef_mean = coef_accum / max(n_models_ok, 1)
    return coef_mean.reshape((H, W))



# -----------------------------
# Latent dimension ranking + per-dimension attribution + ROI feature distillation
# -----------------------------
def rank_latent_dims_with_lr(models, dataloader, device):
    """
    Aggregates z_agg across all subjects (mean across models) and fits a balanced
    LogisticRegression on standardized z to rank latent dimensions.

    Returns:
      - z_mean: (N, D) mean latent across models
      - y: (N,) labels
      - w: (D,) LR weights (for class 1 vs 0)
      - auc_per_dim: (D,) univariate AUC using each z_dim alone (NaN if undefined)
    """
    if LogisticRegression is None or StandardScaler is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    z_models = []
    y_all = None

    with torch.no_grad():
        for model in models:
            z_list, y_list = [], []
            for imgs, lbls in dataloader:
                imgs = imgs.to(device)
                latents = []
                for t in range(imgs.shape[1]):
                    latents.append(model.encoder(imgs[:, t]))
                z_agg, _ = model.attention(torch.stack(latents, dim=1))
                z_list.append(z_agg.cpu().numpy())
                y_list.append(lbls.numpy())
            Z = np.concatenate(z_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            z_models.append(Z)
            if y_all is None:
                y_all = y

    z_mean = np.mean(np.stack(z_models, axis=0), axis=0)  # (N,D)
    y = y_all.astype(int)

    scaler = StandardScaler()
    Zs = scaler.fit_transform(z_mean)
    clf = LogisticRegression(class_weight="balanced", max_iter=4000, solver="liblinear")
    clf.fit(Zs, y)
    w = clf.coef_[0].astype(float)

    # Univariate AUC per dim (raw z, not standardized)
    D = z_mean.shape[1]
    auc_per_dim = np.full((D,), np.nan, dtype=float)
    for d in range(D):
        try:
            # Requires both classes present
            auc_per_dim[d] = roc_auc_score(y, z_mean[:, d])
        except Exception:
            auc_per_dim[d] = np.nan

    return z_mean, y, w, auc_per_dim


def per_dim_gradient_attribution(
    models,
    dataloader,
    device,
    dims,
    signs=None,
):
    """
    Computes per-latent-dimension |d z_dim / d input| attribution maps.

    Returns dict: dim -> HxW map (float64), averaged over models and batches.
    """
    images0, _ = next(iter(dataloader))
    _, _, _, H, W = images0.shape

    if signs is None:
        signs = {int(d): 1.0 for d in dims}

    out = {int(d): np.zeros((H, W), dtype=np.float64) for d in dims}
    counts = {int(d): 0 for d in dims}

    for model in models:
        model.eval()
        for imgs, _ in dataloader:
            imgs = imgs.to(device).requires_grad_(True)

            latents = []
            for t in range(imgs.shape[1]):
                latents.append(model.encoder(imgs[:, t]))
            z_agg, _ = model.attention(torch.stack(latents, dim=1))  # (B,D)

            for d in dims:
                d = int(d)
                s = float(signs.get(d, 1.0))
                target = (z_agg[:, d] * s).sum()

                model.zero_grad(set_to_none=True)
                if imgs.grad is not None:
                    imgs.grad.zero_()

                target.backward(retain_graph=True)

                grads = imgs.grad.detach().abs().mean(dim=(0, 1, 2)).cpu().numpy()  # HxW
                out[d] += grads
                counts[d] += 1

            # Important: clear graph refs
            imgs = imgs.detach()

    for d in out:
        out[d] = out[d] / max(counts[d], 1)

    return out


def roi_from_map(attr_map, emission_axis=None, excitation_axis=None, q=99.0):
    """
    Defines a single ROI as the bounding box covering pixels above the q-th percentile.

    Returns dict with pixel coordinates and (optionally) physical coordinates.
    """
    m = np.asarray(attr_map, dtype=float)
    thr = np.percentile(m, q)
    ys, xs = np.where(m >= thr)
    if len(xs) == 0:
        return dict(q=q, thr=float(thr), empty=True)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cy = float(ys.mean())
    cx = float(xs.mean())
    area = int(len(xs))

    out = dict(
        q=float(q),
        thr=float(thr),
        empty=False,
        y0=y0, y1=y1, x0=x0, x1=x1,
        cy=cy, cx=cx,
        area_px=area,
    )

    # Convert to excitation/emission coords if axes provided
    if excitation_axis is not None and len(excitation_axis) > 0:
        out["exc_min"] = float(excitation_axis[x0])
        out["exc_max"] = float(excitation_axis[x1])
        out["exc_centroid"] = float(excitation_axis[int(round(cx))])
    if emission_axis is not None and len(emission_axis) > 0:
        out["em_min"] = float(emission_axis[y0])
        out["em_max"] = float(emission_axis[y1])
        out["em_centroid"] = float(emission_axis[int(round(cy))])

    return out


def compute_roi_features(dataset, tp_dirs, roi, use_timepoint_means=True):
    """
    Deterministic (non-ML) ROI features from the raw EEM matrices.

    Features per subject:
      - ROI_SUM, ROI_MEAN, ROI_MAX on mean-over-time matrix
      - Optionally per-timepoint ROI_SUM_t, ROI_MEAN_t (if multiple timepoints)

    NOTE: This reads the .xlsx directly via load_excel_with_axes to stay model-agnostic.
    """
    y0, y1, x0, x1 = roi["y0"], roi["y1"], roi["x0"], roi["x1"]

    feats = []
    codes_list = getattr(dataset, 'codes', None) or getattr(dataset, 'subjects', None) or getattr(dataset, 'subject_codes', None)
    if codes_list is None:
        raise AttributeError('Dataset must expose codes list as .codes (or .subjects/.subject_codes).')

    for idx in range(len(codes_list)):
        code = str(codes_list[idx])

        mats = []
        for tp in tp_dirs:
            fp = resolve_excel_path(code, tp)
            mat, _, _ = load_excel_with_axes(fp)  # HxW
            mats.append(mat)
        mats = np.stack(mats, axis=0)  # T x H x W

        mean_mat = mats.mean(axis=0)
        patch = mean_mat[y0:y1 + 1, x0:x1 + 1]

        row = {
            "code": code,
            "ROI_SUM": float(np.sum(patch)),
            "ROI_MEAN": float(np.mean(patch)),
            "ROI_MAX": float(np.max(patch)),
        }

        if not use_timepoint_means:
            pass
        else:
            for t in range(mats.shape[0]):
                p = mats[t, y0:y1 + 1, x0:x1 + 1]
                row[f"ROI_SUM_t{t}"] = float(np.sum(p))
                row[f"ROI_MEAN_t{t}"] = float(np.mean(p))

        feats.append(row)

    return feats

# -----------------------------
# Main
# -----------------------------
def main():
    if torch is None:
        raise ImportError("PyTorch is required to run this script.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch_file", required=True, help="Architecture python file (e.g. conv_autoencoder_detailed_fixed.py)")
    ap.add_argument("--model_glob", default=None, help="Glob for model checkpoints (overrides --models_dir/--model_pattern); e.g. 24h_models/best_model_fold*.pth")
    ap.add_argument("--models_dir", default=None, help="Directory containing model .pth files (alternative to --model_glob)")
    ap.add_argument("--model_pattern", default="*.pth", help="Filename pattern within --models_dir (default: *.pth)")
    ap.add_argument("--timepoints", required=True, help="Comma-separated list of timepoint directories (e.g. out_0h,out_6h,out_24h) or single dir")
    ap.add_argument("--outdir", default="results_analysis", help="Output directory")
    ap.add_argument("--emb_glob", default=None, help="Glob for embeddings csv (optional, used to obtain codes/groups if labels_csv not provided)")
    ap.add_argument("--labels_csv", default=None, help="CSV with code,group columns (optional)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--latent_dim", type=int, default=None, help="Override latent_dim (otherwise inferred from embeddings or set to 512)")
    ap.add_argument("--preferred_group_order", default="CTRL,ALS", help="Comma-separated group order for label mapping (default CTRL,ALS)")
    ap.add_argument("--pos_group", default="ALS", help="Group treated as positive for margin in linear probe (default ALS)")
    ap.add_argument("--top_k_latent_dims", type=int, default=10)
    ap.add_argument("--roi_quantile", type=float, default=99.0, help="Percentile threshold for ROI extraction from attribution maps (default 99).")
    ap.add_argument("--max_dims_report", type=int, default=20, help="Max number of latent dims to report/plot (default 20).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tp_dirs = [t.strip() for t in args.timepoints.split(",") if t.strip()]
    if not tp_dirs:
        raise ValueError("No timepoint directories provided.")

    # Load codes + groups
    codes, groups = load_codes_and_groups(args.emb_glob, args.labels_csv)

    # Label mapping (e.g., TG->0, WT->1)
    preferred = [g.strip() for g in args.preferred_group_order.split(",") if g.strip()]
    g2y = make_label_mapping(groups, preferred_order=preferred)
    labels = [g2y[g] for g in groups]
    y2g = {v: k for k, v in g2y.items()}
    print(f"Groups mapping: {g2y}")

    if len(g2y) != 2:
        raise ValueError(f"This analysis script currently supports binary classification only. Found {len(g2y)} groups: {sorted(g2y.keys())}")

    # Infer latent_dim if not given: from embeddings file if available
    latent_dim = args.latent_dim
    if latent_dim is None and args.emb_glob:
        emb_files = sorted(glob.glob(args.emb_glob))
        if emb_files and pd is not None:
            df = pd.read_csv(emb_files[0])
            z_cols = [c for c in df.columns if c.startswith("z")]
            if z_cols:
                latent_dim = len(z_cols)
    if latent_dim is None:
        latent_dim = 512  # your current default

    # Import arch and create dataset
    arch = import_arch(args.arch_file)
    DatasetCls = getattr(arch, "ExcelImageDataset", None)
    ModelCls = getattr(arch, "ConvAutoencoderWithAttention", None)
    if DatasetCls is None or ModelCls is None:
        raise AttributeError("arch_file must define ExcelImageDataset and ConvAutoencoderWithAttention")

    dataset = DatasetCls(codes=codes, labels=labels, timepoint_dirs=tp_dirs, transform=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Infer H,W from data
    images0, _ = next(iter(loader))
    _, T, C, H, W = images0.shape
    print(f"Data: subjects={len(dataset)} timepoints={T} channels={C} H={H} W={W}")


    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_glob = args.model_glob
    if model_glob is None:
        if args.models_dir is None:
            raise ValueError("Provide either --model_glob or --models_dir")
        model_glob = os.path.join(args.models_dir, args.model_pattern)
    model_files = sorted(glob.glob(model_glob))
    if not model_files:
        raise FileNotFoundError(f"No models matched: {model_glob}")

    models = []
    for mp in model_files:
        m = ModelCls(in_channels=1, latent_dim=latent_dim, num_classes=2).to(device).eval()

        # Load checkpoint (support both raw state_dict and wrapped dicts)
        state = torch.load(mp, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                state = state["model_state_dict"]

        # IMPORTANT: some architectures create encoder.fc / decoder layers lazily on first forward,
        # so we must run a dummy forward before load_state_dict to register those parameters.
        with torch.no_grad():
            try:
                dummy = torch.zeros((1, T, C, H, W), device=device)
                _ = m(dummy)
            except Exception:
                # Fallback to 4D if the model expects no explicit time dimension
                try:
                    dummy = torch.zeros((1, C, H, W), device=device)
                    _ = m(dummy)
                except Exception:
                    pass

        try:
            m.load_state_dict(state, strict=True)
        except RuntimeError as e:
            # As a last resort, allow partial load, but warn loudly
            print(f"[WARN] strict load failed for {mp}: {e}")
            m.load_state_dict(state, strict=False)

        models.append(m)


    # -----------------------------
    # A) Rank most discriminative latent dimensions (model-agnostic summary)
    # -----------------------------
    # Axis extent for plots (excitation/emission units if available)
    extent, origin = infer_axis_extent(tp_dirs[0], codes[0])

    # Load physical axes from the first sample (for ROI coordinate reporting)
    emission_axis = None
    excitation_axis = None
    try:
        mat0, emission_axis, excitation_axis = load_excel_with_axes(resolve_excel_path(codes[0], tp_dirs[0]))
    except Exception:
        emission_axis, excitation_axis = None, None

    z_mean, y_all, w, auc_per_dim = rank_latent_dims_with_lr(models, loader, device)
    order = np.argsort(np.abs(w))[::-1]

    k_report = int(min(args.max_dims_report, len(order)))
    top_dims = order[:k_report].tolist()

    # Save latent-dim ranking
    if pd is None:
        raise ImportError("pandas is required: pip install pandas")
    rows = []
    for rank_i, d in enumerate(top_dims, start=1):
        rows.append({
            "rank": rank_i,
            "dim": int(d),
            "weight": float(w[d]),
            "abs_weight": float(abs(w[d])),
            "sign": float(np.sign(w[d]) if w[d] != 0 else 1.0),
            "auc_univariate": float(auc_per_dim[d]) if np.isfinite(auc_per_dim[d]) else np.nan,
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "latent_dim_ranking.csv"), index=False)

    # Export per-subject latent targets for top-K dims
    # This file is intended for downstream deterministic proxy construction (ridge calibration).
    k_tgt = int(min(args.top_k_latent_dims, len(top_dims)))
    dims_tgt = top_dims[:k_tgt]
    if k_tgt > 0:
        if pd is None:
            raise ImportError("pandas is required: pip install pandas")
        if len(codes) != z_mean.shape[0]:
            raise ValueError(f"Length mismatch: codes={len(codes)} but z_mean has {z_mean.shape[0]} rows")
        if len(groups) != z_mean.shape[0]:
            raise ValueError(f"Length mismatch: groups={len(groups)} but z_mean has {z_mean.shape[0]} rows")
        y_int = np.asarray(y_all).astype(int)
        df_tgt = pd.DataFrame({
            "subject": [str(c) for c in codes],
            "code": [str(c) for c in codes],
            "group": [str(g) for g in groups],
            "y": y_int,
        })
        for d in dims_tgt:
            df_tgt[f"z_dim_{int(d)}"] = z_mean[:, int(d)]
        out_tgt = os.path.join(args.outdir, "latent_targets_topdims.csv")
        df_tgt.to_csv(out_tgt, index=False)
        print(f"Saved latent targets for top dims to: {out_tgt}")


    # -----------------------------
    # B) Per-dimension attribution maps + ROIs + deterministic ROI features
    # -----------------------------
    # Use the most discriminative dims for attribution (typically you want <= 10 on slides)
    k_attr = int(min(args.top_k_latent_dims, len(top_dims)))
    dims_attr = top_dims[:k_attr]
    signs = {int(d): (1.0 if w[d] >= 0 else -1.0) for d in dims_attr}

    attr_maps = per_dim_gradient_attribution(models, loader, device, dims=dims_attr, signs=signs)

    roi_rows = []
    corr_rows = []

    for d in dims_attr:
        amap = attr_maps[int(d)]
        amap_disp = amap[::-1, :] if origin == "lower" else amap
        out_png = os.path.join(args.outdir, f"latent_dim_{int(d):04d}_attribution.png")
        plot_map(amap_disp, f"Attribution |d z[{int(d)}]/dX| (signed by LR weight)", out_png,
                 extent=extent, origin=origin, xlabel="Excitation", ylabel="Emission")

        roi = roi_from_map(amap, emission_axis=emission_axis, excitation_axis=excitation_axis, q=args.roi_quantile)
        roi["dim"] = int(d)
        roi["lr_weight"] = float(w[d])
        roi["abs_lr_weight"] = float(abs(w[d]))
        roi_rows.append(roi)

        if not roi.get("empty", False) and "y0" in roi:
            feats = compute_roi_features(dataset, tp_dirs, roi, use_timepoint_means=True)
            df_feat = pd.DataFrame(feats)
            z_d = z_mean[:, int(d)].astype(float)
            # Align by code order (dataset/codes order equals z_mean order in this script)
            # Correlations for basic deterministic features
            for col in [c for c in df_feat.columns if c.startswith("ROI_")]:
                x = df_feat[col].values.astype(float)
                if np.std(x) == 0 or np.std(z_d) == 0:
                    r = np.nan
                else:
                    r = float(np.corrcoef(x, z_d)[0, 1])
                corr_rows.append({"dim": int(d), "feature": col, "pearson_r": r})

            # Save per-dim ROI feature table (for downstream deterministic algorithm development)
            df_feat.to_csv(os.path.join(args.outdir, f"latent_dim_{int(d):04d}_roi_features.csv"), index=False)

    pd.DataFrame(roi_rows).to_csv(os.path.join(args.outdir, "latent_dim_rois.csv"), index=False)
    pd.DataFrame(corr_rows).to_csv(os.path.join(args.outdir, "roi_feature_correlations_vs_latent.csv"), index=False)
    print(f"Loaded {len(models)} model(s)")
    print(f"Loaded {len(models)} model(s)")


    # 1) Reconstruction maps
    class_mean = compute_reconstruction_error_maps(models, loader, device, label_names=y2g)
    # save per class
    for c, arr in class_mean.items():
        name = y2g.get(c, str(c))
        # flip if using extent/origin lower
        arr_disp = arr[::-1, :] if origin == "lower" else arr
        plot_map(arr_disp, f"Consensus reconstruction MSE ({name})", os.path.join(args.outdir, f"recon_mse_{name}.png"),
                 extent=extent, origin=origin, xlabel="Excitation", ylabel="Emission")


    # difference POS-NEG if binary
    # By default POS is args.pos_group (e.g., ALS); NEG is the other group.
    if len(g2y) == 2 and args.pos_group in g2y:
        pos = args.pos_group
        neg = [g for g in g2y.keys() if g != pos][0]
        pos_map = class_mean[g2y[pos]]
        neg_map = class_mean[g2y[neg]]
        diff = pos_map - neg_map
        diff_disp = diff[::-1, :] if origin == "lower" else diff
        plot_map(
            diff_disp,
            f"Reconstruction MSE diff ({pos} - {neg})",
            os.path.join(args.outdir, f"recon_mse_diff_{pos}-{neg}.png"),
            extent=extent, origin=origin, xlabel="Excitation", ylabel="Emission"
        )

    # 2) Gradient importance
    grad = compute_gradient_importance(models, loader, device, top_k_dims=args.top_k_latent_dims)
    grad_disp = grad[::-1, :] if origin == "lower" else grad
    plot_map(grad_disp, "Gradient input importance (top discriminative latent dims)", os.path.join(args.outdir, "gradient_importance.png"),
             extent=extent, origin=origin, xlabel="Excitation", ylabel="Emission")

    # 3) Linear probe importance (pixels -> margin)
    pos_class = g2y.get(args.pos_group, 0)
    probe = compute_linear_probe_importance(models, loader, device, pos_class=pos_class)
    probe_disp = probe[::-1, :] if origin == "lower" else probe
    plot_map(probe_disp, f"Linear probe importance (pixels -> margin for {args.pos_group})",
             os.path.join(args.outdir, "linear_probe_importance.png"),
             extent=extent, origin=origin, xlabel="Excitation", ylabel="Emission")

    print(f"Done. Outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()