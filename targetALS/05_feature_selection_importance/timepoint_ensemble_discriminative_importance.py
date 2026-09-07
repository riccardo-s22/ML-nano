"""timepoint_ensemble_discriminative_importance.py

Timepoint-separated ensemble input importance using discriminative latent dimensions.

For each timepoint t:
  For each fold-model m:
    - Compute subject latents z_t = encoder(X_t)
    - Fit LogisticRegression probe on z_t (standardized) -> class labels
    - Select top-K latent dims by |w_eff| where w_eff = w_scaled / scaler.scale_
    - Compute an attribution map in input space by backpropagating a weighted
      discriminative target:
          target = sum_d w_eff[d] * z_t[:, d]
      and taking |d target / d X_t| averaged over the dataset.
    - This yields one HxW importance map per model.

  Ensemble across models:
    - mean importance map
    - std (fold variability) map
    - stability map = mean / (std + eps)
    - model-support frequency map (how many folds mark a pixel as "top-q%")

Outputs (per timepoint subfolder):
  - ensemble_mean.npy / .png
  - ensemble_std.npy / .png
  - ensemble_stability.npy / .png
  - ensemble_support.npy / .png
  - per_model_top_latent_dims.csv
  - top_features.csv
  - ensemble_input_importance_report.txt

You must provide:
  - --arch_file: python defining ExcelImageDataset and ConvAutoencoderWithAttention
  - --model_glob: glob for fold checkpoints
  - --timepoints: comma-separated directories for timepoints (order must match training)
  - Either --labels_csv (columns code,group) OR --emb_glob (csv with code,group)

Example:
  python timepoint_ensemble_discriminative_importance.py \
    --arch_file conv_autoencoder_detailed.py \
    --model_glob "C:/.../best_model_fold*.pth" \
    --timepoints "C:/.../out_0h,C:/.../out_6h,C:/.../out_24h" \
    --labels_csv "C:/.../labels.csv" \
    --preferred_group_order "ALS,CTRL" \
    --outdir "C:/.../tp_importance"
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Soft imports (fail with helpful message later)
try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception:
    LogisticRegression = None
    StandardScaler = None


# ---------------------------------------------------------------------
# Excel axis helpers (for wavelength mapping)
# ---------------------------------------------------------------------

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
    """Return (data, emission_axis, excitation_axis) from xlsx with headers."""
    if load_workbook is None:
        raise ImportError("openpyxl is required to read .xlsx files")
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
    return os.path.join(tp_dir, f"{code}.xlsx")


def infer_axis_extent(tp_dir: str, code: str) -> Tuple[Optional[List[float]], str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (extent, origin, emission_axis, excitation_axis) for plotting."""
    try:
        _, emission, excitation = load_excel_with_axes(resolve_excel_path(code, tp_dir))
    except Exception:
        return None, "upper", None, None

    if len(emission) and len(excitation) and np.all(np.isfinite(emission)) and np.all(np.isfinite(excitation)):
        # We'll display with emission increasing upward -> flip vertically + origin='lower'
        em_disp = emission[::-1]
        extent = [float(excitation[0]), float(excitation[-1]), float(em_disp[0]), float(em_disp[-1])]
        return extent, "lower", emission, excitation

    return None, "upper", emission, excitation


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def import_arch(arch_file: str):
    spec = importlib.util.spec_from_file_location("archmod", arch_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import arch file: {arch_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_label_mapping(groups: List[str], preferred_order: Optional[List[str]] = None) -> Dict[str, int]:
    uniq = sorted(set(groups))
    if preferred_order:
        ordered = [g for g in preferred_order if g in uniq] + [g for g in uniq if g not in preferred_order]
    else:
        ordered = uniq
    return {g: i for i, g in enumerate(ordered)}


def load_codes_and_groups(emb_glob: Optional[str], labels_csv: Optional[str]) -> Tuple[List[str], List[str]]:
    if pd is None:
        raise ImportError("pandas is required")

    if labels_csv:
        df = pd.read_csv(labels_csv)
        cols = {c.lower(): c for c in df.columns}
        code_col = cols.get("code") or cols.get("id") or cols.get("sample") or cols.get("subject")
        group_col = cols.get("group") or cols.get("label") or cols.get("class")
        if code_col is None or group_col is None:
            raise ValueError(
                f"Could not infer code/group columns from {labels_csv}. "
                f"Need columns like code, group. Found: {list(df.columns)}"
            )
        codes = df[code_col].astype(str).tolist()
        groups = df[group_col].astype(str).tolist()
        return codes, groups

    if not emb_glob:
        raise ValueError("Provide either --labels_csv or --emb_glob")

    emb_files = sorted(glob.glob(emb_glob))
    if not emb_files:
        raise FileNotFoundError(f"No embeddings matched: {emb_glob}")
    df = pd.read_csv(emb_files[0])
    if "code" not in df.columns or "group" not in df.columns:
        raise ValueError(f"Embeddings CSV must have columns code, group. Found: {list(df.columns)[:30]}")
    return df["code"].astype(str).tolist(), df["group"].astype(str).tolist()


def plot_map(data: np.ndarray, title: str, out_png: str,
             extent: Optional[List[float]], origin: str,
             xlabel: str = "Excitation", ylabel: str = "Emission"):
    if plt is None:
        raise ImportError("matplotlib is required")

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


@dataclass
class ModelTPResult:
    model_path: str
    time_idx: int
    w_eff: np.ndarray           # (D,)
    top_dims: np.ndarray        # (K,)
    top_weights: np.ndarray     # (K,)
    attr_map: np.ndarray        # (H,W)


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)


# ---------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------

def collect_latents_single_tp(model, dataloader, device, time_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z, y) where Z is (N,D) encoder outputs at timepoint time_idx."""
    Z_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)  # (B,T,C,H,W)
            z_t = model.encoder(imgs[:, time_idx])  # (B,D)
            Z_list.append(z_t.detach().cpu().numpy())
            y_list.append(lbls.detach().cpu().numpy())

    Z = np.concatenate(Z_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(int)
    return Z, y


def fit_lr_probe(Z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, StandardScaler, LogisticRegression]:
    """Fit LR on standardized Z; return effective weights in raw Z space (w_eff)."""
    if LogisticRegression is None or StandardScaler is None:
        raise ImportError("scikit-learn is required (LogisticRegression, StandardScaler)")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Zs = scaler.fit_transform(Z)

    clf = LogisticRegression(class_weight="balanced", max_iter=4000, solver="liblinear")
    clf.fit(Zs, y)

    w_scaled = clf.coef_[0].astype(float)  # weights on standardized Z
    scale = scaler.scale_.astype(float).copy()
    scale[scale == 0] = 1.0
    w_eff = w_scaled / scale
    return w_eff, scaler, clf


def compute_attr_map_weighted_target(
    model,
    dataloader,
    device,
    time_idx: int,
    top_dims: np.ndarray,
    top_w_eff: np.ndarray,
) -> np.ndarray:
    """Compute |d (sum_d w[d]*z_d) / dX_t| averaged across batches."""
    # infer H,W
    imgs0, _ = next(iter(dataloader))
    _, _, _, H, W = imgs0.shape

    acc = np.zeros((H, W), dtype=np.float64)
    n = 0

    model.eval()
    w_t = torch.tensor(top_w_eff, device=device, dtype=torch.float32).view(1, -1)
    dims_t = torch.tensor(top_dims.astype(np.int64), device=device)

    for imgs, _ in dataloader:
        x = imgs[:, time_idx].to(device)
        x.requires_grad_(True)

        z = model.encoder(x)  # (B,D)
        z_sel = torch.index_select(z, dim=1, index=dims_t)  # (B,K)

        # Weighted discriminative target (sum over batch + dims)
        target = (z_sel * w_t).sum()

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        target.backward()
        g = x.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()  # (H,W)

        acc += g
        n += 1

        # free graph
        x = x.detach()

    return acc / max(n, 1)


def threshold_mask(m: np.ndarray, q: float) -> np.ndarray:
    thr = np.percentile(m, q)
    return (m >= thr)


def topk_pixels(m: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
    """Return list of (y,x,value) top-k by value, optionally restricted by mask."""
    if mask is None:
        flat = m.ravel()
        idx = np.argpartition(flat, -k)[-k:]
    else:
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return []
        vals = m[ys, xs]
        if len(vals) <= k:
            order = np.argsort(vals)[::-1]
            return [(int(ys[i]), int(xs[i]), float(vals[i])) for i in order]
        idx_local = np.argpartition(vals, -k)[-k:]
        idx = idx_local
        order = np.argsort(vals[idx])[::-1]
        return [(int(ys[idx[i]]), int(xs[idx[i]]), float(vals[idx[i]])) for i in order]

    order = np.argsort(flat[idx])[::-1]
    out = []
    H, W = m.shape
    for j in order:
        linear = int(idx[j])
        y = linear // W
        x = linear % W
        out.append((int(y), int(x), float(m[y, x])))
    return out


def save_report(
    out_path: str,
    tp_name: str,
    n_models: int,
    n_subjects: int,
    top_by_importance: List[Dict],
    top_by_stability: List[Dict],
    q_support: float,
    min_models: int,
):
    lines = []
    lines.append("=" * 70)
    lines.append("TIMEPOINT ENSEMBLE INPUT IMPORTANCE REPORT")
    lines.append(f"Timepoint: {tp_name}")
    lines.append(f"Subjects: {n_subjects}")
    lines.append(f"Analyzed {n_models} fold model(s)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("METHOD:")
    lines.append("- For each fold-model: LR probe on encoder latents at this timepoint")
    lines.append("- Select top-K discriminative latent dims (by |w|)")
    lines.append("- Input attribution: |d(sum w*z)/dX| at this timepoint")
    lines.append("- Ensemble across folds: mean (importance), std (variability), mean/std (stability)")
    lines.append("")
    lines.append(f"MODEL-SUPPORT MAP: pixel counted if in top {100-q_support:.1f}% importance within a fold")
    lines.append(f"CONSENSUS FILTER: reported peaks require support >= {min_models}/{n_models} models")
    lines.append("")

    def _fmt_entry(i: int, r: Dict) -> List[str]:
        pos = f"({r['y']}, {r['x']})"
        wl = ""
        if np.isfinite(r.get("emission", np.nan)) and np.isfinite(r.get("excitation", np.nan)):
            wl = f" [Em {r['emission']:.2f}, Ex {r['excitation']:.2f}]"
        return [
            f"{i}. Position {pos}{wl}",
            f"   Importance: {r['importance']:.6g}, Variability: {r['variability']:.6g}, Stability: {r['stability']:.6g}, Support: {r['support']}",
        ]

    lines.append("TOP FEATURES BY ENSEMBLE IMPORTANCE (support-filtered):")
    lines.append("-" * 70)
    for i, r in enumerate(top_by_importance, start=1):
        lines.extend(_fmt_entry(i, r))
    lines.append("")

    lines.append("MOST STABLE FEATURES (support-filtered):")
    lines.append("-" * 70)
    for i, r in enumerate(top_by_stability, start=1):
        lines.extend(_fmt_entry(i, r))
    lines.append("")

    lines.append("INTERPRETATION:")
    lines.append("- Importance: mean input attribution across folds")
    lines.append("- Variability: std across folds")
    lines.append("- Stability: mean / (std + eps); higher => important + consistent")
    lines.append("- Support: number of folds where pixel is in top-q importance")
    lines.append("")
    lines.append("NEXT STEPS:")
    lines.append("1) Convert top pixels into small ROIs (bounding boxes) and extract deterministic features")
    lines.append("2) Fit a simple model (LR/SVM) on those ROI features per timepoint")
    lines.append("3) Validate stability on held-out cohort or bootstrap resampling")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if torch is None:
        raise ImportError("PyTorch is required")

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch_file", required=True)
    ap.add_argument("--model_glob", required=True, help="Glob for fold checkpoints (e.g., best_model_fold*.pth)")
    ap.add_argument("--timepoints", required=True, help="Comma-separated timepoint directories in training order")
    ap.add_argument("--tp_names", default=None, help="Optional comma-separated names (defaults to folder basenames)")
    ap.add_argument("--outdir", default="tp_ensemble_importance")

    ap.add_argument("--labels_csv", default=None, help="CSV with code,group columns")
    ap.add_argument("--emb_glob", default=None, help="Fallback: glob to embeddings csv with code,group")

    ap.add_argument("--preferred_group_order", default="ALS,CTRL")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--latent_dim", type=int, default=None, help="Override latent dim (else inferred from embeddings or default 512)")

    ap.add_argument("--top_k_latent_dims", type=int, default=10, help="Top-K latent dims per fold probe")
    ap.add_argument("--support_quantile", type=float, default=99.0, help="Pixel is supported if >= this percentile in a fold map")
    ap.add_argument("--min_models_support", type=int, default=3, help="Require >= this many models to report a pixel")

    ap.add_argument("--top_n_features", type=int, default=20)
    ap.add_argument("--top_n_stable", type=int, default=10)
    ap.add_argument("--stability_importance_floor", type=float, default=50.0,
                    help="For stable list: only consider pixels with importance >= this percentile of mean map")

    ap.add_argument("--eps", type=float, default=1e-8)

    args = ap.parse_args()

    if pd is None:
        raise ImportError("pandas is required")

    os.makedirs(args.outdir, exist_ok=True)

    tp_dirs = [t.strip() for t in args.timepoints.split(",") if t.strip()]
    if not tp_dirs:
        raise ValueError("No timepoint directories provided")

    if args.tp_names:
        tp_names = [t.strip() for t in args.tp_names.split(",") if t.strip()]
        if len(tp_names) != len(tp_dirs):
            raise ValueError("--tp_names must match number of --timepoints")
    else:
        tp_names = [os.path.basename(os.path.normpath(t)) for t in tp_dirs]

    # codes + groups
    codes, groups = load_codes_and_groups(args.emb_glob, args.labels_csv)
    preferred = [g.strip() for g in args.preferred_group_order.split(",") if g.strip()]
    g2y = make_label_mapping(groups, preferred_order=preferred)
    if len(g2y) != 2:
        raise ValueError(f"Binary labels required; found groups: {sorted(g2y.keys())}")
    labels = [g2y[g] for g in groups]

    # infer latent dim if possible
    latent_dim = args.latent_dim
    if latent_dim is None and args.emb_glob:
        emb_files = sorted(glob.glob(args.emb_glob))
        if emb_files:
            df = pd.read_csv(emb_files[0])
            z_cols = [c for c in df.columns if c.startswith("z")]
            if z_cols:
                latent_dim = len(z_cols)
    if latent_dim is None:
        latent_dim = 512

    # import architecture
    arch = import_arch(args.arch_file)
    DatasetCls = getattr(arch, "ExcelImageDataset", None)
    ModelCls = getattr(arch, "ConvAutoencoderWithAttention", None)
    if DatasetCls is None or ModelCls is None:
        raise AttributeError("arch_file must define ExcelImageDataset and ConvAutoencoderWithAttention")

    dataset = DatasetCls(codes=codes, labels=labels, timepoint_dirs=tp_dirs, transform=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    imgs0, _ = next(iter(loader))
    _, T, C, H, W = imgs0.shape
    if T != len(tp_dirs):
        print(f"[WARN] dataset timepoints={T} but dirs={len(tp_dirs)}; continuing")

    # axis mapping info (use first timepoint + first subject)
    extent, origin, emission_axis, excitation_axis = infer_axis_extent(tp_dirs[0], codes[0])

    # load model checkpoints
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_files = sorted(glob.glob(args.model_glob))
    if not model_files:
        raise FileNotFoundError(f"No models matched: {args.model_glob}")

    models = []
    for mp in model_files:
        m = ModelCls(in_channels=1, latent_dim=latent_dim, num_classes=2).to(device).eval()

        # Ensure lazy layers are initialized
        with torch.no_grad():
            dummy = torch.zeros((1, T, C, H, W), device=device)
            try:
                _ = m(dummy)
            except Exception:
                # some models might expect 4D input
                _ = m(dummy[:, 0])

        state = torch.load(mp, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            elif "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                state = state["model_state_dict"]

        try:
            m.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f"[WARN] strict load failed for {mp}: {e}")
            m.load_state_dict(state, strict=False)

        models.append(m)

    n_models = len(models)
    n_subjects = len(dataset)

    # output summary csv across timepoints
    all_top_rows = []

    for time_idx, tp_name in enumerate(tp_names):
        tp_out = os.path.join(args.outdir, f"tp_{time_idx}_{tp_name}")
        os.makedirs(tp_out, exist_ok=True)

        per_model_rows = []
        per_model_maps = []
        support_masks = []

        for mi, (model, mp) in enumerate(zip(models, model_files)):
            Z, y = collect_latents_single_tp(model, loader, device, time_idx)
            w_eff, scaler, clf = fit_lr_probe(Z, y)

            K = int(min(args.top_k_latent_dims, w_eff.shape[0]))
            top_dims = np.argsort(np.abs(w_eff))[::-1][:K]
            top_w = w_eff[top_dims]

            # log per-model dims
            for rank, (d, wv) in enumerate(zip(top_dims, top_w), start=1):
                per_model_rows.append({
                    "timepoint": tp_name,
                    "time_idx": time_idx,
                    "model_index": mi,
                    "model_path": mp,
                    "rank": rank,
                    "latent_dim": int(d),
                    "w_eff": float(wv),
                    "abs_w_eff": float(abs(wv)),
                })

            # attribution map for this model/timepoint
            amap = compute_attr_map_weighted_target(model, loader, device, time_idx, top_dims, top_w)
            per_model_maps.append(amap)
            support_masks.append(threshold_mask(amap, args.support_quantile))

        # Save per-model dim ranking table
        pd.DataFrame(per_model_rows).to_csv(os.path.join(tp_out, "per_model_top_latent_dims.csv"), index=False)

        # Ensemble
        stack = np.stack(per_model_maps, axis=0)  # (M,H,W)
        mean_map = stack.mean(axis=0)
        std_map = stack.std(axis=0)
        stability_map = mean_map / (std_map + args.eps)

        support = np.stack(support_masks, axis=0).sum(axis=0).astype(int)  # (H,W)

        # Save arrays
        np.save(os.path.join(tp_out, "ensemble_mean.npy"), mean_map)
        np.save(os.path.join(tp_out, "ensemble_std.npy"), std_map)
        np.save(os.path.join(tp_out, "ensemble_stability.npy"), stability_map)
        np.save(os.path.join(tp_out, "ensemble_support.npy"), support)

        # Plot (flip vertically if using origin='lower')
        mean_disp = mean_map[::-1, :] if origin == "lower" else mean_map
        std_disp = std_map[::-1, :] if origin == "lower" else std_map
        stab_disp = stability_map[::-1, :] if origin == "lower" else stability_map
        sup_disp = support[::-1, :] if origin == "lower" else support

        plot_map(mean_disp, f"{tp_name}: Ensemble mean importance", os.path.join(tp_out, "ensemble_mean.png"), extent, origin)
        plot_map(std_disp, f"{tp_name}: Fold variability (std)", os.path.join(tp_out, "ensemble_std.png"), extent, origin)
        plot_map(stab_disp, f"{tp_name}: Stability (mean/std)", os.path.join(tp_out, "ensemble_stability.png"), extent, origin)
        plot_map(sup_disp, f"{tp_name}: Model support (count)", os.path.join(tp_out, "ensemble_support.png"), extent, origin,
                 xlabel="Excitation", ylabel="Emission")

        # Build support-filtered mask
        support_mask = (support >= args.min_models_support)

        # Top by importance (support-filtered)
        top_imp = topk_pixels(mean_map, args.top_n_features, mask=support_mask)

        # Top by stability: also require importance above a floor percentile
        floor_thr = np.percentile(mean_map, args.stability_importance_floor)
        stable_mask = support_mask & (mean_map >= floor_thr)
        top_stab = topk_pixels(stability_map, args.top_n_stable, mask=stable_mask)

        def _row(yxv, metric_map):
            y, x, _v = yxv
            em = float(emission_axis[y]) if emission_axis is not None and y < len(emission_axis) else np.nan
            ex = float(excitation_axis[x]) if excitation_axis is not None and x < len(excitation_axis) else np.nan
            return {
                "timepoint": tp_name,
                "time_idx": time_idx,
                "y": int(y),
                "x": int(x),
                "emission": em,
                "excitation": ex,
                "importance": float(mean_map[y, x]),
                "variability": float(std_map[y, x]),
                "stability": float(stability_map[y, x]),
                "support": int(support[y, x]),
            }

        top_imp_rows = [_row(t, mean_map) for t in top_imp]
        top_stab_rows = [_row(t, stability_map) for t in top_stab]

        # Save CSV of top features (combined)
        top_df = pd.DataFrame(top_imp_rows)
        top_df.to_csv(os.path.join(tp_out, "top_features_by_importance.csv"), index=False)

        stab_df = pd.DataFrame(top_stab_rows)
        stab_df.to_csv(os.path.join(tp_out, "top_features_by_stability.csv"), index=False)

        # Save unified top-features table (for downstream merging)
        all_top_rows.extend(top_imp_rows)

        # Write a human-readable report
        save_report(
            out_path=os.path.join(tp_out, "ensemble_input_importance_report.txt"),
            tp_name=tp_name,
            n_models=n_models,
            n_subjects=n_subjects,
            top_by_importance=top_imp_rows,
            top_by_stability=top_stab_rows,
            q_support=args.support_quantile,
            min_models=args.min_models_support,
        )

        print(f"[OK] {tp_name}: wrote outputs to {os.path.abspath(tp_out)}")

    # Save all-top-features across timepoints
    pd.DataFrame(all_top_rows).to_csv(os.path.join(args.outdir, "top_features_all_timepoints.csv"), index=False)
    print(f"Done. Root output: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
