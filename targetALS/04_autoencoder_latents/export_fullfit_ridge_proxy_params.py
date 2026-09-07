import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# -----------------------------
# I/O helpers
# -----------------------------
def parse_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_eem_xlsx_robust(path: str, expected=(512, 71)):
    """
    Robustly load EEM Excel and coerce to (512,71).
    Handles common cases:
      - (513,72): header row + emission col -> drop [0] row and [0] col
      - (512,72): has emission col but no header row -> drop first col
      - (513,71): header row but no emission col -> drop first row
      - (512,71): already ok
    """
    df = pd.read_excel(path, header=None, engine="openpyxl")
    arr = df.to_numpy()

    # Try common trims
    candidates = []
    candidates.append(arr)

    if arr.shape[0] >= 513:
        candidates.append(arr[1:, :])
    if arr.shape[1] >= 72:
        candidates.append(arr[:, 1:])
    if arr.shape[0] >= 513 and arr.shape[1] >= 72:
        candidates.append(arr[1:, 1:])

    # Pick first that matches expected after trimming
    for c in candidates:
        if c.shape == expected:
            arr = c
            break

    if arr.shape != expected:
        raise ValueError(f"Unexpected EEM shape {arr.shape} for file {path}. Expected {expected} (after robust trimming).")

    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Min-max normalize to [0,1] per file (matches your pipeline diagnostics)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    arr = arr - mn
    if mx - mn > 0:
        arr = arr / (mx - mn)
    return arr


def build_file_index(tp_dir: Path):
    files = sorted(tp_dir.glob("*.xlsx"))
    return files


def match_code_to_file(code: str, files):
    """
    Try to match sample code to a unique xlsx file within a timepoint directory.
    Strategy:
      1) filename startswith code
      2) exact token match (split on non-alnum)
      3) substring contains code
    """
    code_str = str(code)

    # 1) startswith
    hits = [f for f in files if f.name.startswith(code_str)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # pick shortest (most exact)
        hits = sorted(hits, key=lambda p: len(p.name))
        return hits[0]

    # 2) token match
    pat = re.compile(r"[A-Za-z0-9\.]+")
    for f in files:
        toks = pat.findall(f.stem)
        if code_str in toks:
            return f

    # 3) substring
    hits = [f for f in files if code_str in f.name]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits = sorted(hits, key=lambda p: len(p.name))
        return hits[0]

    return None


def load_dataset(tp_dirs, labels_csv, id_col="code", label_col="group"):
    lab = pd.read_csv(labels_csv)
    if id_col not in lab.columns:
        raise ValueError(f"labels_csv missing id_col='{id_col}'. Columns: {list(lab.columns)}")
    if label_col not in lab.columns:
        raise ValueError(f"labels_csv missing label_col='{label_col}'. Columns: {list(lab.columns)}")

    codes = lab[id_col].astype(str).tolist()
    y = lab[label_col].astype(str).tolist()

    tp_dirs = [Path(d) for d in tp_dirs]
    tp_files = [build_file_index(d) for d in tp_dirs]

    mats = []
    valid_codes = []
    valid_y = []

    for code, yy in zip(codes, y):
        per_tp = []
        ok = True
        for d, files in zip(tp_dirs, tp_files):
            f = match_code_to_file(code, files)
            if f is None:
                ok = False
                break
            per_tp.append(load_eem_xlsx_robust(str(f)))
        if not ok:
            continue
        stack = np.stack(per_tp, axis=0)  # (3,512,71)
        mats.append(stack)
        valid_codes.append(code)
        valid_y.append(yy)

    if len(mats) == 0:
        raise RuntimeError("No samples loaded. Check code<->file matching or labels_csv.")

    X = np.stack(mats, axis=0)  # (N,3,512,71)
    return X, np.array(valid_y), valid_codes


# -----------------------------
# Model/latent helpers
# -----------------------------
def load_arch_module(arch_file: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("arch_mod", arch_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pick_model_class(mod):
    # Prefer known name from your logs
    if hasattr(mod, "ConvAutoencoderWithAttention"):
        return getattr(mod, "ConvAutoencoderWithAttention")

    # fallback: first nn.Module subclass defined in module
    for name in dir(mod):
        obj = getattr(mod, name)
        try:
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                return obj
        except Exception:
            pass
    raise RuntimeError("Could not find a torch.nn.Module model class in arch_file.")


def instantiate_model(ModelClass, latent_dim=512):
    """
    FIXED: Explicitly handles ConvAutoencoderWithAttention signature.
    """
    # 1. Try specific signature (in_channels, latent_dim, num_classes)
    try:
        return ModelClass(in_channels=1, latent_dim=latent_dim, num_classes=2)
    except TypeError:
        pass

    # 2. Try just latent_dim
    try:
        return ModelClass(latent_dim=latent_dim)
    except TypeError:
        pass
        
    # 3. Try empty init
    try:
        return ModelClass()
    except TypeError as e:
        raise RuntimeError(f"Could not instantiate model class {ModelClass.__name__}: {e}")


def load_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unrecognized checkpoint format for {ckpt_path}")


@torch.no_grad()
def get_latent(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # 1) encode() method
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        try:
            z = model.encode(x)
            if isinstance(z, (tuple, list)):
                z = z[-1]
            return z
        except: pass

    # 2) forward returns tuple/list (recon, z) or similar
    try:
        out = model(x)
        if isinstance(out, (tuple, list)):
            # choose first 2D tensor as latent
            for t in out:
                if isinstance(t, torch.Tensor) and t.ndim == 2:
                    return t
            return out[-1] if isinstance(out[-1], torch.Tensor) else out[0]
    except: pass

    # 3) encoder attribute - Manual forward
    if hasattr(model, "encoder"):
        # If input is (N, 3, H, W) and encoder expects (N, 1, H, W), iterate
        if x.dim() == 4 and x.shape[1] == 3: 
             z_stack = []
             for t in range(x.shape[1]):
                 xt = x[:, t].unsqueeze(1)
                 z_stack.append(model.encoder(xt))
             z_stack = torch.stack(z_stack, dim=1)
             
             if hasattr(model, "attention"):
                 z, _ = model.attention(z_stack)
                 return z
             else:
                 return torch.mean(z_stack, dim=1)
        else:
            return model.encoder(x)

    raise RuntimeError("Could not extract latent representation (no encode(), no tuple output, no encoder).")


def compute_consensus_latents(X_np, arch_file, model_dir, latent_dim=512, device="cpu"):
    mod = load_arch_module(arch_file)
    ModelClass = pick_model_class(mod)

    model_paths = sorted(Path(model_dir).glob("best_model_fold*.pth"))
    if len(model_paths) == 0:
        raise RuntimeError(f"No models found in {model_dir} matching best_model_fold*.pth")

    X = torch.from_numpy(X_np).to(torch.float32).to(device)

    Zs = []
    for mp in model_paths:
        model = instantiate_model(ModelClass, latent_dim=latent_dim).to(device)
        sd = load_state_dict(str(mp))
        model.load_state_dict(sd, strict=False)
        model.eval()
        z = get_latent(model, X).detach().cpu().numpy()  # (N,latent_dim)
        Zs.append(z)

    # sign-align to first fold, per-dimension
    Zref = Zs[0]
    Z_aligned = [Zref]
    for Z in Zs[1:]:
        Z2 = Z.copy()
        for d in range(Z2.shape[1]):
            a = Zref[:, d]
            b = Z2[:, d]
            # correlation sign
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            r = np.corrcoef(a, b)[0, 1]
            if np.isfinite(r) and r < 0:
                Z2[:, d] *= -1.0
        Z_aligned.append(Z2)

    Z_stack = np.stack(Z_aligned, axis=0)  # (F,N,D)
    Z_agg = np.mean(Z_stack, axis=0)       # (N,D)
    return Z_agg


# -----------------------------
# ROI + Ridge helpers
# -----------------------------
def corr_map(z, X_tp_flat):
    """
    z: (N,)
    X_tp_flat: (N,P)
    returns corr: (P,)
    """
    z = z.astype(np.float64)
    X = X_tp_flat.astype(np.float64)

    z0 = z - np.mean(z)
    z_std = np.std(z0)
    if z_std == 0:
        return np.zeros(X.shape[1], dtype=np.float32)

    X0 = X - np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X0, axis=0)
    denom = (X_std * z_std)
    denom[denom == 0] = np.nan

    cov = (X0.T @ z0) / max(1, (X.shape[0] - 1))
    r = cov / denom
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return r.astype(np.float32)


def make_roi_from_corr(corr_vec, roi_percentile, min_pixels=50):
    """
    corr_vec: (P,)
    ROI is top percentile of |corr|
    """
    absr = np.abs(corr_vec)
    # robust thresholding with fallback if too small
    pct = roi_percentile
    while True:
        thr = np.percentile(absr, pct)
        idx = np.where(absr >= thr)[0]
        if idx.size >= min_pixels or pct <= 50:
            return idx.astype(np.int32), float(thr), float(pct)
        pct -= 2.5


def choose_alpha_median_oof(results_dir: Path, feature_name: str):
    p = results_dir / "ridge_alpha_selected_oof.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # try common column names
    col_feat = None
    for c in ["feature", "proxy_feature", "name"]:
        if c in df.columns:
            col_feat = c
            break
    if col_feat is None:
        return None
    col_alpha = None
    for c in ["alpha", "ridge_alpha", "selected_alpha"]:
        if c in df.columns:
            col_alpha = c
            break
    if col_alpha is None:
        return None

    vals = df.loc[df[col_feat] == feature_name, col_alpha].to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def choose_alpha_cv(X, y, alpha_grid, kfold=5, seed=1337, metric="r2"):
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    best_a = None
    best_s = -np.inf

    for a in alpha_grid:
        scores = []
        for tr, va in kf.split(X):
            Xt, Xv = X[tr], X[va]
            yt, yv = y[tr], y[va]
            model = Ridge(alpha=a, fit_intercept=True)
            model.fit(Xt, yt)
            pred = model.predict(Xv)
            if metric == "mse":
                s = -np.mean((pred - yv) ** 2)
            else:
                # r2
                ss_res = np.sum((yv - pred) ** 2)
                ss_tot = np.sum((yv - np.mean(yv)) ** 2)
                s = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            scores.append(s)
        s_mean = float(np.mean(scores))
        if s_mean > best_s:
            best_s = s_mean
            best_a = a

    return float(best_a), float(best_s)


def standardize_X(X):
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Your v12 results folder (e.g., proxy_latent_results_v12n)")
    ap.add_argument("--tp_dirs", required=True, help="Comma-separated timepoint dirs (out_0h,out_6h,out_24h)")
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--arch_file", required=True)
    ap.add_argument("--model_dir", required=True)

    ap.add_argument("--id_col", default="code")
    ap.add_argument("--label_col", default="group")

    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--roi_percentile", type=float, default=90.0)
    ap.add_argument("--min_roi_pixels", type=int, default=50)

    ap.add_argument("--feature_source", default="fs_counts",
                   choices=["fs_counts", "oof_matrix"],
                   help="fs_counts: use feature_selection_counts.csv to pick features; oof_matrix: use proxy_feature_matrix_ridge_oof.csv columns.")
    ap.add_argument("--min_fs_count", type=int, default=1,
                   help="If feature_source=fs_counts, keep features with count >= this.")
    ap.add_argument("--top_n_features", type=int, default=0,
                   help="If >0 and feature_source=fs_counts, keep top-N by selection count.")

    ap.add_argument("--alpha_mode", default="median_oof", choices=["median_oof", "cv", "fixed"])
    ap.add_argument("--ridge_alpha", type=float, default=1.0, help="Used if alpha_mode=fixed")
    ap.add_argument("--alpha_grid", default="0.01,0.1,1,10,100", help="Used if alpha_mode=cv")
    ap.add_argument("--alpha_kfold", type=int, default=5)
    ap.add_argument("--alpha_metric", default="r2", choices=["r2", "mse"])

    ap.add_argument("--standardize", action="store_true", help="Standardize ROI pixels before ridge; exports mean/std.")
    ap.add_argument("--out_name", default="ridge_proxy_params_fullfit.json")

    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    tp_dirs = parse_csv_list(args.tp_dirs)
    tp_names = [Path(d).name for d in tp_dirs]

    # 1) decide which proxy features to export
    if args.feature_source == "fs_counts":
        fsc = results_dir / "feature_selection_counts.csv"
        if not fsc.exists():
            raise FileNotFoundError(f"Missing {fsc}. Run v12 pipeline first OR use --feature_source oof_matrix.")
        dfc = pd.read_csv(fsc)
        # expect columns: feature,count (robust)
        col_feat = "feature" if "feature" in dfc.columns else dfc.columns[0]
        col_cnt = "count" if "count" in dfc.columns else dfc.columns[1]
        dfc = dfc.sort_values(col_cnt, ascending=False)
        dfc = dfc[dfc[col_cnt] >= args.min_fs_count]
        if args.top_n_features and args.top_n_features > 0:
            dfc = dfc.head(args.top_n_features)
        features = dfc[col_feat].astype(str).tolist()
    else:
        pm = results_dir / "proxy_feature_matrix_ridge_oof.csv"
        if not pm.exists():
            raise FileNotFoundError(f"Missing {pm}.")
        dfm = pd.read_csv(pm, nrows=1)
        features = [c for c in dfm.columns if c not in ["code", "y", "label", "group"]]

    # parse feature names like dim332_24h
    parsed = []
    pat = re.compile(r"^dim(\d+)_([A-Za-z0-9]+)$")
    for fn in features:
        m = pat.match(fn)
        if not m:
            continue
        dim = int(m.group(1))
        tp = m.group(2)
        if tp not in tp_names:
            # allow suffix match (e.g. "0h" vs "out_0h")
            # if your tp names are out_0h style, tp could be "0h" from older scripts
            # try mapping:
            matches = [t for t in tp_names if t.endswith(tp)]
            if len(matches) == 1:
                tp = matches[0]
            else:
                continue
        tp_idx = tp_names.index(tp)
        parsed.append((fn, dim, tp, tp_idx))

    if len(parsed) == 0:
        raise RuntimeError("No features parsed. Expected names like dim332_24h in the selected feature list.")

    print(f"[INFO] Exporting {len(parsed)} proxy features.")
    print(f"[INFO] Timepoints: {tp_names}")

    # 2) load training dataset (EEMs)
    print("[INFO] Loading EEM dataset...")
    X_eem, y_str, codes = load_dataset(tp_dirs, args.labels_csv, id_col=args.id_col, label_col=args.label_col)
    N = X_eem.shape[0]
    print(f"[INFO] Loaded X: {X_eem.shape} (N={N})")

    # 3) compute consensus latents (targets)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU.")
        device = "cpu"

    print("[INFO] Computing consensus latent targets (z_agg)...")
    Z = compute_consensus_latents(X_eem, args.arch_file, args.model_dir, latent_dim=args.latent_dim, device=device)
    if Z.shape[0] != N:
        raise RuntimeError("Latent targets shape mismatch.")
    print(f"[INFO] z_agg shape: {Z.shape}")

    # 4) fit ridge per feature (fullfit) and export params
    alpha_grid = parse_float_list(args.alpha_grid)

    out = {
        "version": "fullfit_ridge_proxies_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "tp_dirs": tp_dirs,
        "tp_names": tp_names,
        "eem_shape": [512, 71],
        "roi_percentile": float(args.roi_percentile),
        "min_roi_pixels": int(args.min_roi_pixels),
        "alpha_mode": args.alpha_mode,
        "alpha_grid": alpha_grid,
        "alpha_kfold": int(args.alpha_kfold),
        "alpha_metric": args.alpha_metric,
        "standardize": bool(args.standardize),
        "features": []
    }

    # Pre-flatten per timepoint once
    X_flat_by_tp = []
    for t in range(X_eem.shape[1]):
        Xtp = X_eem[:, t, :, :]  # (N,512,71)
        X_flat_by_tp.append(Xtp.reshape(N, -1))  # (N,36352)

    safe_mkdir(results_dir / "fullfit_params")

    for (fname, dim, tp, tp_idx) in parsed:
        print(f"[FIT] {fname} (dim={dim}, tp={tp})")

        y_lat = Z[:, dim].astype(np.float32)

        # corr map + ROI
        cvec = corr_map(y_lat, X_flat_by_tp[tp_idx])  # (P,)
        roi_idx, thr, pct_used = make_roi_from_corr(cvec, args.roi_percentile, min_pixels=args.min_roi_pixels)

        Xroi = X_flat_by_tp[tp_idx][:, roi_idx].astype(np.float32)

        # choose alpha
        alpha = None
        alpha_note = None
        if args.alpha_mode == "fixed":
            alpha = float(args.ridge_alpha)
            alpha_note = "fixed"
        elif args.alpha_mode == "median_oof":
            alpha = choose_alpha_median_oof(results_dir, fname)
            if alpha is None:
                alpha_note = "median_oof_missing->cv"
                alpha, score = choose_alpha_cv(Xroi, y_lat, alpha_grid, kfold=args.alpha_kfold, metric=args.alpha_metric)
            else:
                alpha_note = "median_oof"
        else:
            alpha_note = "cv"
            alpha, score = choose_alpha_cv(Xroi, y_lat, alpha_grid, kfold=args.alpha_kfold, metric=args.alpha_metric)

        # standardize (optional)
        if args.standardize:
            Xfit, mu, sd = standardize_X(Xroi)
        else:
            Xfit = Xroi
            mu = None
            sd = None

        # fit full ridge
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xfit, y_lat)
        coef = model.coef_.astype(np.float32)
        intercept = float(model.intercept_)

        # quick fit diagnostics
        pred = model.predict(Xfit).astype(np.float32)
        mse = float(np.mean((pred - y_lat) ** 2))
        # r2
        ss_res = float(np.sum((y_lat - pred) ** 2))
        ss_tot = float(np.sum((y_lat - float(np.mean(y_lat))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        feat_obj = {
            "name": fname,
            "dim": int(dim),
            "tp": str(tp),
            "tp_index": int(tp_idx),
            "roi_linear_idx": roi_idx.astype(int).tolist(),
            "roi_abs_corr_threshold": float(thr),
            "roi_percentile_used": float(pct_used),
            "alpha": float(alpha),
            "alpha_source": alpha_note,
            "intercept": intercept,
            "coef": coef.tolist(),
            "standardize": bool(args.standardize),
            "x_mean": mu.astype(np.float32).tolist() if mu is not None else None,
            "x_std": sd.astype(np.float32).tolist() if sd is not None else None,
            "fullfit_mse": mse,
            "fullfit_r2": r2
        }
        out["features"].append(feat_obj)

    # 5) write params json
    out_path = results_dir / "fullfit_params" / args.out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] Wrote fullfit ridge params: {out_path}")

    # 6) also write fullfit proxy feature matrix for the TRAINING dataset (sanity check)
    feat_names = [fo["name"] for fo in out["features"]]
    Xproxy = np.zeros((N, len(out["features"])), dtype=np.float32)

    for j, fo in enumerate(out["features"]):
        tp_idx = fo["tp_index"]
        roi_idx = np.array(fo["roi_linear_idx"], dtype=np.int32)
        Xroi = X_flat_by_tp[tp_idx][:, roi_idx].astype(np.float32)
        if fo["standardize"]:
            mu = np.array(fo["x_mean"], dtype=np.float32)
            sd = np.array(fo["x_std"], dtype=np.float32)
            Xroi = (Xroi - mu) / sd
        coef = np.array(fo["coef"], dtype=np.float32)
        Xproxy[:, j] = Xroi @ coef + float(fo["intercept"])

    df_out = pd.DataFrame(Xproxy, columns=feat_names)
    df_out.insert(0, "code", codes)
    df_out.insert(1, "group", y_str.tolist())
    proxy_path = results_dir / "fullfit_params" / "proxy_feature_matrix_fullfit.csv"
    df_out.to_csv(proxy_path, index=False)
    print(f"[DONE] Wrote fullfit proxy matrix (training set): {proxy_path}")


if __name__ == "__main__":
    main()