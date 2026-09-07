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

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. I/O & Data Loading
# -----------------------------
def parse_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_eem_xlsx_robust(path: str, expected=(512, 71)):
    """Robust EEM loader (matches v12 pipeline)"""
    try:
        df = pd.read_excel(path, header=None, engine="openpyxl")
        arr = df.to_numpy()
        
        # Trim headers/indices if needed
        candidates = [arr]
        if arr.shape[0] >= 513: candidates.append(arr[1:, :])
        if arr.shape[1] >= 72: candidates.append(arr[:, 1:])
        if arr.shape[0] >= 513 and arr.shape[1] >= 72: candidates.append(arr[1:, 1:])
        
        for c in candidates:
            if c.shape == expected:
                arr = c
                break
        
        if arr.shape != expected:
            # Last ditch: crop
            padded = np.zeros(expected, dtype=np.float32)
            h, w = min(expected[0], arr.shape[0]), min(expected[1], arr.shape[1])
            padded[:h, :w] = arr[:h, :w]
            arr = padded

        arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Min-Max Normalize
        mn, mx = arr.min(), arr.max()
        arr = arr - mn
        if mx > mn: arr = arr / (mx - mn)
        return arr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(expected, dtype=np.float32)

def build_file_index(tp_dir: Path):
    return sorted(tp_dir.glob("*.xlsx"))

def match_code_to_file(code: str, files):
    code_str = str(code)
    # 1. Startswith
    hits = [f for f in files if f.name.startswith(code_str)]
    if len(hits) == 1: return hits[0]
    # 2. Token match
    pat = re.compile(r"[A-Za-z0-9\.]+")
    for f in files:
        if code_str in pat.findall(f.stem): return f
    # 3. Substring
    hits = [f for f in files if code_str in f.name]
    if len(hits) >= 1: return sorted(hits, key=lambda p: len(p.name))[0]
    return None

def load_dataset(tp_dirs, labels_csv, id_col="code", label_col="group"):
    lab = pd.read_csv(labels_csv)
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
        if not ok: continue
        
        mats.append(np.stack(per_tp, axis=0))
        valid_codes.append(code)
        valid_y.append(yy)

    if not mats: raise RuntimeError("No samples loaded.")
    return np.stack(mats, axis=0), np.array(valid_y), valid_codes

# -----------------------------
# 2. Model & Latent Extraction (with Recon)
# -----------------------------
def load_arch_module(arch_file: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("arch_mod", arch_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def instantiate_model(ModelClass, latent_dim=512):
    try: return ModelClass(in_channels=1, latent_dim=latent_dim, num_classes=2)
    except: 
        try: return ModelClass(latent_dim=latent_dim)
        except: return ModelClass()

def get_consensus_latent_and_recon(X_np, arch_file, model_dir, latent_dim=512, device="cpu"):
    """
    Computes Consensus Latents AND Average Reconstruction (for MSE Filter).
    FIXED: Strictly calculates Z using encoder+attention to avoid getting
    classification logits (size 2) from forward().
    """
    mod = load_arch_module(arch_file)
    # Find model class
    ModelClass = None
    if hasattr(mod, "ConvAutoencoderWithAttention"): ModelClass = getattr(mod, "ConvAutoencoderWithAttention")
    else:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                ModelClass = obj
                break
    
    model_paths = sorted(Path(model_dir).glob("best_model_fold*.pth"))
    
    # Add channel dimension (N, 3, 512, 71) -> (N, 3, 1, 512, 71)
    X = torch.from_numpy(X_np).to(torch.float32).to(device)
    if X.ndim == 4:
        X = X.unsqueeze(2)
    
    Zs = []
    X_recon_accum = None
    
    print(f"[INFO] Running Inference on {len(model_paths)} models...")
    
    for mp in model_paths:
        model = instantiate_model(ModelClass, latent_dim).to(device)
        sd = torch.load(mp, map_location=device)
        if "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        model.eval()
        
        with torch.no_grad():
            # 1. LATENT EXTRACTION (Manual, V12-style)
            # We assume the model has .encoder and optional .attention
            z_stack = []
            for t in range(X.shape[1]):
                xt = X[:, t] # (B, 1, H, W)
                z_stack.append(model.encoder(xt))
            
            z_stack = torch.stack(z_stack, dim=1)
            
            if hasattr(model, 'attention'): 
                z, _ = model.attention(z_stack)
            else: 
                z = torch.mean(z_stack, dim=1)
            
            z = z.cpu().numpy()

            # 2. RECONSTRUCTION (Try forward, fall back to manual)
            try:
                out = model(X)
                if isinstance(out, tuple):
                    recon = out[0] # Assume recon is first element
                else:
                    recon = out
                recon = recon.cpu().numpy()
            except:
                # Fallback manual reconstruction
                x_recon_stack = []
                for t in range(X.shape[1]):
                    # We re-use z_stack from above which holds (B, latent)
                    zt = z_stack[:, t]
                    if hasattr(model, 'decoder') and model.decoder is not None:
                        x_recon_stack.append(model.decoder(zt))
                    else:
                        x_recon_stack.append(torch.zeros_like(X[:, t]))

                recon = torch.stack(x_recon_stack, dim=1).cpu().numpy()

            Zs.append(z)
            if X_recon_accum is None: X_recon_accum = recon
            else: X_recon_accum += recon

    # Consensus Z
    Zref = Zs[0]
    Z_aligned = [Zref]
    for Z in Zs[1:]:
        Z2 = Z.copy()
        for d in range(Z2.shape[1]):
            r = np.corrcoef(Zref[:, d], Z2[:, d])[0, 1]
            if r < 0: Z2[:, d] *= -1.0
        Z_aligned.append(Z2)
    
    Z_agg = np.mean(np.stack(Z_aligned, axis=0), axis=0)
    X_recon_avg = X_recon_accum / len(model_paths)
    
    # Remove channel dim from recon for MSE calc: (N, 3, 1, H, W) -> (N, 3, H, W)
    if X_recon_avg.ndim == 5:
        X_recon_avg = X_recon_avg.squeeze(2)
        
    return Z_agg, X_recon_avg

# -----------------------------
# 3. Main Logic (V12 Exact)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tp_dirs", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--arch_file", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--id_col", default="code")
    ap.add_argument("--label_col", default="group")
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    
    # V12 Specific Params
    ap.add_argument("--roi_percentile", type=float, default=90.0)
    ap.add_argument("--mse_threshold", type=float, default=0.1)
    
    # Feature Selection Params
    ap.add_argument("--feature_source", default="fs_counts", choices=["fs_counts", "oof_matrix"])
    ap.add_argument("--min_fs_count", type=int, default=1)
    ap.add_argument("--top_n_features", type=int, default=0)
    
    ap.add_argument("--out_name", default="v12_exact_biomarker_kit.json")

    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    tp_dirs = parse_csv_list(args.tp_dirs)
    tp_names = [Path(d).name for d in tp_dirs]

    # 1. Identify Features to Export
    print("[INFO] Selecting features...")
    if args.feature_source == "fs_counts":
        dfc = pd.read_csv(results_dir / "feature_selection_counts.csv")
        col_cnt = "count" if "count" in dfc.columns else dfc.columns[1]
        dfc = dfc[dfc[col_cnt] >= args.min_fs_count].sort_values(col_cnt, ascending=False)
        if args.top_n_features > 0: dfc = dfc.head(args.top_n_features)
        features = dfc.iloc[:, 0].astype(str).tolist()
    else:
        dfm = pd.read_csv(results_dir / "proxy_feature_matrix_ridge_oof.csv", nrows=1)
        features = [c for c in dfm.columns if "dim" in c]

    parsed = []
    for fn in features:
        # Expected format: dim123_0h or dim123_out_0h
        try:
            m = re.match(r"dim(\d+)_", fn)
            dim = int(m.group(1))
            found_tp = None
            for idx, name in enumerate(tp_names):
                if name in fn or (name.replace("out_", "") in fn):
                    found_tp = (name, idx)
                    break
            if found_tp: parsed.append((fn, dim, found_tp[0], found_tp[1]))
        except: pass

    print(f"[INFO] Exporting {len(parsed)} features.")

    # 2. Load Data
    print("[INFO] Loading Dataset...")
    X_eem, y_str, codes = load_dataset(tp_dirs, args.labels_csv, args.id_col, args.label_col)
    N = X_eem.shape[0]
    
    # 3. Get Targets & Recon
    device = args.device if torch.cuda.is_available() else "cpu"
    Z_target, X_recon = get_consensus_latent_and_recon(X_eem, args.arch_file, args.model_dir, args.latent_dim, device)
    
    print(f"[INFO] Latent Shape: {Z_target.shape}")
    if Z_target.shape[1] != args.latent_dim:
        raise ValueError(f"Latent dimension mismatch! Expected {args.latent_dim}, got {Z_target.shape[1]}. Check if model outputting class logits instead of latent.")

    # 4. Compute MSE Map (V12 Critical Step)
    print("[INFO] Computing Global MSE Map...")
    if X_recon.shape != X_eem.shape:
        print(f"[WARN] Resizing recon from {X_recon.shape} to {X_eem.shape}")
        mse_map = np.zeros((3, 512, 71))
    else:
        sq_diff = (X_eem - X_recon) ** 2
        mse_map = sq_diff.mean(axis=0) # (3, 512, 71)

    # 5. Fit & Export
    out = {
        "version": "v12_exact_fullfit",
        "created_at": datetime.now().isoformat(),
        "roi_percentile": args.roi_percentile,
        "mse_threshold": args.mse_threshold,
        "features": []
    }
    
    # Prepare flat arrays
    X_flat_tp = [X_eem[:, t].reshape(N, -1) for t in range(3)]
    
    safe_mkdir(results_dir / "fullfit_v12")
    
    for (fname, dim, tp, tp_idx) in parsed:
        print(f"  -> Fitting {fname}...")
        
        # A. MSE Filter
        mse_flat = mse_map[tp_idx].flatten()
        mask_mse = mse_flat < args.mse_threshold
        if mask_mse.sum() < 50:
             mask_mse = mse_flat < np.percentile(mse_flat, 50)
             
        # B. Ridge Probe (Define ROI)
        y_lat = Z_target[:, dim]
        X_pixels = X_flat_tp[tp_idx]
        
        # Standardize for probe
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_pixels)
        
        # Probe only MSE-valid pixels
        X_probe = X_sc[:, mask_mse]
        if X_probe.shape[1] == 0: continue
        
        probe_model = Ridge(alpha=1.0)
        probe_model.fit(X_probe, y_lat)
        
        # Select ROI
        coefs = np.abs(probe_model.coef_)
        thresh = np.percentile(coefs, args.roi_percentile)
        roi_subset_mask = coefs > thresh
        
        # Map back to full image
        full_mask = np.zeros(X_pixels.shape[1], dtype=bool)
        valid_indices = np.where(mask_mse)[0]
        full_mask[valid_indices[roi_subset_mask]] = True
        
        # C. Final Ridge Fit
        X_final = X_pixels[:, full_mask]
        
        # Optimize Alpha (CV)
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        cv_model = RidgeCV(alphas=alphas)
        cv_model.fit(X_final, y_lat)
        
        # Stats
        pred = cv_model.predict(X_final)
        mse_fit = float(np.mean((pred - y_lat)**2))
        r2_fit = float(1.0 - (np.sum((y_lat - pred)**2) / np.sum((y_lat - y_lat.mean())**2)))

        out["features"].append({
            "name": fname,
            "dim": dim,
            "tp_index": tp_idx,
            "roi_mask_indices": np.where(full_mask)[0].tolist(),
            "alpha": float(cv_model.alpha_),
            "coef": cv_model.coef_.tolist(),
            "intercept": float(cv_model.intercept_),
            "stats": {"mse": mse_fit, "r2": r2_fit}
        })

    # Save JSON
    out_path = results_dir / "fullfit_v12" / args.out_name
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    print(f"[DONE] Saved Kit: {out_path}")
    
    # Save Matrix
    feat_names = [f["name"] for f in out["features"]]
    X_proxy = np.zeros((N, len(out["features"])))
    for i, f in enumerate(out["features"]):
        tp = f["tp_index"]
        idx = f["roi_mask_indices"]
        X_roi = X_flat_tp[tp][:, idx]
        X_proxy[:, i] = X_roi @ np.array(f["coef"]) + f["intercept"]
        
    df = pd.DataFrame(X_proxy, columns=feat_names)
    df.insert(0, "code", codes)
    df.insert(1, "group", y_str)
    df.to_csv(results_dir / "fullfit_v12" / "proxy_matrix_v12_fullfit.csv", index=False)
    print(f"[DONE] Saved Matrix: proxy_matrix_v12_fullfit.csv")

if __name__ == "__main__":
    main()