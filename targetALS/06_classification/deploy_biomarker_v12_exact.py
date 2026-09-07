import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import glob
import importlib.util
from openpyxl import load_workbook
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_PERCENTILE = 90.0   # Top 10% (V12 Setting)
MSE_THRESHOLD = 0.1     # Max allowed reconstruction error (V12 Setting)
TOP_K_FEATURES = 10     # Number of biomarkers to keep in the kit

# --- UTILS ---
def load_arch_module(arch_file):
    spec = importlib.util.spec_from_file_location("arch", arch_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_excel_as_array(filepath, shape=(512, 71)):
    """Robust loader with Min-Max Normalization (V12 Logic)"""
    try:
        wb = load_workbook(filepath, data_only=True)
        data = [[float(x) if x is not None else 0.0 for x in r] for r in wb.active.iter_rows(min_row=2, min_col=2, values_only=True)]
        arr = np.array(data, dtype=np.float32)
        if arr.size == 0: return np.zeros(shape, dtype=np.float32)
        
        # Min-Max Normalize per file
        mn, mx = arr.min(), arr.max()
        if mx > mn: arr = (arr - mn) / (mx - mn)
        
        # Pad/Crop
        if arr.shape != shape:
            padded = np.zeros(shape, dtype=np.float32)
            h, w = min(shape[0], arr.shape[0]), min(shape[1], arr.shape[1])
            padded[:h, :w] = arr[:h, :w]
            return padded
        return arr
    except: return np.zeros(shape, dtype=np.float32)

def get_consensus_latent_and_recon(models, loader):
    """
    Extracts Latents AND Reconstructions (Needed for MSE Filter).
    Robustly handles different model architectures (Manual loop vs Forward).
    """
    print("  -> Running Inference (Latents + Reconstruction)...")
    Z_list = []
    X_recon_accum = None
    count = 0
    
    for model_idx, model in enumerate(models):
        model.eval()
        z_curr = []
        
        with torch.no_grad():
            batch_recon_list = []
            for X, _ in loader:
                X = X.to(DEVICE) # (B, 3, 512, 71)
                
                # --- STRATEGY 1: Try Standard Forward Pass (Easiest) ---
                # Most models return (recon, latent) or just recon
                try:
                    output = model(X)
                    
                    # Case A: Output is tuple (recon, latent, ...)
                    if isinstance(output, tuple):
                        recon_batch = output[0]
                        latent_batch = output[1] if len(output) > 1 else None
                    else:
                        # Case B: Output is just recon
                        recon_batch = output
                        latent_batch = None # We will fetch latent manually if needed
                    
                    # Check if recon shape matches input
                    if recon_batch.shape == X.shape:
                        batch_recon_list.append(recon_batch.cpu().numpy())
                    else:
                        raise ValueError("Shape mismatch in forward pass")

                    # If latent wasn't returned, extract it manually
                    if latent_batch is None:
                        # Manual Encoder extraction
                        if hasattr(model, 'encoder'):
                            z_stack = []
                            for t in range(X.shape[1]):
                                xt = X[:, t].unsqueeze(1)
                                z_stack.append(model.encoder(xt))
                            z_stack = torch.stack(z_stack, dim=1)
                            if hasattr(model, "attention"): 
                                z_agg, _ = model.attention(z_stack)
                            else: 
                                z_agg = torch.mean(z_stack, dim=1)
                            z_curr.append(z_agg.cpu().numpy())
                        else:
                            # Cannot find latent
                            print("Warning: Could not extract latent vector.")
                            z_curr.append(np.zeros((X.shape[0], 512)))
                    else:
                        # Use the returned latent
                        # If latent is (B, 512), perfect.
                        if latent_batch.dim() == 2:
                            z_curr.append(latent_batch.cpu().numpy())
                        else:
                             z_curr.append(latent_batch.mean(dim=1).cpu().numpy())
                             
                except Exception as e:
                    # --- STRATEGY 2: Manual Loop (Fallback) ---
                    # If forward() fails (e.g. strict signature), do manual components
                    # print(f"Fallback to manual loop due to: {e}")
                    
                    if hasattr(model, 'encoder'):
                        z_stack = []
                        x_recon_stack = []
                        
                        for t in range(X.shape[1]):
                            xt = X[:, t].unsqueeze(1) # (B, 1, H, W)
                            zt = model.encoder(xt)
                            z_stack.append(zt)
                            
                            # Check for decoder
                            if hasattr(model, 'decoder') and model.decoder is not None:
                                try:
                                    x_r = model.decoder(zt)
                                    x_recon_stack.append(x_r.squeeze(1))
                                except: pass
                            elif hasattr(model, 'dec') and model.dec is not None:
                                try:
                                    x_r = model.dec(zt)
                                    x_recon_stack.append(x_r.squeeze(1))
                                except: pass

                        # Agg Latent
                        z_stack_t = torch.stack(z_stack, dim=1)
                        if hasattr(model, "attention"): 
                            z_agg, _ = model.attention(z_stack_t)
                        else: 
                            z_agg = torch.mean(z_stack_t, dim=1)
                        z_curr.append(z_agg.cpu().numpy())
                        
                        # Agg Recon
                        if len(x_recon_stack) == 3:
                            x_recon_batch = torch.stack(x_recon_stack, dim=1)
                            batch_recon_list.append(x_recon_batch.cpu().numpy())
                        else:
                            # If no decoder found, append Zeros (Disables MSE filter effectively)
                            batch_recon_list.append(np.zeros_like(X.cpu().numpy()))
                    else:
                         print("CRITICAL: Model has no 'encoder' and forward() failed.")
                         return None, None

            # Aggregate Recon for this model
            if batch_recon_list:
                full_recon = np.concatenate(batch_recon_list, axis=0)
                if X_recon_accum is None: X_recon_accum = full_recon
                else: X_recon_accum += full_recon
                count += 1
                
        Z_list.append(np.concatenate(z_curr, axis=0))
    
    # Average Recon across models
    if count > 0:
        X_recon_avg = X_recon_accum / count
    else:
        # Emergency fallback: Zeros
        print("Warning: Could not compute reconstructions. MSE Filter will be disabled.")
        X_recon_avg = np.zeros_like(X_recon_accum)

    # Consensus Latents
    Z_ref = Z_list[0]
    Z_aligned = [Z_ref]
    for i in range(1, len(Z_list)):
        z_curr = Z_list[i].copy()
        for d in range(z_curr.shape[1]):
            if np.corrcoef(Z_ref[:, d], z_curr[:, d])[0, 1] < 0:
                z_curr[:, d] = -z_curr[:, d]
        Z_aligned.append(z_curr)
    
    return np.mean(Z_aligned, axis=0), X_recon_avg

# --- CORE LOGIC ---

def train_master(args):
    print("\n=== TRAINING MASTER BIOMARKER (V12 EXACT LOGIC) ===")
    
    # 1. Load Data
    print("Loading Training Data...")
    df = pd.read_csv(args.labels_csv)
    codes = df['code'].astype(str).str.strip().tolist()
    y = np.array([1 if str(x).strip() == "ALS" else 0 for x in df['group']])
    
    tp_paths = args.tp_dirs.split(',')
    X_img = []
    for code in codes:
        tps = [load_excel_as_array(os.path.join(p, f"{code}.xlsx")) for p in tp_paths]
        X_img.append(np.array(tps))
    X_img = np.array(X_img) # (N, 3, 512, 71)
    
    # 2. Get Targets AND Reconstructions (for MSE Filter)
    arch = load_arch_module(args.arch_file)
    ModelClass = getattr(arch, "ConvAutoencoderWithAttention", None) or getattr(arch, "ConvAutoencoder", None)
    
    models = []
    for mp in glob.glob(os.path.join(args.models_dir, "*.pth")):
        try: m = ModelClass(in_channels=1, latent_dim=512, num_classes=2).to(DEVICE)
        except: m = ModelClass(latent_dim=512, num_classes=2).to(DEVICE)
        try: m.load_state_dict(torch.load(mp, map_location=DEVICE))
        except: m.load_state_dict(torch.load(mp, map_location=DEVICE), strict=False)
        models.append(m)
        
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_img), torch.zeros(len(y))), batch_size=4)
    Z_target, X_recon = get_consensus_latent_and_recon(models, loader)
    
    # 3. Calculate Global MSE Map
    print("Calculating Global MSE Map...")
    if X_recon is not None:
        sq_diff = (X_img - X_recon) ** 2
        mse_map = sq_diff.mean(axis=0) # (3, 512, 71)
    else:
        mse_map = np.zeros((3, 512, 71)) # Disable filter if no recon
    
    # 4. Train Proxies (V12 Ridge Probe + MSE Filter)
    print("Training Ridge Proxies (Global ROI)...")
    proxies = {} 
    X_proxy_matrix = []
    feature_names = []
    tp_names = ["0h", "6h", "24h"]
    
    for d in range(512):
        if d % 50 == 0: print(f"  ... Dim {d}/512")
        target_vec = Z_target[:, d]
        
        for t_idx, tp_name in enumerate(tp_names):
            pixel_matrix = X_img[:, t_idx, :, :].reshape(len(y), -1)
            
            # MSE Filter
            mse_flat = mse_map[t_idx].flatten()
            mask_mse = mse_flat < MSE_THRESHOLD
            if mask_mse.sum() < 50:
                fallback_thresh = np.percentile(mse_flat, 50)
                mask_mse = mse_flat < fallback_thresh
            
            # Ridge Probe
            scaler_pix = StandardScaler()
            X_sc = scaler_pix.fit_transform(pixel_matrix)
            X_probe_in = X_sc[:, mask_mse]
            if X_probe_in.shape[1] == 0: continue
            
            probe = RidgeCV(alphas=[0.1, 1.0, 10.0])
            probe.fit(X_probe_in, target_vec)
            
            coefs = np.abs(probe.coef_)
            thresh = np.percentile(coefs, ROI_PERCENTILE)
            mask_roi_subset = coefs > thresh
            
            full_mask = np.zeros(pixel_matrix.shape[1], dtype=bool)
            valid_indices = np.where(mask_mse)[0]
            keep_indices = valid_indices[mask_roi_subset]
            full_mask[keep_indices] = True
            
            if full_mask.sum() == 0: continue
            
            # Final Proxy
            X_masked = pixel_matrix[:, full_mask]
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_masked, target_vec)
            
            feat_name = f"dim{d}_{tp_name}"
            proxies[feat_name] = {'mask': full_mask, 'model': ridge, 'tp_idx': t_idx}
            
            X_proxy_matrix.append(ridge.predict(X_masked))
            feature_names.append(feat_name)
            
    X_proxy_matrix = np.array(X_proxy_matrix).T
    
    # 5. Train Classifier
    print("Selecting Best Features & Training Classifier...")
    clf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=TOP_K_FEATURES)),
        ('clf', LogisticRegression(C=1.0, solver='liblinear'))
    ])
    
    clf_pipe.fit(X_proxy_matrix, y)
    
    mask_selected = clf_pipe.named_steps['selector'].get_support()
    selected_feats = np.array(feature_names)[mask_selected]
    print(f"  -> Top {TOP_K_FEATURES} Features: {selected_feats}")
    
    # 6. Save Kit
    master_dict = {
        'proxies': {k: v for k, v in proxies.items() if k in selected_feats},
        'classifier_pipe': clf_pipe,
        'feature_order': feature_names
    }
    
    joblib.dump(master_dict, args.save_path)
    print(f"\nSUCCESS. Master Biomarker (V12 Exact) saved to: {args.save_path}")

def predict_batch(args):
    print(f"\n=== VALIDATING ON NEW BATCH: {args.batch_dir} ===")
    
    if not os.path.exists(args.load_path):
        print(f"Error: Biomarker file {args.load_path} not found.")
        return
    
    kit = joblib.load(args.load_path)
    proxies = kit['proxies']
    clf_pipe = kit['classifier_pipe']
    expected_features = kit['feature_order']
    
    y_true_new = None
    if args.new_labels_csv:
        print(f"Loading New Labels from: {args.new_labels_csv}")
        df_new = pd.read_csv(args.new_labels_csv)
        label_map = {str(r['code']).strip(): (1 if str(r['group']).strip() == "ALS" else 0) for _, r in df_new.iterrows()}
    
    tp_dirs = args.batch_dir.split(',')
    if len(tp_dirs) != 3:
        print("Error: Provide 3 folders separated by comma (0h,6h,24h)")
        return

    codes = [os.path.basename(f).replace('.xlsx','') for f in glob.glob(os.path.join(tp_dirs[0], "*.xlsx"))]
    print(f"Found {len(codes)} samples.")
    
    print("Generating Proxies...")
    X_matrix = np.zeros((len(codes), len(expected_features)))
    valid_indices = []
    y_true_ordered = []
    
    for i, code in enumerate(codes):
        if args.new_labels_csv:
            if code not in label_map:
                print(f"Warning: Sample {code} not in labels CSV.")
                continue
            y_true_ordered.append(label_map[code])
            
        valid_indices.append(i)
        
        imgs = [load_excel_as_array(os.path.join(p, f"{code}.xlsx")) for p in tp_dirs]
        imgs = np.array(imgs)
        
        for feat_idx, feat_name in enumerate(expected_features):
            if feat_name in proxies:
                p_data = proxies[feat_name]
                mask = p_data['mask']
                ridge = p_data['model']
                tp_idx = p_data['tp_idx']
                
                pixels = imgs[tp_idx].flatten().reshape(1, -1)
                val = ridge.predict(pixels[:, mask])[0]
                X_matrix[i, feat_idx] = val
            else:
                X_matrix[i, feat_idx] = 0.0

    X_matrix = X_matrix[valid_indices]
    final_codes = [codes[i] for i in valid_indices]

    print("Classifying...")
    probs = clf_pipe.predict_proba(X_matrix)[:, 1]
    preds = clf_pipe.predict(X_matrix)
    
    print("\nRESULTS:")
    print(f"{'Sample':<20} | {'Prob(ALS)':<10} | {'Prediction'}")
    print("-" * 45)
    results = []
    for i, code in enumerate(final_codes):
        lbl = "ALS" if preds[i]==1 else "Control"
        print(f"{code:<20} | {probs[i]:.4f}     | {lbl}")
        res_row = {'sample': code, 'prob_als': probs[i], 'prediction': lbl}
        if args.new_labels_csv:
            res_row['true_label'] = "ALS" if y_true_ordered[i]==1 else "Control"
            res_row['correct'] = (preds[i] == y_true_ordered[i])
        results.append(res_row)
        
    pd.DataFrame(results).to_csv("batch_predictions_v12.csv", index=False)
    
    if args.new_labels_csv and len(y_true_ordered) > 0:
        y_true = np.array(y_true_ordered)
        acc = accuracy_score(y_true, preds)
        try: auc = roc_auc_score(y_true, probs)
        except: auc = 0.0
        
        print("\n" + "="*40)
        print("EXTERNAL VALIDATION METRICS")
        print("="*40)
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC:      {auc:.4f}")
        print("-" * 40)
        print(confusion_matrix(y_true, preds))
        print(classification_report(y_true, preds, target_names=["Control", "ALS"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'predict'])
    parser.add_argument('--labels_csv', help="Path to original labels")
    parser.add_argument('--tp_dirs', help="Comma-sep paths to original 0h,6h,24h")
    parser.add_argument('--models_dir', help="Path to DL models")
    parser.add_argument('--arch_file', help="Path to arch file")
    parser.add_argument('--save_path', default="als_biomarker_v12.pkl", help="Save location")
    parser.add_argument('--batch_dir', help="Comma-sep paths to NEW 0h,6h,24h folders")
    parser.add_argument('--load_path', default="als_biomarker_v12.pkl", help="Path to frozen model")
    parser.add_argument('--new_labels_csv', help="Path to NEW labels csv")
    
    args = parser.parse_args()
    if args.mode == 'train':
        train_master(args)
    else:
        predict_batch(args)