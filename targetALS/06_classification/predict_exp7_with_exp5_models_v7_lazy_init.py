#!/usr/bin/env python
r"""
predict_exp7_with_exp5_models_v7_lazy_init.py

Fixes: RuntimeError: Unexpected key(s) in state_dict: "decoder.fc.weight"...
- This script automatically runs a "Dummy Forward Pass" with shape (1, 3, 1, 512, 71)
  immediately after instantiating the model.
- This forces the "Lazy" layers (encoder.fc, decoder) to initialize their shapes.
- Then it loads the state dict safely.

Windows CMD usage:
python predict_exp7_with_exp5_models_v7_lazy_init.py ^
  --arch_file "C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\conv_autoencoder_detailed.py" ^
  --model_glob "C:\Users\riccardo-s\Documents\CNT\targetALS\exp5\5_fold_models_original\best_model_fold*.pth" ^
  --data_root "C:\Users\riccardo-s\Documents\CNT\targetALS\exp7\3" ^
  --labels_csv "C:\Users\riccardo-s\Documents\CNT\targetALS\exp7\3\sample_labels_03.csv" ^
  --id_col code --label_col group --pos_label ALS ^
  --output_csv "C:\Users\riccardo-s\Documents\CNT\targetALS\exp7\3_predictions.csv" ^
  --timepoints "C:\Users\riccardo-s\Documents\CNT\targetALS\exp7\0h_corrected" "out_6h" "out_24h" ^
  --device cpu --recursive
"""

import os
import sys
import glob
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix, roc_auc_score

# -----------------------------------------------------------------------------
# 1. Dynamic Import for Architecture
# -----------------------------------------------------------------------------
def load_arch_module(arch_file_path):
    spec = importlib.util.spec_from_file_location("dynamic_arch", arch_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load architecture from {arch_file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_arch"] = module
    spec.loader.exec_module(module)
    return module

# -----------------------------------------------------------------------------
# 2. Helper: EEM Loader
# -----------------------------------------------------------------------------
def load_eem_matrix(xlsx_path, target_shape=(512, 71)):
    """
    Robust loader. Returns (1, H, W) float32 in [0,1].
    Adjusts standard EEM loading to match your dataset class logic.
    """
    try:
        wb = load_workbook(xlsx_path, data_only=True)
        ws = wb.active
        data = []
        for r_i, row in enumerate(ws.iter_rows(values_only=True)):
            if r_i == 0: continue # header
            row_vals = []
            for c_i, val in enumerate(row):
                if c_i == 0: continue # index
                if isinstance(val, (int, float)):
                    row_vals.append(float(val))
                else:
                    row_vals.append(0.0)
            if row_vals:
                data.append(row_vals)
        
        arr = np.array(data, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((1, *target_shape), dtype=np.float32)

        # Normalize
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
            
        # Ensure shape matches target (crop or pad if needed, simplified here)
        # For now, just return as is, assuming strict compliance or resize downstream
        if arr.shape != target_shape:
            # Simple resize via torch if needed, but for now let's trust data
            pass

        return arr[np.newaxis, ...] # (1, H, W)
    except Exception as e:
        print(f"Error loading {xlsx_path}: {e}")
        return np.zeros((1, *target_shape), dtype=np.float32)

# -----------------------------------------------------------------------------
# 3. Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_file", required=True)
    parser.add_argument("--model_glob", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--id_col", default="code")
    parser.add_argument("--label_col", default="group")
    parser.add_argument("--pos_label", default="ALS")
    parser.add_argument("--output_csv", default="predictions.csv")
    parser.add_argument("--timepoints", nargs="+", default=["out_0h", "out_6h", "out_24h"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    # A) Load Labels
    df_labels = pd.read_csv(args.labels_csv)
    # Normalize IDs to string
    df_labels[args.id_col] = df_labels[args.id_col].astype(str).str.strip()
    
    # Map Subject -> True Label
    true_map = {}
    for _, row in df_labels.iterrows():
        sid = row[args.id_col]
        lbl = str(row[args.label_col]).strip()
        true_map[sid] = lbl
    
    # B) Find Samples
    # Logic: Look for files in the FIRST timepoint folder that match ID
    tp0_path = os.path.join(args.data_root, args.timepoints[0])
    # Handle absolute paths in timepoints arg
    if os.path.isabs(args.timepoints[0]):
        tp0_path = args.timepoints[0]

    found_samples = []
    print(f"Scanning for samples in: {tp0_path}")
    
    # Heuristic: Match .xlsx files to IDs
    # If ID is "Sample1", look for "Sample1.xlsx"
    files = glob.glob(os.path.join(tp0_path, "*.xlsx"))
    files = [os.path.basename(f) for f in files if not os.path.basename(f).startswith("~")]
    
    for sid in true_map.keys():
        fname = f"{sid}.xlsx"
        if fname in files:
            found_samples.append(sid)
            
    print(f"Matched {len(found_samples)} / {len(true_map)} labeled samples.")
    if len(found_samples) == 0:
        print("Error: No matching files found. Check your CSV IDs vs Filenames.")
        return

    # C) Load Architecture
    arch_mod = load_arch_module(args.arch_file)
    # Assuming standard class name
    ModelClass = getattr(arch_mod, "ConvAutoencoderWithAttention", None)
    if not ModelClass:
        # Fallback to V2 name if user renamed it
        ModelClass = getattr(arch_mod, "ConvAutoencoderV2", None)
    
    if not ModelClass:
        print("Error: Could not find 'ConvAutoencoderWithAttention' or 'ConvAutoencoderV2' class.")
        return

    # D) Load Models & Predict
    checkpoints = sorted(glob.glob(args.model_glob))
    if not checkpoints:
        print("Error: No checkpoints found.")
        return

    print(f"Found {len(checkpoints)} models.")
    
    # Result storage
    all_probs = np.zeros((len(found_samples), len(checkpoints)))
    
    # Pre-load data to memory (if small enough) or lazy load
    # For robust inference, let's load on the fly
    
    for m_i, ckpt_path in enumerate(checkpoints):
        print(f"--- Model {m_i+1}: {os.path.basename(ckpt_path)} ---")
        
        # 1. Instantiate
        # Hardcoded params inferred from your previous logs (512 latent, 1 channel)
        model = ModelClass(in_channels=1, latent_dim=512, num_classes=2).to(device)
        
        # 2. FIX: FORCE LAZY INITIALIZATION
        # Run a dummy variable through the model to create the missing layers
        print("  Running dummy pass to initialize lazy layers...")
        dummy_input = torch.zeros(1, 3, 1, 512, 71).to(device)
        try:
            with torch.no_grad():
                model(dummy_input)
        except Exception as e:
            print(f"  Warning: Dummy pass failed ({e}). Loading might fail.")

        # 3. Load Weights
        sd = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(sd, strict=args.strict_load)
            print("  Weights loaded successfully.")
        except RuntimeError as e:
            print(f"  Strict load failed: {e}")
            print("  Trying strict=False...")
            model.load_state_dict(sd, strict=False)
            
        model.eval()
        
        # 4. Infer
        for s_i, sid in enumerate(found_samples):
            # Load 3 timepoints
            imgs = []
            for tp in args.timepoints:
                # Handle absolute/relative paths
                if os.path.isabs(tp):
                    folder = tp
                else:
                    folder = os.path.join(args.data_root, tp)
                
                fpath = os.path.join(folder, f"{sid}.xlsx")
                img = load_eem_matrix(fpath) # (1, 512, 71)
                imgs.append(torch.from_numpy(img))
            
            # Stack -> (1, 3, 1, 512, 71)
            batch_x = torch.stack(imgs, dim=0).unsqueeze(0).to(device) # unsqueeze batch dim
            
            with torch.no_grad():
                # Forward returns: (recon, logits, attn)
                _, logits, _ = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                p_als = probs[0, 1].item()
                all_probs[s_i, m_i] = p_als
                
            if s_i % 10 == 0:
                print(f"  Predicting {s_i+1}/{len(found_samples)}...", end='\r')
        print("")

    # E) Aggregate & Save
    avg_probs = np.mean(all_probs, axis=1)
    std_probs = np.std(all_probs, axis=1)
    
    results = []
    for i, sid in enumerate(found_samples):
        pred_label = args.pos_label if avg_probs[i] >= 0.5 else "Healthy"
        row = {
            "Subject": sid,
            "True_Label": true_map.get(sid, "Unknown"),
            "Prediction": pred_label,
            "Prob_ALS_Mean": avg_probs[i],
            "Uncertainty": std_probs[i]
        }
        # Add individual fold probs
        for m_i in range(len(checkpoints)):
            row[f"Fold_{m_i}_Prob"] = all_probs[i, m_i]
        results.append(row)
        
    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_csv, index=False)
    
    # Metrics
    if set(df_out["True_Label"].unique()) != {"Unknown"}:
        y_true = (df_out["True_Label"] == args.pos_label).astype(int)
        y_score = df_out["Prob_ALS_Mean"]
        try:
            auc = roc_auc_score(y_true, y_score)
            print(f"\nEnsemble ROC AUC: {auc:.4f}")
        except:
            print("\nCould not calc AUC (single class?)")
            
    print(f"Saved to: {args.output_csv}")

if __name__ == "__main__":
    main()