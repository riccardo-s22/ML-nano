#!/usr/bin/env python3
"""
Proxy Latent Inference Engine (Strategy E)
==========================================

Lightweight inference script for computing proxy latent features on NEW
EEM samples using pre-exported encoder weights. No training, no fitting,
no scikit-learn dependency for core inference.

WHAT THIS DOES:
  For each new sample (3 EEM images at 0h, 6h, 24h):
    1. Normalize each image to [0, 1]
    2. Run through fold-specific conv layers -> conv features [3, F]
    3. Apply FC layer -> z_per_tp [3, 512]
    4. Apply attention network -> z_agg [512,]
    5. Extract selected dims, sign-flip and z-score normalize
    6. Output proxy latent vector ready for classification

REQUIRED INPUTS:
  - inference_bundle.pt  (exported by step2, contains all model weights)
  - 3 EEM Excel files (0h, 6h, 24h) for the new sample

EXPORT:
  The bundle is created by step2 with:
    python step2_encoder_weight_proxy_v3.py ... --export_inference

  Or standalone:
    python proxy_latent_inference.py --export \
        --arch_file ... --model_dir ... --opt_config ... \
        --step2_dir ./step2_encoder_proxy_v3_E \
        --output_bundle ./inference_bundle.pt

INFERENCE:
  Single sample:
    python proxy_latent_inference.py --predict \
        --bundle ./inference_bundle.pt \
        --eem_files sample_0h.xlsx,sample_6h.xlsx,sample_24h.xlsx

  Batch:
    python proxy_latent_inference.py --predict_batch \
        --bundle ./inference_bundle.pt \
        --tp_dirs ./new_data_0h,./new_data_6h,./new_data_24h \
        --output_csv ./new_sample_predictions.csv

  Classify (requires scikit-learn):
    python proxy_latent_inference.py --predict_and_classify \
        --bundle ./inference_bundle.pt \
        --eem_files sample_0h.xlsx,sample_6h.xlsx,sample_24h.xlsx
"""

import argparse
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TP_NAMES = ["0h", "6h", "24h"]


# =============================================================================
# EEM loading (self-contained, no external dependencies)
# =============================================================================

def load_eem_excel(filepath: str) -> np.ndarray:
    """Load an EEM Excel file and normalize to [0, 1]."""
    from openpyxl import load_workbook
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    data = []
    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
        if row_idx == 0:
            continue  # skip header
        row_data = []
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue  # skip row label
            if isinstance(cell, (int, float)) and cell is not None:
                row_data.append(float(cell))
            else:
                row_data.append(0.0)
        if row_data:
            data.append(row_data)
    arr = np.array(data, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def load_sample_3tp(files_0h_6h_24h: List[str]) -> np.ndarray:
    """Load 3 timepoint EEM files for one sample. Returns [3, H, W]."""
    images = []
    for fpath in files_0h_6h_24h:
        images.append(load_eem_excel(fpath))
    # Ensure consistent shape
    shapes = [a.shape for a in images]
    if len(set(shapes)) > 1:
        min_h = min(s[0] for s in shapes)
        min_w = min(s[1] for s in shapes)
        images = [a[:min_h, :min_w] for a in images]
    return np.stack(images, axis=0)  # [3, H, W]


# =============================================================================
# Analytical forward pass (numpy only, no PyTorch needed for inference)
# =============================================================================

def conv_forward_numpy(x: np.ndarray, conv_params: Dict) -> np.ndarray:
    """
    Apply conv layers using PyTorch (needed for conv operations).
    x: [1, 1, H, W] tensor
    Returns: flattened conv4 output [F,]
    """
    # We need PyTorch for conv operations - but only for forward pass
    x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    for layer_name in ["conv1", "conv2", "conv3", "conv4"]:
        layer_p = conv_params[layer_name]
        # Each conv block is: Conv2d -> BatchNorm -> ReLU -> MaxPool
        # We reconstruct using the stored state dict
        for sub_key, sub_params in layer_p.items():
            if sub_key.startswith("conv") or sub_key == "0":
                # Conv2d
                weight = torch.tensor(sub_params["weight"], dtype=torch.float32).to(DEVICE)
                bias = torch.tensor(sub_params["bias"], dtype=torch.float32).to(DEVICE) if "bias" in sub_params else None
                x_t = torch.nn.functional.conv2d(x_t, weight, bias, padding=1)
            elif sub_key.startswith("bn") or sub_key == "1":
                # BatchNorm2d
                weight = torch.tensor(sub_params["weight"], dtype=torch.float32).to(DEVICE)
                bias = torch.tensor(sub_params["bias"], dtype=torch.float32).to(DEVICE)
                mean = torch.tensor(sub_params["running_mean"], dtype=torch.float32).to(DEVICE)
                var = torch.tensor(sub_params["running_var"], dtype=torch.float32).to(DEVICE)
                x_t = torch.nn.functional.batch_norm(x_t, mean, var, weight, bias, training=False)
            elif sub_key.startswith("relu") or sub_key == "2":
                x_t = torch.nn.functional.relu(x_t)
            elif sub_key.startswith("pool") or sub_key == "3":
                x_t = torch.nn.functional.max_pool2d(x_t, kernel_size=2, stride=2)

    return x_t.view(-1).cpu().numpy()


def compute_z_agg_from_conv_feats(
    conv_feats_3tp: np.ndarray,  # [3, F]
    fc_W: np.ndarray,            # [latent_dim, F]
    fc_b: np.ndarray,            # [latent_dim]
    attn_params: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute z_agg for a single sample from conv features.
    Returns: (z_agg [latent_dim,], attn_weights [3,])
    """
    T = 3
    latent_dim = fc_W.shape[0]

    # Per-timepoint latent vectors
    z_per_tp = np.zeros((T, latent_dim))
    for t in range(T):
        z_per_tp[t, :] = conv_feats_3tp[t, :] @ fc_W.T + fc_b

    # Attention
    w1 = attn_params["attn_w1"]
    b1 = attn_params["attn_b1"]
    w2 = attn_params["attn_w2"]
    b2 = attn_params["attn_b2"]

    scores = np.zeros(T)
    for t in range(T):
        h = np.tanh(z_per_tp[t, :] @ w1.T + b1)
        scores[t] = float(h @ w2.T + b2)

    # Softmax
    scores_exp = np.exp(scores - scores.max())
    attn_weights = scores_exp / scores_exp.sum()

    # Weighted sum
    z_agg = sum(attn_weights[t] * z_per_tp[t, :] for t in range(T))
    return z_agg, attn_weights


# =============================================================================
# Inference engine
# =============================================================================

class ProxyLatentInference:
    """
    Lightweight inference engine for computing proxy latent features.

    Usage:
        engine = ProxyLatentInference.from_bundle("inference_bundle.pt")
        proxy_vec = engine.predict_single(["sample_0h.xlsx", "sample_6h.xlsx", "sample_24h.xlsx"])
        # proxy_vec is a dict: {"best_model_fold3_dim477": 0.123, ...}
    """

    def __init__(self, bundle: Dict):
        self.features = bundle["features"]         # list of feature configs
        self.fold_models = bundle["fold_models"]    # fold_idx -> model params
        self.normalization = bundle["normalization"]  # per-feature sign/scale info
        self.classifier_params = bundle.get("classifier_params", None)
        self.metadata = bundle.get("metadata", {})

    @classmethod
    def from_bundle(cls, bundle_path: str) -> "ProxyLatentInference":
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        return cls(bundle)

    def _extract_conv_features_single(
        self, sample_3tp: np.ndarray, fold_idx: int
    ) -> np.ndarray:
        """
        Run a single sample [3, H, W] through fold's conv layers.
        Returns: [3, F] conv features.

        Uses the full PyTorch model for reliable conv forward pass.
        """
        model_data = self.fold_models[fold_idx]

        # Reconstruct model and load weights
        arch_mod = model_data["arch_module_code"]
        state_dict = model_data["state_dict"]

        # We need to reconstruct the model - use stored architecture
        # Create a temporary module from stored code
        import types
        mod = types.ModuleType("arch_temp")
        exec(arch_mod, mod.__dict__)

        latent_dim = self.metadata.get("latent_dim", 512)
        model = mod.ConvAutoencoderWithAttention(
            in_channels=1, latent_dim=latent_dim, num_classes=2
        )
        model.to(DEVICE)
        model.eval()

        # Dummy forward to initialize dynamic layers
        H, W = sample_3tp.shape[1], sample_3tp.shape[2]
        dummy = torch.zeros(1, 3, 1, H, W, device=DEVICE)
        with torch.no_grad():
            _ = model(dummy)

        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Extract conv features
        X_t = torch.tensor(sample_3tp, dtype=torch.float32)  # [3, H, W]
        conv_feats = np.zeros((3, model_data["flatten_size"]))
        with torch.no_grad():
            for t in range(3):
                x = X_t[t:t+1].unsqueeze(0).to(DEVICE)  # [1, 1, H, W]
                x = model.encoder.conv1(x)
                x = model.encoder.conv2(x)
                x = model.encoder.conv3(x)
                x = model.encoder.conv4(x)
                conv_feats[t, :] = x.view(-1).cpu().numpy()

        del model
        return conv_feats

    def predict_single(self, eem_files: List[str]) -> Dict[str, float]:
        """
        Predict proxy latent values for a single new sample.

        Args:
            eem_files: [path_0h, path_6h, path_24h] - three EEM Excel files

        Returns:
            dict mapping feature names to proxy latent values
        """
        assert len(eem_files) == 3, "Need exactly 3 EEM files (0h, 6h, 24h)"
        sample = load_sample_3tp(eem_files)  # [3, H, W]

        results = {}
        # Cache conv features per fold (avoid recomputation if multiple features use same fold)
        conv_cache = {}

        for feat_cfg in self.features:
            fi = feat_cfg["fold_idx"]
            dim = feat_cfg["dim"]
            fname = feat_cfg["feature"]

            # Get conv features for this fold
            if fi not in conv_cache:
                conv_cache[fi] = self._extract_conv_features_single(sample, fi)
            conv_feats = conv_cache[fi]  # [3, F]

            # Get FC and attention params
            fc_W = self.fold_models[fi]["fc_W"]
            fc_b = self.fold_models[fi]["fc_b"]
            attn_p = self.fold_models[fi]["attn_params"]

            # Compute z_agg
            z_agg, attn_w = compute_z_agg_from_conv_feats(conv_feats, fc_W, fc_b, attn_p)

            # Extract dim d
            z_raw = z_agg[dim]

            # Apply sign flip
            norm = self.normalization[fname]
            if norm["sign_flip"]:
                z_raw = -z_raw

            # Z-score normalize using training statistics
            z_normalized = (z_raw - norm["train_mean"]) / max(norm["train_std"], 1e-12)
            # Scale to target distribution
            z_proxy = z_normalized * norm["target_std"] + norm["target_mean"]

            results[fname] = float(z_proxy)

        return results

    def predict_batch(
        self, codes: List[str], tp_dirs: List[str]
    ) -> pd.DataFrame:
        """
        Predict proxy latent values for multiple samples.

        Args:
            codes: sample code list
            tp_dirs: [dir_0h, dir_6h, dir_24h]

        Returns:
            DataFrame with columns: code, feature1, feature2, ...
        """
        rows = []
        for code in codes:
            # Find files
            files = []
            for tp_dir in tp_dirs:
                fpath = self._find_file(tp_dir, code)
                if fpath is None:
                    raise FileNotFoundError(f"No file for '{code}' in {tp_dir}")
                files.append(fpath)

            proxy_vals = self.predict_single(files)
            row = {"code": code}
            row.update(proxy_vals)
            rows.append(row)

        return pd.DataFrame(rows)

    def predict_and_classify(self, eem_files: List[str]) -> Dict:
        """
        Predict proxy latent values AND run classification.
        Requires classifier_params in the bundle.
        """
        proxy_vals = self.predict_single(eem_files)

        if self.classifier_params is None:
            return {"proxy_values": proxy_vals, "classification": None,
                    "warning": "No classifier in bundle. Use --export_classifier."}

        # Build feature vector in correct order
        feature_names = [f["feature"] for f in self.features]
        X = np.array([[proxy_vals[fn] for fn in feature_names]])

        # Run classifiers
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        predictions = {}
        for clf_name, clf_data in self.classifier_params.items():
            # Reconstruct classifier from stored params
            clf = clf_data["classifier"]
            scaler = clf_data.get("scaler", None)

            if scaler is not None:
                X_s = scaler.transform(X)
            else:
                X_s = X

            pred = clf.predict(X_s)[0]
            prob = clf.predict_proba(X_s)[0] if hasattr(clf, "predict_proba") else None

            predictions[clf_name] = {
                "prediction": "ALS" if pred == 1 else "CTRL",
                "probability_ALS": float(prob[1]) if prob is not None else None,
            }

        return {"proxy_values": proxy_vals, "classification": predictions}

    def predict_classify_batch(
        self, codes: List[str], tp_dirs: List[str],
        labels: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Predict proxy latent values + classify for multiple samples.

        Args:
            codes: sample code list
            tp_dirs: [dir_0h, dir_6h, dir_24h]
            labels: optional dict {code: "ALS"/"CTRL"} for accuracy reporting

        Returns:
            DataFrame with proxy values, predictions, and probabilities per sample
        """
        feature_names = [f["feature"] for f in self.features]
        has_clf = self.classifier_params is not None
        clf_names = list(self.classifier_params.keys()) if has_clf else []

        rows = []
        for i, code in enumerate(codes):
            # Find files
            files = []
            for tp_dir in tp_dirs:
                fpath = self._find_file(tp_dir, code)
                if fpath is None:
                    print(f"  [SKIP] No file for '{code}' in {tp_dir}")
                    break
                files.append(fpath)
            if len(files) != 3:
                continue

            result = self.predict_and_classify(files)
            row = {"code": code}

            # True label if available
            if labels and code in labels:
                row["true_group"] = labels[code]

            # Proxy values
            for fn in feature_names:
                row[fn] = result["proxy_values"].get(fn, np.nan)

            # Classification results
            if result["classification"]:
                for clf_name in clf_names:
                    pred = result["classification"][clf_name]
                    row[f"{clf_name}_pred"] = pred["prediction"]
                    row[f"{clf_name}_P_ALS"] = pred["probability_ALS"]

            rows.append(row)
            print(f"  [{i+1}/{len(codes)}] {code}", end="")
            if result["classification"]:
                # Show consensus
                preds = [result["classification"][cn]["prediction"] for cn in clf_names]
                consensus = max(set(preds), key=preds.count)
                n_als = sum(1 for p in preds if p == "ALS")
                print(f"  -> {consensus} ({n_als}/{len(preds)} classifiers)")
            else:
                print()

        df = pd.DataFrame(rows)

        # Summary if labels provided
        if labels and has_clf and len(df) > 0 and "true_group" in df.columns:
            print(f"\n  Classification accuracy on {len(df)} samples:")
            for clf_name in clf_names:
                pred_col = f"{clf_name}_pred"
                if pred_col in df.columns:
                    correct = (df["true_group"] == df[pred_col]).sum()
                    acc = correct / len(df)
                    # AUC if we have probabilities
                    prob_col = f"{clf_name}_P_ALS"
                    y_true = (df["true_group"] == "ALS").astype(int)
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(y_true, df[prob_col])
                        print(f"    {clf_name}: acc={acc:.3f} ({correct}/{len(df)}), AUC={auc:.3f}")
                    except Exception:
                        print(f"    {clf_name}: acc={acc:.3f} ({correct}/{len(df)})")

        return df

    @staticmethod
    def _find_file(tp_dir: str, code: str) -> Optional[str]:
        tp_path = Path(tp_dir)
        exact = tp_path / f"{code}.xlsx"
        if exact.exists():
            return str(exact)
        for f in tp_path.glob("*.xlsx"):
            if f.name.startswith("~$"):
                continue
            stem = f.stem
            for suffix in ["_0h", "_6h", "_24h"]:
                if stem.endswith(suffix):
                    file_code = stem[:-len(suffix)]
                    if file_code == code or file_code.replace('.', '_') == code:
                        return str(f)
        return None


# =============================================================================
# Bundle export function (called from step2 or standalone)
# =============================================================================

def export_inference_bundle(
    arch_file: str,
    fold_paths: List[Path],
    selected_features: List[Dict],
    all_conv_feats: List[np.ndarray],    # per fold: [N, 3, F]
    all_fc_W: List[np.ndarray],          # per fold: [latent_dim, F]
    all_fc_b: List[np.ndarray],          # per fold: [latent_dim]
    all_attn_params: List[Dict],
    Z_target: np.ndarray,                # [N, latent_dim]
    latent_dim: int,
    output_path: str,
    train_codes: List[str] = None,
    train_labels: np.ndarray = None,
    export_classifier: bool = True,
):
    """
    Export everything needed for inference on new samples.
    """
    print("\n" + "=" * 70)
    print("EXPORTING INFERENCE BUNDLE")
    print("=" * 70)

    # Read architecture source code
    with open(arch_file, "r") as f:
        arch_code = f.read()

    # Determine which folds are actually needed
    needed_folds = set(sf["fold_idx"] for sf in selected_features)
    print(f"  Features: {len(selected_features)}")
    print(f"  Folds needed: {sorted(needed_folds)}")

    # Build fold model data (only needed folds)
    fold_models = {}
    for fi in needed_folds:
        # Load the state dict directly from the .pth file
        sd = torch.load(str(fold_paths[fi]), map_location="cpu", weights_only=False)
        # Clean up keys
        clean_sd = OrderedDict()
        for k, v in sd.items():
            clean_sd[k.replace("module.", "")] = v

        fold_models[fi] = {
            "state_dict": clean_sd,
            "arch_module_code": arch_code,
            "fc_W": all_fc_W[fi],
            "fc_b": all_fc_b[fi],
            "attn_params": all_attn_params[fi],
            "flatten_size": int(all_fc_W[fi].shape[1]),
            "fold_path_name": fold_paths[fi].name,
        }
    print(f"  Stored {len(fold_models)} fold model(s)")

    # Compute normalization params per feature (from training data)
    normalization = {}
    for sf in selected_features:
        fi = sf["fold_idx"]
        d = sf["dim"]
        fname = sf["feature"]

        # Compute this fold's z_agg for dim d on training data
        from step2_encoder_weight_proxy_v3 import compute_z_agg_analytical
        z_agg, _ = compute_z_agg_analytical(
            all_conv_feats[fi], all_fc_W[fi], all_fc_b[fi], all_attn_params[fi]
        )
        z_fold_d = z_agg[:, d]

        # Determine sign flip by correlation with Z_target
        z_target_d = Z_target[:, d]
        r = float(np.corrcoef(z_target_d, z_fold_d)[0, 1])
        sign_flip = r < 0

        if sign_flip:
            z_fold_d = -z_fold_d

        normalization[fname] = {
            "sign_flip": sign_flip,
            "train_mean": float(z_fold_d.mean()),
            "train_std": float(z_fold_d.std()),
            "target_mean": float(z_target_d.mean()),
            "target_std": float(z_target_d.std()),
            "raw_correlation": float(r),
        }
        print(f"  {fname}: sign_flip={sign_flip}, r={abs(r):.4f}")

    # Build classifier if requested
    classifier_params = None
    if export_classifier and train_labels is not None:
        print("  Training classifiers for export...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # Build training proxy matrix using analytical z values
        proxy_train = np.zeros((len(train_labels), len(selected_features)))
        for i, sf in enumerate(selected_features):
            fi = sf["fold_idx"]
            d = sf["dim"]
            fname = sf["feature"]
            z_agg_fi, _ = compute_z_agg_analytical(
                all_conv_feats[fi], all_fc_W[fi], all_fc_b[fi], all_attn_params[fi]
            )
            z_d = z_agg_fi[:, d]
            norm = normalization[fname]
            if norm["sign_flip"]:
                z_d = -z_d
            z_d = (z_d - norm["train_mean"]) / max(norm["train_std"], 1e-12)
            z_d = z_d * norm["target_std"] + norm["target_mean"]
            proxy_train[:, i] = z_d

        classifier_params = {}
        SEED = 42
        classifiers = {
            "SVM": (StandardScaler(), SVC(kernel="linear", C=1.0, probability=True,
                                           class_weight="balanced", random_state=SEED)),
            "RF": (None, RandomForestClassifier(n_estimators=100, random_state=SEED,
                                                 class_weight="balanced_subsample")),
            "LR": (StandardScaler(), LogisticRegression(max_iter=5000, C=1.0,
                                                         class_weight="balanced",
                                                         solver="liblinear", random_state=SEED)),
        }
        for clf_name, (scaler, clf) in classifiers.items():
            X_tr = proxy_train.copy()
            if scaler is not None:
                X_tr = scaler.fit_transform(X_tr)
            clf.fit(X_tr, train_labels)
            classifier_params[clf_name] = {
                "classifier": clf,
                "scaler": scaler,
            }
        print(f"  Exported 3 classifiers (SVM, RF, LR)")

    # Assemble bundle
    bundle = {
        "features": selected_features,
        "fold_models": fold_models,
        "normalization": normalization,
        "classifier_params": classifier_params,
        "metadata": {
            "latent_dim": latent_dim,
            "n_training_samples": int(Z_target.shape[0]),
            "n_features": len(selected_features),
            "strategy": "E_analytical",
            "train_codes": train_codes,
        },
    }

    torch.save(bundle, output_path)
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n  Bundle saved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Contains: {len(fold_models)} fold model(s), "
          f"{len(selected_features)} features, "
          f"{'classifiers' if classifier_params else 'no classifiers'}")

    return bundle


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Proxy Latent Inference Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- Export ---
    p_export = subparsers.add_parser("export", help="Export inference bundle from step2 outputs")
    p_export.add_argument("--arch_file", type=str, required=True)
    p_export.add_argument("--model_dir", type=str, required=True)
    p_export.add_argument("--step2_dir", type=str, required=True,
                          help="Step2 output dir (needs step2_manifest.json, proxy_latent_matrix.csv)")
    p_export.add_argument("--opt_config", type=str, required=True)
    p_export.add_argument("--labels_csv", type=str, required=True)
    p_export.add_argument("--tp_dirs", type=str, required=True)
    p_export.add_argument("--step1_dir", type=str, default=None)
    p_export.add_argument("--latent_dim", type=int, default=512)
    p_export.add_argument("--output_bundle", type=str, default="./inference_bundle.pt")
    p_export.add_argument("--export_classifier", action="store_true", default=True)

    # --- Predict single ---
    p_pred = subparsers.add_parser("predict", help="Predict proxy latent values for a single sample")
    p_pred.add_argument("--bundle", type=str, required=True)
    p_pred.add_argument("--eem_files", type=str, required=True,
                        help="Comma-separated: path_0h,path_6h,path_24h")

    # --- Predict batch ---
    p_batch = subparsers.add_parser("predict_batch", help="Predict for multiple samples")
    p_batch.add_argument("--bundle", type=str, required=True)
    p_batch.add_argument("--tp_dirs", type=str, required=True,
                         help="Comma-separated: dir_0h,dir_6h,dir_24h")
    p_batch.add_argument("--codes", type=str, default=None,
                         help="Comma-separated sample codes. If omitted, auto-discovers from dirs.")
    p_batch.add_argument("--output_csv", type=str, default="./batch_predictions.csv")

    # --- Predict and classify ---
    p_clf = subparsers.add_parser("predict_classify",
                                   help="Predict proxy values + classify ALS/CTRL (single sample)")
    p_clf.add_argument("--bundle", type=str, required=True)
    p_clf.add_argument("--eem_files", type=str, required=True)

    # --- Predict and classify batch ---
    p_clf_batch = subparsers.add_parser("predict_classify_batch",
                                         help="Predict + classify a whole dataset from TP directories")
    p_clf_batch.add_argument("--bundle", type=str, required=True)
    p_clf_batch.add_argument("--tp_dirs", type=str, required=True,
                              help="Comma-separated: dir_0h,dir_6h,dir_24h")
    p_clf_batch.add_argument("--labels_csv", type=str, default=None,
                              help="Optional CSV with 'code' and 'group' columns for accuracy reporting")
    p_clf_batch.add_argument("--codes", type=str, default=None,
                              help="Comma-separated sample codes. If omitted, auto-discovers from dirs.")
    p_clf_batch.add_argument("--output_csv", type=str, default="./batch_classify_results.csv")

    args = parser.parse_args()

    if args.command == "export":
        _cmd_export(args)
    elif args.command == "predict":
        _cmd_predict(args)
    elif args.command == "predict_batch":
        _cmd_predict_batch(args)
    elif args.command == "predict_classify":
        _cmd_predict_classify(args)
    elif args.command == "predict_classify_batch":
        _cmd_predict_classify_batch(args)
    else:
        parser.print_help()


def _cmd_export(args):
    """Export inference bundle."""
    from step2_encoder_weight_proxy_v3 import (
        import_arch_module, load_labels, list_sample_codes,
        load_all_samples, load_model, extract_conv_features,
        extract_fc_weights, extract_attention_weights,
        compute_z_agg_analytical, ensemble_sign_align,
        parse_selected_features,
    )

    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]

    # Load data
    codes, y, groups = load_labels(args.labels_csv)
    common = set(list_sample_codes(tp_dirs[0])) & set(list_sample_codes(tp_dirs[1])) & set(list_sample_codes(tp_dirs[2]))
    keep = [c in common for c in codes]
    codes = [c for c, k in zip(codes, keep) if k]
    y = y[np.array(keep)]

    X_np = load_all_samples(codes, tp_dirs)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    # Load models
    arch_mod = import_arch_module(args.arch_file)
    model_dir = Path(args.model_dir)
    fold_paths = sorted(model_dir.glob("best_model_fold*.pth"))
    if not fold_paths:
        fold_paths = sorted(model_dir.glob("best_model_fold_*.pth"))

    all_conv_feats, all_fc_W, all_fc_b, all_attn_params, all_z_agg = [], [], [], [], []
    for fp in fold_paths:
        model = load_model(arch_mod, str(fp), args.latent_dim, X_tensor)
        all_conv_feats.append(extract_conv_features(model, X_tensor))
        W, b = extract_fc_weights(model)
        all_fc_W.append(W)
        all_fc_b.append(b)
        all_attn_params.append(extract_attention_weights(model))
        z_agg, _ = compute_z_agg_analytical(all_conv_feats[-1], W, b, all_attn_params[-1])
        all_z_agg.append(z_agg)
        del model

    # Z_target
    Z_target = ensemble_sign_align(all_z_agg)
    if args.step1_dir:
        z_path = Path(args.step1_dir) / "ensemble_z_agg.csv"
        if z_path.exists():
            df_z = pd.read_csv(z_path)
            dim_cols = [c for c in df_z.columns if c.startswith("dim")]
            Z_target = df_z[dim_cols].values

    # Parse features
    with open(args.opt_config) as f:
        opt_config = json.load(f)
    selected_features = parse_selected_features(opt_config, fold_paths)

    # Export
    export_inference_bundle(
        arch_file=args.arch_file,
        fold_paths=fold_paths,
        selected_features=selected_features,
        all_conv_feats=all_conv_feats,
        all_fc_W=all_fc_W,
        all_fc_b=all_fc_b,
        all_attn_params=all_attn_params,
        Z_target=Z_target,
        latent_dim=args.latent_dim,
        output_path=args.output_bundle,
        train_codes=codes,
        train_labels=y,
        export_classifier=args.export_classifier,
    )


def _cmd_predict(args):
    """Predict single sample."""
    files = [f.strip() for f in args.eem_files.split(",")]
    engine = ProxyLatentInference.from_bundle(args.bundle)
    result = engine.predict_single(files)

    print("\nProxy Latent Values:")
    for fname, val in result.items():
        print(f"  {fname}: {val:.6f}")


def _cmd_predict_batch(args):
    """Predict batch."""
    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]
    engine = ProxyLatentInference.from_bundle(args.bundle)

    # Discover codes
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
    else:
        # Auto-discover from first TP dir
        codes = []
        for f in Path(tp_dirs[0]).glob("*.xlsx"):
            if f.name.startswith("~$"):
                continue
            name = f.stem
            for suffix in ["_0h", "_6h", "_24h"]:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            codes.append(name)
        codes = sorted(set(codes))

    df = engine.predict_batch(codes, tp_dirs)
    df.to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved: {args.output_csv} ({len(df)} samples)")


def _cmd_predict_classify(args):
    """Predict and classify single sample."""
    files = [f.strip() for f in args.eem_files.split(",")]
    engine = ProxyLatentInference.from_bundle(args.bundle)
    result = engine.predict_and_classify(files)

    print("\nProxy Latent Values:")
    for fname, val in result["proxy_values"].items():
        print(f"  {fname}: {val:.6f}")

    if result["classification"]:
        print("\nClassification:")
        for clf_name, pred in result["classification"].items():
            prob_str = f" (P(ALS)={pred['probability_ALS']:.3f})" if pred["probability_ALS"] is not None else ""
            print(f"  {clf_name}: {pred['prediction']}{prob_str}")
    else:
        print(f"\n{result.get('warning', 'No classifier available')}")


def _cmd_predict_classify_batch(args):
    """Predict and classify a whole dataset."""
    tp_dirs = [d.strip() for d in args.tp_dirs.split(",")]
    engine = ProxyLatentInference.from_bundle(args.bundle)

    # Discover sample codes
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
    else:
        # Auto-discover: find codes present in ALL 3 TP dirs
        per_dir_codes = []
        for tp_dir in tp_dirs:
            dir_codes = set()
            for f in Path(tp_dir).glob("*.xlsx"):
                if f.name.startswith("~$"):
                    continue
                name = f.stem
                for suffix in ["_0h", "_6h", "_24h"]:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
                dir_codes.add(name)
            per_dir_codes.append(dir_codes)
        common_codes = per_dir_codes[0]
        for s in per_dir_codes[1:]:
            common_codes = common_codes & s
        codes = sorted(common_codes)

    print(f"\nFound {len(codes)} samples across all timepoint directories")

    # Load labels if provided
    labels = None
    if args.labels_csv:
        df_labels = pd.read_csv(args.labels_csv)
        labels = dict(zip(df_labels["code"].astype(str), df_labels["group"].astype(str)))
        n_matched = sum(1 for c in codes if c in labels)
        print(f"Labels loaded: {n_matched}/{len(codes)} samples matched")

    # Run batch
    print("\nProcessing samples...")
    df = engine.predict_classify_batch(codes, tp_dirs, labels)

    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved: {args.output_csv} ({len(df)} samples)")


if __name__ == "__main__":
    main()
