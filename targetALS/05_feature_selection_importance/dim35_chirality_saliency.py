#!/usr/bin/env python3
"""
dim35-Specific Gradient Saliency at Chirality Positions
========================================================

Loads pre-trained Model 1 (fold 1) and computes:
  1. Per-timepoint gradient saliency |∂z_agg[35]/∂x_t| for dim35 specifically
  2. Attention weights α_0h, α_6h, α_24h for each sample
  3. Quantifies saliency at known chirality (n,m) positions vs background
  4. Compares dim35 chirality enrichment against random latent dims (null)
  5. Tests whether ALS vs CTRL differ in chirality-region saliency

USAGE:
------
python dim35_chirality_saliency.py \
    --model_path  <path_to_best_model_fold1.pth> \
    --tp_dirs     <out_0h>,<out_6h>,<out_24h> \
    --labels_csv  <sample_labels.csv> \
    --output_dir  ./dim35_saliency_results

NOTE: This must run on the machine where model weights + EEM data are stored.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from openpyxl import load_workbook
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# Known chirality (n,m) emission / excitation wavelengths (nm)
# These are approximate E_11 transitions for SWCNTs commonly observed in EEM
# =============================================================================
CHIRALITY_POSITIONS = {
    # (n,m): (emission_nm, excitation_nm)
    'ch6_5':  (976,  566),
    'ch7_5':  (1024, 647),
    'ch7_6':  (1120, 648),
    'ch8_3':  (952,  665),
    'ch8_4':  (1121, 588),
    'ch8_6':  (1173, 718),
    'ch8_7':  (1120, 728),   # <-- NFL-correlated (6h)
    'ch9_4':  (1101, 732),   # <-- NFL-correlated (6h)
    'ch9_5':  (1241, 670),   # <-- NFL-correlated (24h)
    'ch10_2': (1059, 734),
    'ch10_3': (1249, 638),
    'ch10_5': (1249, 786),
}

# The three chiralities most correlated with NFL concentration
NFL_CHIRALITIES = ['ch8_7', 'ch9_4', 'ch9_5']


# =============================================================================
# MODEL ARCHITECTURE (must match saved models exactly)
# =============================================================================
class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.flatten_size = None
        self.fc = None
        self.latent_dim = latent_dim
        self.encoded_shape = None
        self.input_shape = None

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.encoded_shape is None:
            self.encoded_shape = (x.shape[1], x.shape[2], x.shape[3])
        if self.fc is None:
            self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, encoded_shape, target_shape):
        super().__init__()
        self.encoded_channels, self.encoded_h, self.encoded_w = encoded_shape
        self.target_h, self.target_w = target_shape
        self.fc = nn.Linear(latent_dim, self.encoded_channels * self.encoded_h * self.encoded_w)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final_conv = nn.Conv2d(32, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.encoded_channels, self.encoded_h, self.encoded_w)
        x = self.deconv1(x); x = self.deconv2(x); x = self.deconv3(x)
        x = self.final_conv(x)
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = nn.functional.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        return self.sigmoid(x)


class AttentionAggregation(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(latent_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, latent_vectors):
        scores = self.attention_net(latent_vectors).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        aggregated = torch.sum(latent_vectors * weights.unsqueeze(-1), dim=1)
        return aggregated, weights


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, num_classes))

    def forward(self, z_agg):
        return self.classifier(z_agg)


class ConvAutoencoderWithAttention(nn.Module):
    def __init__(self, in_channels, latent_dim, num_classes):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, latent_dim)
        self.attention = AttentionAggregation(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)
        self.decoder = None
        self.latent_dim = latent_dim
        self.in_channels = in_channels

    def init_decoder(self, encoded_shape, target_shape):
        self.decoder = ConvDecoder(self.latent_dim, self.in_channels, encoded_shape, target_shape)

    def forward(self, timepoint_images):
        B, T = timepoint_images.shape[0], timepoint_images.shape[1]
        latents = [self.encoder(timepoint_images[:, t]) for t in range(T)]
        if self.decoder is None and self.encoder.encoded_shape is not None:
            self.decoder = ConvDecoder(self.latent_dim, self.in_channels,
                                       self.encoder.encoded_shape, self.encoder.input_shape).to(timepoint_images.device)
        latent_vectors = torch.stack(latents, dim=1)
        z_agg, attn = self.attention(latent_vectors)
        recons = torch.stack([self.decoder(latent_vectors[:, t]) for t in range(T)], dim=1)
        logits = self.classifier(z_agg)
        return recons, logits, attn


# =============================================================================
# DATA
# =============================================================================
class ExcelImageDataset(Dataset):
    def __init__(self, codes, labels, timepoint_dirs, file_suffix=".xlsx"):
        self.codes = codes
        self.labels = labels
        self.timepoint_dirs = timepoint_dirs
        self.file_suffix = file_suffix
        self._cache = {}

    def __len__(self):
        return len(self.codes)

    def load_excel_as_image(self, filepath):
        if filepath in self._cache:
            return self._cache[filepath]
        wb = load_workbook(filepath, data_only=True, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        data = []
        for row in rows[1:]:
            vals = []
            for cell in row[1:]:
                try:
                    vals.append(float(cell))
                except:
                    vals.append(0.0)
            data.append(vals)
        mat = np.array(data, dtype=np.float32)
        mn, mx = mat.min(), mat.max()
        if mx > mn:
            mat = (mat - mn) / (mx - mn)
        self._cache[filepath] = mat
        return mat

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        imgs = []
        for tp_dir in self.timepoint_dirs:
            fp = os.path.join(tp_dir, f"{code}{self.file_suffix}")
            img = self.load_excel_as_image(fp)
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            imgs.append(torch.FloatTensor(img))
        return torch.stack(imgs), torch.tensor(label, dtype=torch.long), code


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    codes = df['code'].tolist()
    groups = df['group'].values
    labels = (groups == 'ALS').astype(int)
    return codes, labels, groups


def get_wavelength_axes(tp_dir, code, suffix=".xlsx"):
    """Extract emission and excitation axes from one EEM file."""
    fp = os.path.join(tp_dir, f"{code}{suffix}")
    wb = load_workbook(fp, data_only=True, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    import re
    header = rows[0]
    excitation = []
    for h in header[1:]:
        if h is None:
            continue
        m = re.search(r"(\d+\.?\d*)", str(h))
        if m:
            excitation.append(float(m.group(1)))
    excitation = np.array(excitation)

    emission = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        emission.append(float(row[0]))
    emission = np.array(emission)

    return emission, excitation


def nm_to_index(axis, target_nm):
    return int(np.argmin(np.abs(axis - target_nm)))


# =============================================================================
# CORE ANALYSIS: Gradient saliency for specific latent dim
# =============================================================================

def compute_dim_gradient_saliency(model, loader, dim_idx, device, sign=1.0):
    """
    Compute |∂z_agg[dim_idx]/∂x| per timepoint, averaged across all samples.

    Returns:
        grad_maps: dict {t: np.ndarray of shape (H, W)} for t in 0,1,2
        per_sample_grads: dict {t: list of (H,W) arrays} for per-sample analysis
        attention_per_sample: list of (3,) arrays — attention weights per sample
    """
    model.eval()
    sample = next(iter(loader))[0]
    _, T, _, H, W = sample.shape

    grad_accum = {t: np.zeros((H, W), dtype=np.float64) for t in range(T)}
    per_sample_grads = {t: [] for t in range(T)}
    attention_per_sample = []
    count = 0

    for images, labels, codes in loader:
        images = images.to(device).requires_grad_(True)

        latents = [model.encoder(images[:, t]) for t in range(T)]
        z_agg, attn_weights = model.attention(torch.stack(latents, dim=1))

        # Target: signed dim value (sign from logistic regression weight)
        target = (z_agg[:, dim_idx] * sign).sum()

        model.zero_grad()
        if images.grad is not None:
            images.grad.zero_()
        target.backward(retain_graph=True)

        grads = images.grad.detach().abs()  # (B, T, C, H, W)

        for b in range(grads.shape[0]):
            for t in range(T):
                g = grads[b, t].mean(dim=0).cpu().numpy()  # (H, W)
                grad_accum[t] += g
                per_sample_grads[t].append(g)
            attention_per_sample.append(attn_weights[b].detach().cpu().numpy())
            count += 1

        images = images.detach()

    for t in range(T):
        grad_accum[t] /= count

    return grad_accum, per_sample_grads, attention_per_sample


# =============================================================================
# CHIRALITY POSITION ANALYSIS
# =============================================================================

def extract_chirality_saliency(grad_map, emission_axis, excitation_axis,
                                chirality_positions, patch_radius=3):
    """
    For each chirality position, extract mean saliency in a (2r+1)×(2r+1) patch.

    Returns:
        dict: {chirality_name: {'mean_saliency': float, 'max_saliency': float,
                                'em_idx': int, 'ex_idx': int}}
    """
    H, W = grad_map.shape
    results = {}
    for name, (em_nm, ex_nm) in chirality_positions.items():
        em_idx = nm_to_index(emission_axis, em_nm)
        ex_idx = nm_to_index(excitation_axis, ex_nm)

        r0 = max(0, em_idx - patch_radius)
        r1 = min(H, em_idx + patch_radius + 1)
        c0 = max(0, ex_idx - patch_radius)
        c1 = min(W, ex_idx + patch_radius + 1)

        patch = grad_map[r0:r1, c0:c1]
        results[name] = {
            'mean_saliency': float(patch.mean()),
            'max_saliency': float(patch.max()),
            'em_idx': em_idx,
            'ex_idx': ex_idx,
            'em_nm': em_nm,
            'ex_nm': ex_nm,
            'patch_size': patch.shape,
        }
    return results


def compute_background_saliency(grad_map, emission_axis, excitation_axis,
                                 chirality_positions, patch_radius=3):
    """
    Compute average saliency over pixels NOT in any chirality patch (background).
    """
    H, W = grad_map.shape
    mask = np.ones((H, W), dtype=bool)
    for name, (em_nm, ex_nm) in chirality_positions.items():
        em_idx = nm_to_index(emission_axis, em_nm)
        ex_idx = nm_to_index(excitation_axis, ex_nm)
        r0 = max(0, em_idx - patch_radius)
        r1 = min(H, em_idx + patch_radius + 1)
        c0 = max(0, ex_idx - patch_radius)
        c1 = min(W, ex_idx + patch_radius + 1)
        mask[r0:r1, c0:c1] = False

    bg_mean = float(grad_map[mask].mean())
    bg_std = float(grad_map[mask].std())
    return bg_mean, bg_std


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="dim35 chirality saliency analysis")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model_fold1.pth (or any single fold)")
    parser.add_argument("--tp_dirs", type=str, required=True,
                        help="Comma-separated: out_0h,out_6h,out_24h")
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dim35_saliency_results")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--target_dim", type=int, default=35,
                        help="Latent dimension to analyze (default: 35)")
    parser.add_argument("--n_null_dims", type=int, default=50,
                        help="Number of random dims for null distribution")
    parser.add_argument("--patch_radius", type=int, default=3,
                        help="Radius for chirality patch extraction")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tp_dirs = args.tp_dirs.split(',')
    tp_labels = ['0h', '6h', '24h']

    print("=" * 70)
    print(f"DIM{args.target_dim} CHIRALITY SALIENCY ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")

    # --- Load data ---
    codes, labels, groups = load_labels(args.labels_csv)
    emission_axis, excitation_axis = get_wavelength_axes(tp_dirs[0], codes[0])
    print(f"Spectral grid: {len(emission_axis)} em × {len(excitation_axis)} exc")
    print(f"Emission: {emission_axis[0]:.1f}–{emission_axis[-1]:.1f} nm")
    print(f"Excitation: {excitation_axis[0]:.0f}–{excitation_axis[-1]:.0f} nm")

    dataset = ExcelImageDataset(codes, labels, tp_dirs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Load model ---
    model = ConvAutoencoderWithAttention(in_channels=1, latent_dim=args.latent_dim, num_classes=2).to(device)
    sample_batch, _, _ = next(iter(loader))
    with torch.no_grad():
        _ = model.encoder(sample_batch[0, 0:1].to(device))
    if model.encoder.encoded_shape is not None:
        target_shape = (sample_batch.shape[3], sample_batch.shape[4])
        model.init_decoder(model.encoder.encoded_shape, target_shape)
        model = model.to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model loaded: {args.model_path}")

    # --- Determine sign from logistic regression (optional, default +1) ---
    # Collect z_agg for all samples
    z_list, y_list = [], []
    with torch.no_grad():
        for images, label, code in loader:
            images = images.to(device)
            latents = [model.encoder(images[:, t]) for t in range(3)]
            z_agg, _ = model.attention(torch.stack(latents, dim=1))
            z_list.append(z_agg.cpu().numpy())
            y_list.append(label.numpy())
    z_all = np.concatenate(z_list)
    y_all = np.concatenate(y_list)

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_all)
    clf = LogisticRegression(class_weight='balanced', max_iter=5000, solver='liblinear')
    clf.fit(z_scaled, y_all)
    dim_sign = float(np.sign(clf.coef_[0][args.target_dim])) if clf.coef_[0][args.target_dim] != 0 else 1.0
    print(f"dim{args.target_dim} classification weight sign: {dim_sign:+.0f}")

    # =========================================================================
    # ANALYSIS 1: Gradient saliency for target dim
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: Gradient Saliency for dim{args.target_dim}")
    print(f"{'='*70}")

    grad_maps, per_sample_grads, attn_per_sample = compute_dim_gradient_saliency(
        model, loader, args.target_dim, device, sign=dim_sign
    )

    # =========================================================================
    # ANALYSIS 2: Chirality position enrichment
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: Chirality Position Enrichment")
    print(f"{'='*70}")

    enrichment_rows = []
    for t in range(3):
        gmap = grad_maps[t]
        chir_sal = extract_chirality_saliency(gmap, emission_axis, excitation_axis,
                                               CHIRALITY_POSITIONS, args.patch_radius)
        bg_mean, bg_std = compute_background_saliency(gmap, emission_axis, excitation_axis,
                                                       CHIRALITY_POSITIONS, args.patch_radius)

        print(f"\n  --- {tp_labels[t]} ---")
        print(f"  Background saliency: mean={bg_mean:.6f}, std={bg_std:.6f}")
        print(f"  {'Chirality':<10} {'Mean sal.':<12} {'Max sal.':<12} {'Fold-over-BG':<14} {'Z-score':<10} {'NFL-top?'}")

        for name in sorted(chir_sal.keys()):
            s = chir_sal[name]
            fold_over_bg = s['mean_saliency'] / bg_mean if bg_mean > 0 else np.inf
            z_score = (s['mean_saliency'] - bg_mean) / bg_std if bg_std > 0 else 0
            is_nfl = '  ★' if name in NFL_CHIRALITIES else ''

            print(f"  {name:<10} {s['mean_saliency']:<12.6f} {s['max_saliency']:<12.6f} "
                  f"{fold_over_bg:<14.2f} {z_score:<10.2f} {is_nfl}")

            enrichment_rows.append({
                'timepoint': tp_labels[t],
                'chirality': name,
                'em_nm': s['em_nm'],
                'ex_nm': s['ex_nm'],
                'mean_saliency': s['mean_saliency'],
                'max_saliency': s['max_saliency'],
                'background_mean': bg_mean,
                'background_std': bg_std,
                'fold_over_background': fold_over_bg,
                'z_score': z_score,
                'is_nfl_top': name in NFL_CHIRALITIES,
            })

    enrich_df = pd.DataFrame(enrichment_rows)
    enrich_df.to_csv(outdir / 'chirality_saliency_enrichment.csv', index=False)

    # Statistical test: NFL chiralities vs others
    for t in range(3):
        sub = enrich_df[enrich_df['timepoint'] == tp_labels[t]]
        nfl_z = sub[sub['is_nfl_top']]['z_score'].values
        other_z = sub[~sub['is_nfl_top']]['z_score'].values
        if len(nfl_z) > 0 and len(other_z) > 0:
            stat, pval = stats.mannwhitneyu(nfl_z, other_z, alternative='greater')
            print(f"\n  {tp_labels[t]}: NFL chiralities vs others — "
                  f"NFL mean z={nfl_z.mean():.2f}, Other mean z={other_z.mean():.2f}, "
                  f"Mann-Whitney p={pval:.4f}")

    # =========================================================================
    # ANALYSIS 3: Attention weights (timepoint-level)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3: Attention Weights")
    print(f"{'='*70}")

    attn_arr = np.array(attn_per_sample)
    print(f"  Mean attention: 0h={attn_arr[:, 0].mean():.3f}, "
          f"6h={attn_arr[:, 1].mean():.3f}, 24h={attn_arr[:, 2].mean():.3f}")

    # ALS vs CTRL
    als_mask = y_all == 1
    ctrl_mask = y_all == 0
    for t_idx, tp in enumerate(tp_labels):
        als_attn = attn_arr[als_mask, t_idx]
        ctrl_attn = attn_arr[ctrl_mask, t_idx]
        stat, pval = stats.mannwhitneyu(als_attn, ctrl_attn, alternative='two-sided')
        print(f"  {tp}: ALS={als_attn.mean():.3f} vs CTRL={ctrl_attn.mean():.3f}, p={pval:.4f}")

    attn_df = pd.DataFrame(attn_arr, columns=['attn_0h', 'attn_6h', 'attn_24h'])
    attn_df['code'] = codes
    attn_df['group'] = groups
    attn_df.to_csv(outdir / 'attention_weights_per_sample.csv', index=False)

    # =========================================================================
    # ANALYSIS 4: Null distribution — random dims
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 4: Specificity (dim{args.target_dim} vs {args.n_null_dims} random dims)")
    print(f"{'='*70}")

    np.random.seed(SEED)
    null_dims = np.random.choice([d for d in range(args.latent_dim) if d != args.target_dim],
                                  size=args.n_null_dims, replace=False)

    # For the target dim, compute mean saliency at NFL chirality positions
    target_nfl_sal = {}
    for t in range(3):
        gmap = grad_maps[t]
        chir_sal = extract_chirality_saliency(gmap, emission_axis, excitation_axis,
                                               {k: v for k, v in CHIRALITY_POSITIONS.items() if k in NFL_CHIRALITIES},
                                               args.patch_radius)
        target_nfl_sal[t] = np.mean([chir_sal[k]['mean_saliency'] for k in chir_sal])

    # Null: compute for each random dim
    null_nfl_sal = {t: [] for t in range(3)}
    for i, d in enumerate(null_dims):
        if (i + 1) % 10 == 0:
            print(f"  Processing null dim {i+1}/{args.n_null_dims}...")
        d_sign = float(np.sign(clf.coef_[0][d])) if clf.coef_[0][d] != 0 else 1.0
        d_grads, _, _ = compute_dim_gradient_saliency(model, loader, int(d), device, sign=d_sign)
        for t in range(3):
            chir_sal = extract_chirality_saliency(d_grads[t], emission_axis, excitation_axis,
                                                   {k: v for k, v in CHIRALITY_POSITIONS.items() if k in NFL_CHIRALITIES},
                                                   args.patch_radius)
            null_nfl_sal[t].append(np.mean([chir_sal[k]['mean_saliency'] for k in chir_sal]))

    for t in range(3):
        null_arr = np.array(null_nfl_sal[t])
        pct = (null_arr < target_nfl_sal[t]).mean() * 100
        p_emp = 1 - pct / 100
        print(f"  {tp_labels[t]}: dim{args.target_dim} NFL-chirality saliency={target_nfl_sal[t]:.6f}, "
              f"null mean={null_arr.mean():.6f}±{null_arr.std():.6f}, "
              f"percentile={pct:.1f}th, p={p_emp:.4f}")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("Creating visualizations...")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for t in range(3):
        # Top row: gradient saliency maps with chirality positions marked
        ax = axes[0, t]
        extent = [excitation_axis[0], excitation_axis[-1],
                  emission_axis[0], emission_axis[-1]]
        im = ax.imshow(grad_maps[t], cmap='inferno', aspect='auto',
                       origin='lower', extent=extent)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Mark chirality positions
        for name, (em_nm, ex_nm) in CHIRALITY_POSITIONS.items():
            if em_nm < emission_axis[0] or em_nm > emission_axis[-1]:
                continue
            if ex_nm < excitation_axis[0] or ex_nm > excitation_axis[-1]:
                continue
            color = 'lime' if name in NFL_CHIRALITIES else 'cyan'
            marker = '*' if name in NFL_CHIRALITIES else 'o'
            size = 120 if name in NFL_CHIRALITIES else 40
            ax.scatter(ex_nm, em_nm, c=color, marker=marker, s=size,
                      edgecolors='white', linewidths=1, zorder=5)
            ax.annotate(name, (ex_nm, em_nm), fontsize=6, color='white',
                       xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel('Excitation (nm)')
        ax.set_ylabel('Emission (nm)')
        ax.set_title(f'dim{args.target_dim} Gradient Saliency — {tp_labels[t]}', fontweight='bold')

        # Bottom row: bar charts of chirality enrichment
        ax2 = axes[1, t]
        sub = enrich_df[enrich_df['timepoint'] == tp_labels[t]].sort_values('z_score', ascending=True)
        colors = ['#e74c3c' if nfl else '#3498db' for nfl in sub['is_nfl_top']]
        ax2.barh(range(len(sub)), sub['z_score'].values, color=colors, edgecolor='k', linewidth=0.5)
        ax2.set_yticks(range(len(sub)))
        ax2.set_yticklabels(sub['chirality'].values, fontsize=8)
        ax2.set_xlabel('Z-score (saliency vs background)')
        ax2.axvline(0, color='k', linewidth=0.5)
        ax2.axvline(1.96, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax2.set_title(f'{tp_labels[t]} — Chirality Enrichment', fontweight='bold')

    # Legend
    nfl_patch = mpatches.Patch(color='#e74c3c', label='NFL-correlated (ch8_7, ch9_4, ch9_5)')
    other_patch = mpatches.Patch(color='#3498db', label='Other chiralities')
    fig.legend(handles=[nfl_patch, other_patch], loc='lower center', ncol=2, fontsize=11)

    plt.suptitle(f'dim{args.target_dim} Gradient Saliency at Chirality Positions',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(outdir / f'dim{args.target_dim}_chirality_saliency.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Null distribution figure
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for t in range(3):
        ax = axes2[t]
        null_arr = np.array(null_nfl_sal[t])
        ax.hist(null_arr, bins=20, color='#bdc3c7', edgecolor='k', alpha=0.8)
        ax.axvline(target_nfl_sal[t], color='red', linewidth=2.5,
                   label=f'dim{args.target_dim} = {target_nfl_sal[t]:.6f}')
        pct = (null_arr < target_nfl_sal[t]).mean() * 100
        ax.set_title(f'{tp_labels[t]} — {pct:.1f}th percentile', fontweight='bold')
        ax.set_xlabel('NFL-chirality mean saliency')
        ax.set_ylabel('Count (random dims)')
        ax.legend(fontsize=9)

    plt.suptitle(f'dim{args.target_dim} Specificity: NFL-Chirality Saliency vs Null',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / f'dim{args.target_dim}_null_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Target dimension: dim{args.target_dim}")
    print(f"Attention weights (mean): 0h={attn_arr[:, 0].mean():.3f}, "
          f"6h={attn_arr[:, 1].mean():.3f}, 24h={attn_arr[:, 2].mean():.3f}")

    for t in range(3):
        null_arr = np.array(null_nfl_sal[t])
        pct = (null_arr < target_nfl_sal[t]).mean() * 100
        sub = enrich_df[enrich_df['timepoint'] == tp_labels[t]]
        nfl_mean_z = sub[sub['is_nfl_top']]['z_score'].mean()
        other_mean_z = sub[~sub['is_nfl_top']]['z_score'].mean()
        print(f"\n  {tp_labels[t]}:")
        print(f"    NFL chirality mean z-score: {nfl_mean_z:.2f}")
        print(f"    Other chirality mean z-score: {other_mean_z:.2f}")
        print(f"    Specificity percentile: {pct:.1f}th")

    print(f"\nResults saved to: {outdir}")
    print("  chirality_saliency_enrichment.csv")
    print("  attention_weights_per_sample.csv")
    print(f"  dim{args.target_dim}_chirality_saliency.png")
    print(f"  dim{args.target_dim}_null_distribution.png")


if __name__ == "__main__":
    main()
