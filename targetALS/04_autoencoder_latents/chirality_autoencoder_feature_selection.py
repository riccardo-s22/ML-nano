#!/usr/bin/env python3
"""
CHIRALITY-INFORMED AUTOENCODER FOR FEATURE SELECTION
=====================================================

A lightweight autoencoder that extracts features from known chirality ROIs
for ALS vs CTRL classification. Designed for small sample sizes (~40 subjects).

KEY DESIGN PRINCIPLES:
----------------------
1. Physics-informed: Only extracts patches around known chirality centers
2. Shallow architecture: 2 conv layers to prevent overfitting on limited data
3. Dual attention: Learns importance of both chiralities AND timepoints
4. Feature extraction: Outputs interpretable latent features for downstream ML
5. No height normalization: Uses only spectral coordinates (emission, excitation)

ARCHITECTURE:
-------------
Input: 3 timepoints × 12 chiralities × patch_size × patch_size
       ↓
Patch Encoder (shared across all chiralities/timepoints)
       ↓
Latent vectors: (batch, 3, 12, latent_dim)
       ↓
    ├── Chirality Attention → weighted sum over 12 chiralities
    │         ↓
    │   Timepoint Attention → weighted sum over 3 timepoints
    │         ↓
    │   Aggregated latent → Classifier → ALS/CTRL prediction
    │
    └── Patch Decoder → Reconstructed patches (regularization)

OUTPUT FEATURES FOR ML:
-----------------------
1. Raw latent vectors per chirality/timepoint (36 × latent_dim features)
2. Attention-weighted aggregated features
3. Chirality attention weights (which chiralities matter)
4. Timepoint attention weights (which timepoints matter)

USAGE:
------
python chirality_autoencoder_feature_selection.py \
    --data_dir /path/to/data \
    --labels_csv /path/to/sample_labels.csv \
    --output_dir ./autoencoder_results \
    --epochs 100 \
    --latent_dim 32

Author: Claude (Anthropic)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# For Excel loading
try:
    from openpyxl import load_workbook
except ImportError:
    print("Please install openpyxl: pip install openpyxl")
    sys.exit(1)


# ==============================================================================
# CHIRALITY DEFINITIONS (coordinates only, no height)
# ==============================================================================

@dataclass
class Chirality:
    """Single nanotube chirality with spectral coordinates."""
    name: str
    emission_nm: float
    excitation_nm: float

CHIRALITIES = [
    Chirality("(8,3)",  973.98,  673.94),
    Chirality("(6,5)",  987.82,  577.12),
    Chirality("(7,5)",  1047.81, 653.32),
    Chirality("(10,2)", 1080.60, 745.92),
    Chirality("(9,4)",  1131.96, 731.39),
    Chirality("(8,4)",  1130.34, 599.78),
    Chirality("(7,6)",  1138.19, 659.79),
    Chirality("(8,6)",  1200.03, 727.40),
    Chirality("(8,7)",  1288.27, 740.87),
    Chirality("(9,5)",  1262.98, 685.15),
    Chirality("(10,3)", 1267.70, 648.97),
    Chirality("(10,5)", 1282.97, 801.23),
]

TIMEPOINT_SUFFIXES = ['0h', '6h', '24h']


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_excel_as_array(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Excel EEM file and return data matrix with axes.
    
    Returns:
        data: 2D array (n_emission, n_excitation)
        emission_axis: 1D array of emission wavelengths
        excitation_axis: 1D array of excitation wavelengths
    """
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    
    # Parse header for excitation values
    header = rows[0]
    excitation_axis = []
    for h in header[1:]:
        if h is None:
            continue
        s = str(h).replace('Excitation_', '').replace('nm', '').strip()
        try:
            excitation_axis.append(float(s))
        except:
            pass
    excitation_axis = np.array(excitation_axis)
    
    # Parse data rows
    emission_axis = []
    data_rows = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        try:
            em = float(row[0])
            emission_axis.append(em)
            row_data = []
            for val in row[1:len(excitation_axis)+1]:
                if isinstance(val, (int, float)) and val is not None:
                    row_data.append(float(val))
                else:
                    row_data.append(0.0)
            data_rows.append(row_data)
        except:
            continue
    
    emission_axis = np.array(emission_axis)
    data = np.array(data_rows, dtype=np.float32)
    
    return data, emission_axis, excitation_axis


def find_nearest_idx(arr: np.ndarray, value: float) -> int:
    """Find index of nearest value in array."""
    return int(np.abs(arr - value).argmin())


def extract_chirality_patch(data: np.ndarray, 
                           emission_axis: np.ndarray,
                           excitation_axis: np.ndarray,
                           chirality: Chirality,
                           patch_size: int = 9) -> np.ndarray:
    """
    Extract a square patch centered on chirality coordinates.
    
    Args:
        data: EEM matrix (n_emission, n_excitation)
        emission_axis: Emission wavelengths
        excitation_axis: Excitation wavelengths
        chirality: Chirality object with center coordinates
        patch_size: Size of square patch (odd number recommended)
    
    Returns:
        patch: 2D array of shape (patch_size, patch_size)
    """
    # Find center indices
    em_idx = find_nearest_idx(emission_axis, chirality.emission_nm)
    ex_idx = find_nearest_idx(excitation_axis, chirality.excitation_nm)
    
    half = patch_size // 2
    
    # Handle boundary conditions with padding
    pad_top = max(0, half - em_idx)
    pad_bottom = max(0, (em_idx + half + 1) - data.shape[0])
    pad_left = max(0, half - ex_idx)
    pad_right = max(0, (ex_idx + half + 1) - data.shape[1])
    
    # Extract with boundary handling
    em_start = max(0, em_idx - half)
    em_end = min(data.shape[0], em_idx + half + 1)
    ex_start = max(0, ex_idx - half)
    ex_end = min(data.shape[1], ex_idx + half + 1)
    
    patch = data[em_start:em_end, ex_start:ex_end]
    
    # Pad if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                      mode='edge')
    
    # Ensure exact size
    if patch.shape != (patch_size, patch_size):
        # Resize if needed (rare edge case)
        from scipy.ndimage import zoom
        zoom_factors = (patch_size / patch.shape[0], patch_size / patch.shape[1])
        patch = zoom(patch, zoom_factors, order=1)
    
    return patch.astype(np.float32)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize patch to [0, 1] range."""
    pmin, pmax = patch.min(), patch.max()
    if pmax > pmin:
        return (patch - pmin) / (pmax - pmin)
    return np.zeros_like(patch)


class ChiralityPatchDataset(Dataset):
    """
    Dataset that extracts chirality patches from EEM files.
    
    Returns:
        patches: (3, 12, patch_size, patch_size) - 3 timepoints, 12 chiralities
        label: 0 or 1
        sample_id: string identifier
    """
    
    def __init__(self, 
                 sample_ids: List[str],
                 labels: List[int],
                 data_dir: str,
                 patch_size: int = 9,
                 normalize: bool = True):
        self.sample_ids = sample_ids
        self.labels = labels
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.normalize = normalize
        self._cache = {}
        
    def __len__(self):
        return len(self.sample_ids)
    
    def _find_file(self, sample_id: str, timepoint: str) -> Optional[Path]:
        """Find the Excel file for a sample and timepoint."""
        # Common naming patterns
        patterns = [
            f"{sample_id}_{timepoint}.xlsx",
            f"{sample_id}_3d__570_opj_with_emission_{timepoint}.xlsx",
            f"{sample_id}.3d__570.opj_with_emission_{timepoint}.xlsx",
            f"{sample_id}_3d_570_opj_with_emission_{timepoint}.xlsx",
            f"{sample_id}.3d_570.opj_with_emission_{timepoint}.xlsx",
        ]
        
        # Also try with dots replaced by underscores and vice versa
        sample_variants = [
            sample_id,
            sample_id.replace('.', '_'),
            sample_id.replace('_', '.'),
        ]
        
        for variant in sample_variants:
            for pattern in patterns:
                p = pattern.replace(sample_id, variant)
                filepath = self.data_dir / p
                if filepath.exists():
                    return filepath
        
        # Search directory
        for f in self.data_dir.iterdir():
            fname = f.name.lower()
            if f.suffix == '.xlsx' and timepoint in fname:
                for variant in sample_variants:
                    if variant.lower() in fname or variant.replace('.', '_').lower() in fname:
                        return f
        
        return None
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]
        
        cache_key = sample_id
        if cache_key in self._cache:
            patches = self._cache[cache_key]
        else:
            patches = []
            
            for tp in TIMEPOINT_SUFFIXES:
                filepath = self._find_file(sample_id, tp)
                if filepath is None:
                    raise FileNotFoundError(f"Could not find file for {sample_id} at {tp}")
                
                data, em_axis, ex_axis = load_excel_as_array(str(filepath))
                
                tp_patches = []
                for chirality in CHIRALITIES:
                    patch = extract_chirality_patch(
                        data, em_axis, ex_axis, chirality, self.patch_size
                    )
                    if self.normalize:
                        patch = normalize_patch(patch)
                    tp_patches.append(patch)
                
                patches.append(np.stack(tp_patches, axis=0))  # (12, H, W)
            
            patches = np.stack(patches, axis=0)  # (3, 12, H, W)
            self._cache[cache_key] = patches
        
        return (
            torch.FloatTensor(patches),
            torch.LongTensor([label]),
            sample_id
        )


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class PatchEncoder(nn.Module):
    """
    Shallow CNN encoder for chirality patches.
    
    Input: (batch, patch_size, patch_size)
    Output: (batch, latent_dim)
    """
    
    def __init__(self, patch_size: int, latent_dim: int):
        super().__init__()
        
        # Two conv layers - shallow to avoid overfitting
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # patch_size/2
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Always 2x2
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, latent_dim),
            nn.ReLU(),
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        """
        Args:
            x: (batch, H, W) single-channel patch
        Returns:
            z: (batch, latent_dim)
        """
        x = x.unsqueeze(1)  # Add channel dim: (batch, 1, H, W)
        x = self.conv(x)
        z = self.fc(x)
        return z


class PatchDecoder(nn.Module):
    """
    Decoder to reconstruct patches from latent vectors.
    
    Input: (batch, latent_dim)
    Output: (batch, patch_size, patch_size)
    """
    
    def __init__(self, patch_size: int, latent_dim: int):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32 * 4),
            nn.ReLU(),
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_recon: (batch, patch_size, patch_size)
        """
        x = self.fc(z)
        x = x.view(-1, 32, 2, 2)
        x = self.deconv(x)
        
        # Resize to exact patch size
        x = nn.functional.interpolate(x, size=(self.patch_size, self.patch_size), 
                                      mode='bilinear', align_corners=False)
        x = x.squeeze(1)  # Remove channel dim
        return x


class ChiralityAttention(nn.Module):
    """
    Attention over 12 chirality latent vectors.
    Learns which chiralities are most important for classification.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        
    def forward(self, z):
        """
        Args:
            z: (batch, n_chiralities, latent_dim)
        Returns:
            z_agg: (batch, latent_dim) - weighted sum
            weights: (batch, n_chiralities) - attention weights
        """
        scores = self.attention(z).squeeze(-1)  # (batch, n_chiralities)
        weights = torch.softmax(scores, dim=1)
        z_agg = (z * weights.unsqueeze(-1)).sum(dim=1)
        return z_agg, weights


class TimepointAttention(nn.Module):
    """
    Attention over 3 timepoint latent vectors.
    Learns which timepoints are most important for classification.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        
    def forward(self, z):
        """
        Args:
            z: (batch, n_timepoints, latent_dim)
        Returns:
            z_agg: (batch, latent_dim) - weighted sum
            weights: (batch, n_timepoints) - attention weights
        """
        scores = self.attention(z).squeeze(-1)  # (batch, n_timepoints)
        weights = torch.softmax(scores, dim=1)
        z_agg = (z * weights.unsqueeze(-1)).sum(dim=1)
        return z_agg, weights


class ChiralityAutoencoder(nn.Module):
    """
    Complete chirality-informed autoencoder with dual attention.
    
    Architecture:
        Input patches (batch, 3, 12, H, W)
              ↓
        Shared PatchEncoder → latent (batch, 3, 12, latent_dim)
              ↓
        ChiralityAttention → (batch, 3, latent_dim)
              ↓
        TimepointAttention → (batch, latent_dim)
              ↓
        Classifier → (batch, 2)
              
        Also: PatchDecoder for reconstruction loss
    """
    
    def __init__(self, patch_size: int = 9, latent_dim: int = 32, num_classes: int = 2):
        super().__init__()
        
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Shared encoder/decoder
        self.encoder = PatchEncoder(patch_size, latent_dim)
        self.decoder = PatchDecoder(patch_size, latent_dim)
        
        # Dual attention
        self.chirality_attention = ChiralityAttention(latent_dim)
        self.timepoint_attention = TimepointAttention(latent_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
        
    def encode_all_patches(self, patches):
        """
        Encode all patches through shared encoder.
        
        Args:
            patches: (batch, 3, 12, H, W)
        Returns:
            latents: (batch, 3, 12, latent_dim)
        """
        batch_size = patches.shape[0]
        n_tp = patches.shape[1]
        n_chir = patches.shape[2]
        
        # Flatten to process all patches
        flat = patches.view(-1, self.patch_size, self.patch_size)
        z_flat = self.encoder(flat)
        
        # Reshape back
        latents = z_flat.view(batch_size, n_tp, n_chir, self.latent_dim)
        return latents
    
    def decode_all_patches(self, latents):
        """
        Decode all latent vectors through shared decoder.
        
        Args:
            latents: (batch, 3, 12, latent_dim)
        Returns:
            recon: (batch, 3, 12, H, W)
        """
        batch_size = latents.shape[0]
        n_tp = latents.shape[1]
        n_chir = latents.shape[2]
        
        flat = latents.view(-1, self.latent_dim)
        recon_flat = self.decoder(flat)
        
        recon = recon_flat.view(batch_size, n_tp, n_chir, self.patch_size, self.patch_size)
        return recon
    
    def forward(self, patches):
        """
        Full forward pass.
        
        Args:
            patches: (batch, 3, 12, H, W)
        
        Returns:
            recon: Reconstructed patches (batch, 3, 12, H, W)
            logits: Classification logits (batch, num_classes)
            chirality_weights: (batch, 12) attention weights
            timepoint_weights: (batch, 3) attention weights
            latents: (batch, 3, 12, latent_dim) all latent vectors
        """
        batch_size = patches.shape[0]
        
        # Encode all patches
        latents = self.encode_all_patches(patches)  # (batch, 3, 12, latent_dim)
        
        # Aggregate over chiralities (per timepoint)
        # latents: (batch, 3, 12, latent_dim)
        chirality_agg = []
        chirality_weights_list = []
        for t in range(3):
            z_t = latents[:, t, :, :]  # (batch, 12, latent_dim)
            z_agg_t, w_t = self.chirality_attention(z_t)
            chirality_agg.append(z_agg_t)
            chirality_weights_list.append(w_t)
        
        chirality_agg = torch.stack(chirality_agg, dim=1)  # (batch, 3, latent_dim)
        chirality_weights = torch.stack(chirality_weights_list, dim=1).mean(dim=1)  # (batch, 12) avg across tp
        
        # Aggregate over timepoints
        z_final, timepoint_weights = self.timepoint_attention(chirality_agg)  # (batch, latent_dim)
        
        # Classify
        logits = self.classifier(z_final)
        
        # Reconstruct
        recon = self.decode_all_patches(latents)
        
        return recon, logits, chirality_weights, timepoint_weights, latents
    
    def extract_features(self, patches):
        """
        Extract features for downstream ML.
        
        Args:
            patches: (batch, 3, 12, H, W)
        
        Returns:
            features: dict with various feature representations
        """
        with torch.no_grad():
            recon, logits, chir_w, tp_w, latents = self.forward(patches)
        
        batch_size = patches.shape[0]
        
        # 1. Flattened latents (all 36 chirality-timepoint combinations)
        flat_latents = latents.view(batch_size, -1).cpu().numpy()  # (batch, 3*12*latent_dim)
        
        # 2. Mean latent per chirality (averaged over timepoints)
        chir_latents = latents.mean(dim=1).view(batch_size, -1).cpu().numpy()  # (batch, 12*latent_dim)
        
        # 3. Mean latent per timepoint (averaged over chiralities)
        tp_latents = latents.mean(dim=2).view(batch_size, -1).cpu().numpy()  # (batch, 3*latent_dim)
        
        # 4. Attention-weighted aggregated latent
        # Recompute to get single vector
        chirality_agg = []
        for t in range(3):
            z_t = latents[:, t, :, :]
            z_agg_t, _ = self.chirality_attention(z_t)
            chirality_agg.append(z_agg_t)
        chirality_agg = torch.stack(chirality_agg, dim=1)
        z_final, _ = self.timepoint_attention(chirality_agg)
        agg_latent = z_final.cpu().numpy()  # (batch, latent_dim)
        
        # 5. Attention weights
        chir_weights = chir_w.cpu().numpy()  # (batch, 12)
        tp_weights = tp_w.cpu().numpy()  # (batch, 3)
        
        return {
            'flat_latents': flat_latents,
            'chirality_latents': chir_latents,
            'timepoint_latents': tp_latents,
            'aggregated_latent': agg_latent,
            'chirality_weights': chir_weights,
            'timepoint_weights': tp_weights,
        }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_epoch(model, loader, optimizer, device, alpha=1.0, beta=1.0):
    """
    Train for one epoch.
    
    Loss = alpha * reconstruction_loss + beta * classification_loss
    """
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_class = 0
    correct = 0
    total = 0
    
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    for patches, labels, _ in loader:
        patches = patches.to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        
        recon, logits, _, _, _ = model(patches)
        
        recon_loss = recon_criterion(recon, patches)
        class_loss = class_criterion(logits, labels)
        loss = alpha * recon_loss + beta * class_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_class += class_loss.item()
        
        _, pred = logits.max(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    n = len(loader)
    return total_loss/n, total_recon/n, total_class/n, 100*correct/total


def validate(model, loader, device, alpha=1.0, beta=1.0):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_recon = 0
    total_class = 0
    correct = 0
    total = 0
    
    all_probs = []
    all_labels = []
    
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for patches, labels, _ in loader:
            patches = patches.to(device)
            labels = labels.squeeze().to(device)
            
            recon, logits, _, _, _ = model(patches)
            
            recon_loss = recon_criterion(recon, patches)
            class_loss = class_criterion(logits, labels)
            loss = alpha * recon_loss + beta * class_loss
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_class += class_loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    n = len(loader)
    acc = 100*correct/total
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return total_loss/n, total_recon/n, total_class/n, acc, auc


# ==============================================================================
# FEATURE EXTRACTION & DOWNSTREAM ML
# ==============================================================================

def extract_all_features(model, loader, device):
    """Extract features from all samples."""
    model.eval()
    
    all_features = {
        'flat_latents': [],
        'chirality_latents': [],
        'timepoint_latents': [],
        'aggregated_latent': [],
        'chirality_weights': [],
        'timepoint_weights': [],
    }
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for patches, labels, sample_ids in loader:
            patches = patches.to(device)
            
            features = model.extract_features(patches)
            
            for key in all_features:
                all_features[key].append(features[key])
            
            all_labels.extend(labels.squeeze().numpy())
            all_ids.extend(sample_ids)
    
    # Concatenate
    for key in all_features:
        all_features[key] = np.vstack(all_features[key])
    
    return all_features, np.array(all_labels), all_ids


def evaluate_downstream_ml(features_dict, labels, feature_type='aggregated_latent', cv_folds=5):
    """
    Evaluate extracted features with traditional ML classifiers.
    """
    X = features_dict[feature_type]
    y = labels
    
    scaler = StandardScaler()
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM_RBF': SVC(kernel='rbf', probability=True, class_weight='balanced'),
    }
    
    results = {name: {'acc': [], 'auc': []} for name in models}
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            results[name]['acc'].append(accuracy_score(y_val, y_pred))
            try:
                results[name]['auc'].append(roc_auc_score(y_val, y_prob))
            except:
                results[name]['auc'].append(0.5)
    
    # Average
    summary = {}
    for name in results:
        summary[name] = {
            'accuracy': np.mean(results[name]['acc']),
            'accuracy_std': np.std(results[name]['acc']),
            'auc': np.mean(results[name]['auc']),
            'auc_std': np.std(results[name]['auc']),
        }
    
    return summary


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_attention_weights(chirality_weights, timepoint_weights, output_path):
    """Plot average attention weights."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chirality attention
    ax = axes[0]
    chir_names = [c.name for c in CHIRALITIES]
    mean_w = chirality_weights.mean(axis=0)
    std_w = chirality_weights.std(axis=0)
    
    bars = ax.bar(range(len(chir_names)), mean_w, yerr=std_w, capsize=3, 
                  color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(chir_names)))
    ax.set_xticklabels(chir_names, rotation=45, ha='right')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Chirality Attention Weights')
    ax.axhline(1/12, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax.legend()
    
    # Timepoint attention
    ax = axes[1]
    tp_names = TIMEPOINT_SUFFIXES
    mean_w = timepoint_weights.mean(axis=0)
    std_w = timepoint_weights.std(axis=0)
    
    bars = ax.bar(range(len(tp_names)), mean_w, yerr=std_w, capsize=3,
                  color='coral', edgecolor='black')
    ax.set_xticks(range(len(tp_names)))
    ax.set_xticklabels(tp_names)
    ax.set_ylabel('Attention Weight')
    ax.set_title('Timepoint Attention Weights')
    ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    
    # Reconstruction loss
    ax = axes[0, 1]
    ax.plot(history['train_recon'], label='Train')
    ax.plot(history['val_recon'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    
    # Classification loss
    ax = axes[1, 0]
    ax.plot(history['train_class'], label='Train')
    ax.plot(history['val_class'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classification Loss')
    ax.set_title('Classification Loss')
    ax.legend()
    
    # Accuracy
    ax = axes[1, 1]
    ax.plot(history['train_acc'], label='Train')
    ax.plot(history['val_acc'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_downstream_comparison(results_by_feature, output_path):
    """Compare downstream ML performance across feature types."""
    feature_types = list(results_by_feature.keys())
    models = list(results_by_feature[feature_types[0]].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(feature_types))
    width = 0.25
    
    colors = ['steelblue', 'coral', 'seagreen']
    
    # Accuracy
    ax = axes[0]
    for i, model in enumerate(models):
        accs = [results_by_feature[ft][model]['accuracy'] for ft in feature_types]
        stds = [results_by_feature[ft][model]['accuracy_std'] for ft in feature_types]
        ax.bar(x + i*width, accs, width, yerr=stds, label=model, color=colors[i], capsize=3)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Downstream ML - Accuracy')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_types, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # AUC
    ax = axes[1]
    for i, model in enumerate(models):
        aucs = [results_by_feature[ft][model]['auc'] for ft in feature_types]
        stds = [results_by_feature[ft][model]['auc_std'] for ft in feature_types]
        ax.bar(x + i*width, aucs, width, yerr=stds, label=model, color=colors[i], capsize=3)
    
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Downstream ML - AUC')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_types, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chirality-informed autoencoder for feature selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing Excel EEM files')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV with columns: code, group (ALS/CTRL)')
    parser.add_argument('--output_dir', type=str, default='./autoencoder_results')
    parser.add_argument('--patch_size', type=int, default=9,
                       help='Size of patches around chirality centers')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension per patch')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    if 'sample_id' in labels_df.columns:
        labels_df['code'] = labels_df['sample_id']
    
    sample_ids = labels_df['code'].astype(str).tolist()
    groups = labels_df['group'].tolist()
    labels = np.array([1 if g == 'ALS' else 0 for g in groups])
    
    print(f"\nLoaded {len(sample_ids)} samples")
    print(f"  ALS: {sum(labels)}, CTRL: {len(labels) - sum(labels)}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    all_features_collected = []
    all_labels_collected = []
    all_chirality_weights = []
    all_timepoint_weights = []
    
    print(f"\n{'='*70}")
    print(f"TRAINING WITH {args.n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sample_ids, labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{args.n_folds}")
        print(f"{'='*70}")
        
        # Split data
        train_ids = [sample_ids[i] for i in train_idx]
        train_labels = labels[train_idx].tolist()
        val_ids = [sample_ids[i] for i in val_idx]
        val_labels = labels[val_idx].tolist()
        
        print(f"  Train: {len(train_ids)} ({sum(train_labels)} ALS)")
        print(f"  Val: {len(val_ids)} ({sum(val_labels)} ALS)")
        
        # Datasets
        train_dataset = ChiralityPatchDataset(
            train_ids, train_labels, args.data_dir, args.patch_size
        )
        val_dataset = ChiralityPatchDataset(
            val_ids, val_labels, args.data_dir, args.patch_size
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Model
        model = ChiralityAutoencoder(
            patch_size=args.patch_size,
            latent_dim=args.latent_dim,
            num_classes=2
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Training history
        history = {
            'train_loss': [], 'train_recon': [], 'train_class': [], 'train_acc': [],
            'val_loss': [], 'val_recon': [], 'val_class': [], 'val_acc': [], 'val_auc': []
        }
        
        best_val_auc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Train
            train_loss, train_recon, train_class, train_acc = train_epoch(
                model, train_loader, optimizer, device, args.alpha, args.beta
            )
            
            # Validate
            val_loss, val_recon, val_class, val_acc, val_auc = validate(
                model, val_loader, device, args.alpha, args.beta
            )
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_recon'].append(train_recon)
            history['train_class'].append(train_class)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_recon'].append(val_recon)
            history['val_class'].append(val_class)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, val_auc={val_auc:.3f}")
            
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Save fold model
        torch.save(model.state_dict(), output_dir / f'model_fold{fold+1}.pth')
        
        # Extract features from validation set
        features, val_labels_np, _ = extract_all_features(model, val_loader, device)
        
        all_features_collected.append(features)
        all_labels_collected.extend(val_labels_np)
        all_chirality_weights.append(features['chirality_weights'])
        all_timepoint_weights.append(features['timepoint_weights'])
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_auc': best_val_auc,
            'best_val_acc': max(history['val_acc']),
            'final_epoch': len(history['train_loss']),
        })
        
        print(f"  Best AUC: {best_val_auc:.3f}")
        
        # Save training history plot
        plot_training_history(history, output_dir / f'training_fold{fold+1}.png')
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    avg_auc = np.mean([r['best_val_auc'] for r in fold_results])
    std_auc = np.std([r['best_val_auc'] for r in fold_results])
    avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc = np.std([r['best_val_acc'] for r in fold_results])
    
    print(f"\nAutoencoder Classification:")
    print(f"  AUC: {avg_auc:.3f} ± {std_auc:.3f}")
    print(f"  Accuracy: {avg_acc:.1f}% ± {std_acc:.1f}%")
    
    # Aggregate attention weights
    all_chir_w = np.vstack(all_chirality_weights)
    all_tp_w = np.vstack(all_timepoint_weights)
    
    print(f"\nChirality Attention Weights (avg):")
    mean_chir_w = all_chir_w.mean(axis=0)
    for i, c in enumerate(CHIRALITIES):
        print(f"  {c.name}: {mean_chir_w[i]:.3f}")
    
    print(f"\nTimepoint Attention Weights (avg):")
    mean_tp_w = all_tp_w.mean(axis=0)
    for i, tp in enumerate(TIMEPOINT_SUFFIXES):
        print(f"  {tp}: {mean_tp_w[i]:.3f}")
    
    # Plot attention weights
    plot_attention_weights(all_chir_w, all_tp_w, output_dir / 'attention_weights.png')
    
    # Evaluate downstream ML with different feature types
    print(f"\n{'='*70}")
    print("DOWNSTREAM ML EVALUATION")
    print(f"{'='*70}")
    
    # Combine features from all folds
    combined_features = {key: np.vstack([f[key] for f in all_features_collected]) 
                        for key in all_features_collected[0].keys() 
                        if key not in ['chirality_weights', 'timepoint_weights']}
    combined_labels = np.array(all_labels_collected)
    
    feature_types_to_test = ['aggregated_latent', 'chirality_latents', 'timepoint_latents', 'flat_latents']
    
    results_by_feature = {}
    for ft in feature_types_to_test:
        print(f"\nFeature type: {ft} ({combined_features[ft].shape[1]} dims)")
        results = evaluate_downstream_ml(combined_features, combined_labels, ft, cv_folds=5)
        results_by_feature[ft] = results
        
        for model_name, metrics in results.items():
            print(f"  {model_name}: Acc={metrics['accuracy']:.3f}±{metrics['accuracy_std']:.3f}, "
                  f"AUC={metrics['auc']:.3f}±{metrics['auc_std']:.3f}")
    
    # Plot comparison
    plot_downstream_comparison(results_by_feature, output_dir / 'downstream_comparison.png')
    
    # Save features for external use
    np.savez(
        output_dir / 'extracted_features.npz',
        flat_latents=combined_features['flat_latents'],
        chirality_latents=combined_features['chirality_latents'],
        timepoint_latents=combined_features['timepoint_latents'],
        aggregated_latent=combined_features['aggregated_latent'],
        labels=combined_labels,
        chirality_names=[c.name for c in CHIRALITIES],
        timepoint_names=TIMEPOINT_SUFFIXES,
    )
    
    # Save attention weights
    attention_df = pd.DataFrame({
        **{f'chir_{c.name}': all_chir_w[:, i] for i, c in enumerate(CHIRALITIES)},
        **{f'tp_{tp}': all_tp_w[:, i] for i, tp in enumerate(TIMEPOINT_SUFFIXES)},
    })
    attention_df.to_csv(output_dir / 'attention_weights.csv', index=False)
    
    # Summary
    summary = {
        'config': vars(args),
        'cv_results': fold_results,
        'avg_auc': float(avg_auc),
        'std_auc': float(std_auc),
        'avg_acc': float(avg_acc),
        'std_acc': float(std_acc),
        'chirality_attention': {c.name: float(mean_chir_w[i]) for i, c in enumerate(CHIRALITIES)},
        'timepoint_attention': {tp: float(mean_tp_w[i]) for i, tp in enumerate(TIMEPOINT_SUFFIXES)},
        'downstream_ml': {ft: {m: {k: float(v) for k, v in metrics.items()} 
                              for m, metrics in results.items()}
                         for ft, results in results_by_feature.items()},
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Text summary
    summary_text = f"""
CHIRALITY-INFORMED AUTOENCODER - RESULTS
{'='*50}

CONFIGURATION:
  Patch size: {args.patch_size}
  Latent dim: {args.latent_dim}
  Epochs: {args.epochs}
  Learning rate: {args.lr}
  Alpha (recon): {args.alpha}
  Beta (class): {args.beta}

AUTOENCODER CLASSIFICATION (5-fold CV):
  AUC: {avg_auc:.3f} ± {std_auc:.3f}
  Accuracy: {avg_acc:.1f}% ± {std_acc:.1f}%

TOP CHIRALITIES BY ATTENTION:
{chr(10).join(f'  {i+1}. {c.name}: {mean_chir_w[i]:.3f}' for i, c in enumerate(sorted(enumerate(CHIRALITIES), key=lambda x: mean_chir_w[x[0]], reverse=True)[:5]))}

TIMEPOINT IMPORTANCE:
{chr(10).join(f'  {tp}: {mean_tp_w[i]:.3f}' for i, tp in enumerate(TIMEPOINT_SUFFIXES))}

BEST DOWNSTREAM ML (aggregated_latent features):
  RandomForest: AUC={results_by_feature['aggregated_latent']['RandomForest']['auc']:.3f}

OUTPUT FILES:
  - model_fold*.pth: Trained models
  - extracted_features.npz: Features for external ML
  - attention_weights.csv: Per-sample attention weights
  - attention_weights.png: Attention visualization
  - downstream_comparison.png: ML comparison
"""
    
    print(summary_text)
    
    with open(output_dir / 'SUMMARY.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
