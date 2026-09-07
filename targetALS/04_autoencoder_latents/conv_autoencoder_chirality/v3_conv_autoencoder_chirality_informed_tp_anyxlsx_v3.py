"""
Chirality-Informed Convolutional Autoencoder with Temporal Attention
=====================================================================

Modified version that incorporates known chirality positions to guide learning.

KEY MODIFICATIONS FROM ORIGINAL:
1. Chirality-weighted reconstruction loss (emphasizes chirality regions)
2. Spatial attention guided by chirality coordinates
3. Shallower architecture (2 conv blocks) to prevent overfitting
4. Dual attention: spatial (chirality) + temporal (timepoint)

The model learns to focus on the 12 known chirality positions in the EEM matrix,
rather than treating all spectral regions equally.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import os
import glob
import re
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


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


def create_chirality_mask(emission_axis: np.ndarray, 
                          excitation_axis: np.ndarray,
                          sigma_em: float = 15.0,
                          sigma_ex: float = 8.0,
                          base_weight: float = 0.1) -> np.ndarray:
    """
    Create a 2D mask that emphasizes chirality regions.
    
    Args:
        emission_axis: Array of emission wavelengths (nm)
        excitation_axis: Array of excitation wavelengths (nm)
        sigma_em: Gaussian spread in emission direction (nm)
        sigma_ex: Gaussian spread in excitation direction (nm)
        base_weight: Minimum weight for non-chirality regions
    
    Returns:
        mask: 2D array (n_emission, n_excitation) with values in [base_weight, 1]
    """
    H, W = len(emission_axis), len(excitation_axis)
    mask = np.ones((H, W), dtype=np.float32) * base_weight
    
    # Create meshgrid for vectorized computation
    em_grid, ex_grid = np.meshgrid(emission_axis, excitation_axis, indexing='ij')
    
    for chir in CHIRALITIES:
        # Gaussian bump centered at chirality position
        gauss = np.exp(
            -((em_grid - chir.emission_nm)**2 / (2 * sigma_em**2) +
              (ex_grid - chir.excitation_nm)**2 / (2 * sigma_ex**2))
        )
        mask = np.maximum(mask, gauss)
    
    # Normalize to [base_weight, 1]
    mask = np.clip(mask, base_weight, 1.0)
    
    return mask


def create_chirality_attention_prior(emission_axis: np.ndarray,
                                     excitation_axis: np.ndarray,
                                     sigma_em: float = 20.0,
                                     sigma_ex: float = 10.0) -> np.ndarray:
    """
    Create attention prior map for spatial attention mechanism.
    
    Returns soft attention map with peaks at chirality positions.
    """
    H, W = len(emission_axis), len(excitation_axis)
    attention = np.zeros((H, W), dtype=np.float32)
    
    em_grid, ex_grid = np.meshgrid(emission_axis, excitation_axis, indexing='ij')
    
    for chir in CHIRALITIES:
        gauss = np.exp(
            -((em_grid - chir.emission_nm)**2 / (2 * sigma_em**2) +
              (ex_grid - chir.excitation_nm)**2 / (2 * sigma_ex**2))
        )
        attention += gauss
    
    # Normalize to sum to 1
    attention = attention / (attention.sum() + 1e-10)
    
    return attention


# ==============================================================================
# DATASET
# ==============================================================================

class ExcelImageDataset(Dataset):
    """
    Dataset for loading Excel EEM files with chirality mask support.

    Robust to arbitrary filename suffixes inside each timepoint folder:
    it indexes *.xlsx files in each tp folder and maps sample code -> filepath.
    """

    def __init__(self, codes: List[str], labels: List[int],
                 timepoint_dirs: List[str], transform=None, strict: bool = True):
        self.codes = [self._normalize_code(c) for c in codes]
        self.labels = labels
        self.timepoint_dirs = timepoint_dirs
        self.transform = transform
        self.strict = strict

        self._cache = {}
        self._axes_cache = {}

        # Build one map per timepoint dir: code -> filepath
        self._tp_maps: Dict[str, Dict[str, str]] = {}
        for d in self.timepoint_dirs:
            self._tp_maps[d] = self._index_timepoint_dir(d)

        # Validate presence (optional but helpful)
        if self.strict:
            missing = []
            for c in self.codes:
                for d in self.timepoint_dirs:
                    if c not in self._tp_maps.get(d, {}):
                        missing.append((c, d))
            if missing:
                preview = "\n".join([f"{c} missing in {d}" for c, d in missing[:30]])
                raise FileNotFoundError(
                    "Some sample codes were not found in all timepoint folders.\n"
                    f"{preview}\n..."
                )

    def __len__(self):
        return len(self.codes)

    @staticmethod
    def _normalize_code(s: str) -> str:
        return str(s).strip().lower()

    @staticmethod
    def _extract_code_from_filename(filename: str) -> str:
        """Extract sample code from an .xlsx filename.

        Match using the full filename stem (no extension), normalized.
        Example: '10.3d__570.opj_with_emission.xlsx' -> '10.3d__570.opj_with_emission'
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        return ExcelImageDataset._normalize_code(stem)

    def _index_timepoint_dir(self, tp_dir: str) -> Dict[str, str]:
        """
        Index all .xlsx files in a timepoint directory and map code -> filepath.
        If multiple files map to the same code, keeps the shortest filename.
        """
        out: Dict[str, str] = {}
        for fp in glob.glob(os.path.join(tp_dir, "*.xlsx")):
            base = os.path.basename(fp)
            if base.startswith("~$") or base.startswith("~"):
                continue

            code = self._extract_code_from_filename(base)

            if (code not in out) or (len(base) < len(os.path.basename(out[code]))):
                out[code] = fp

        return out

    def _find_file(self, tp_dir: str, code: str) -> str:
        """
        Find Excel file for a sample code in a given timepoint directory.
        Primary: lookup in indexed map (no suffix assumptions).
        Fallback: legacy substring scan.
        """
        code = self._normalize_code(code)

        m = self._tp_maps.get(tp_dir, {})
        if code in m:
            return m[code]

        # Fallback (legacy behavior)
        code_lower = code.lower().replace('.', '_').replace('-', '_')
        for f in os.listdir(tp_dir):
            if f.endswith('.xlsx') and not f.startswith('~'):
                f_lower = f.lower().replace('.', '_').replace('-', '_')
                if code_lower in f_lower:
                    return os.path.join(tp_dir, f)

        raise FileNotFoundError(f"Could not find file for {code} in {tp_dir}")

    def load_excel_with_axes(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load Excel and return data with spectral axes."""
        if filepath in self._cache:
            return self._cache[filepath]

        wb = load_workbook(filepath, data_only=True, read_only=True)
        ws = wb.active

        excitation_axis: List[float] = []
        emission_axis: List[float] = []
        data_rows: List[List[float]] = []

        for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
            if row is None:
                continue

            # Header row: excitation in columns 1..end
            if row_idx == 0:
                for h in row[1:]:
                    if h is None:
                        continue
                    s = str(h).replace('Excitation_', '').replace('nm', '').strip()
                    try:
                        excitation_axis.append(float(s))
                    except:
                        pass
                continue

            # Data rows: emission in col0, intensities in col1..colN
            if row[0] is None:
                continue
            try:
                em = float(row[0])
            except:
                continue

            emission_axis.append(em)

            row_data: List[float] = []
            # Use the number of excitation values we parsed to keep consistent width.
            # If excitation_axis is empty, fall back to "all remaining columns".
            n_ex = len(excitation_axis)
            vals = row[1:(1 + n_ex)] if n_ex > 0 else row[1:]

            for cell in vals:
                if isinstance(cell, (int, float)) and cell is not None:
                    row_data.append(float(cell))
                else:
                    row_data.append(0.0)

            if len(row_data) > 0:
                data_rows.append(row_data)

        excitation_axis_arr = np.array(excitation_axis, dtype=np.float32)
        emission_axis_arr = np.array(emission_axis, dtype=np.float32)
        data = np.array(data_rows, dtype=np.float32)

        if data.size == 0:
            raise ValueError(f"No valid data found in {filepath}")

        # Normalize to [0, 1]
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        else:
            data = np.zeros_like(data)

        self._cache[filepath] = (data, emission_axis_arr, excitation_axis_arr)
        return data, emission_axis_arr, excitation_axis_arr

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]

        timepoint_images = []
        emission_axis = None
        excitation_axis = None

        for tp_dir in self.timepoint_dirs:
            filepath = self._find_file(tp_dir, code)
            data, em_ax, ex_ax = self.load_excel_with_axes(filepath)

            if emission_axis is None:
                emission_axis = em_ax
                excitation_axis = ex_ax

            # Add channel dimension: (H, W) -> (1, H, W)
            if len(data.shape) == 2:
                data = data[np.newaxis, :, :]

            timepoint_images.append(torch.FloatTensor(data))

        # Stack timepoints: (3, 1, H, W)
        images = torch.stack(timepoint_images)

        return images, torch.tensor(label, dtype=torch.long), (emission_axis, excitation_axis)



# ==============================================================================
# MODEL COMPONENTS
# ==============================================================================

class ChiralitySpatialAttention(nn.Module):
    """
    Spatial attention module guided by chirality positions.
    
    Combines learned attention with chirality prior to focus on relevant regions.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Learned attention map
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable weight for combining with prior
        self.prior_weight = nn.Parameter(torch.tensor(0.5))
        
        # Will store chirality prior
        self.register_buffer('chirality_prior', None)
    
    def set_chirality_prior(self, prior: torch.Tensor):
        """Set the chirality attention prior."""
        self.chirality_prior = prior
    
    def forward(self, x):
        """
        Apply spatial attention.
        
        Args:
            x: Feature maps (batch, channels, H, W)
        Returns:
            Attention-weighted features (batch, channels, H, W)
            Attention map (batch, 1, H, W)
        """
        # Learned attention
        learned_attn = self.conv(x)  # (batch, 1, H, W)
        
        if self.chirality_prior is not None:
            # Resize prior to match feature map size
            prior = self.chirality_prior.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            if prior.shape[-2:] != x.shape[-2:]:
                prior = nn.functional.interpolate(
                    prior, size=x.shape[-2:], mode='bilinear', align_corners=False
                )
            prior = prior.expand(x.shape[0], -1, -1, -1)
            
            # Combine learned attention with prior
            w = torch.sigmoid(self.prior_weight)
            attention = w * learned_attn + (1 - w) * prior
        else:
            attention = learned_attn
        
        # Apply attention
        out = x * attention
        
        return out, attention


class ConvEncoderWithChiralityAttention(nn.Module):
    """
    Shallower encoder (2 conv blocks) with chirality spatial attention.
    """
    
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Spatial attention after first block
        self.spatial_attention = ChiralitySpatialAttention(32)
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Third conv block (no pooling)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.flatten_size = None
        self.fc = None
        self.latent_dim = latent_dim
        self.encoded_shape = None
        self.input_shape = None
        
    def set_chirality_prior(self, prior: torch.Tensor):
        """Set chirality prior for spatial attention."""
        self.spatial_attention.set_chirality_prior(prior)
    
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = (x.shape[2], x.shape[3])
        
        # First conv + attention
        x = self.conv1(x)
        x, attn_map = self.spatial_attention(x)
        
        # Remaining convs
        x = self.conv2(x)
        x = self.conv3(x)
        
        if self.encoded_shape is None:
            self.encoded_shape = (x.shape[1], x.shape[2], x.shape[3])
        
        if self.fc is None:
            self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
        
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        
        return z, attn_map


class ConvDecoder(nn.Module):
    """Decoder to reconstruct from latent space."""
    
    def __init__(self, latent_dim: int, out_channels: int, 
                 encoded_shape: tuple, target_shape: tuple):
        super().__init__()
        
        self.encoded_channels, self.encoded_h, self.encoded_w = encoded_shape
        self.target_h, self.target_w = target_shape
        
        self.fc = nn.Linear(latent_dim, self.encoded_channels * self.encoded_h * self.encoded_w)
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.encoded_channels, self.encoded_h, self.encoded_w)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.final_conv(x)
        
        # Resize to exact target dimensions
        if x.shape[2] != self.target_h or x.shape[3] != self.target_w:
            x = nn.functional.interpolate(
                x, size=(self.target_h, self.target_w),
                mode='bilinear', align_corners=False
            )
        
        x = self.sigmoid(x)
        return x


class TemporalAttention(nn.Module):
    """Attention over timepoints."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, latent_vectors):
        """
        Args:
            latent_vectors: (batch, num_timepoints, latent_dim)
        Returns:
            aggregated: (batch, latent_dim)
            weights: (batch, num_timepoints)
        """
        scores = self.attention_net(latent_vectors).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        aggregated = (latent_vectors * weights.unsqueeze(-1)).sum(dim=1)
        return aggregated, weights


class Classifier(nn.Module):
    """Classification head."""
    
    def __init__(self, latent_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, z):
        return self.classifier(z)


class ChiralityInformedAutoencoder(nn.Module):
    """
    Complete chirality-informed autoencoder with temporal attention.
    
    Architecture:
        Input: (batch, 3, 1, H, W) - 3 timepoints
              ↓
        Encoder with Chirality Spatial Attention (shared)
              ↓
        Latent vectors: (batch, 3, latent_dim)
              ↓
        Temporal Attention → aggregated latent
              ↓
        Classifier → ALS/CTRL
        
        Also: Decoder for chirality-weighted reconstruction
    """
    
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, num_classes: int = 2):
        super().__init__()
        
        self.encoder = ConvEncoderWithChiralityAttention(in_channels, latent_dim)
        self.temporal_attention = TemporalAttention(latent_dim)
        self.classifier = Classifier(latent_dim, num_classes)
        self.decoder = None
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Chirality mask for weighted reconstruction loss
        self.register_buffer('chirality_mask', None)
        
    def set_chirality_info(self, emission_axis: np.ndarray, excitation_axis: np.ndarray):
        """Set chirality mask and attention prior based on spectral axes."""
        # Create mask for reconstruction loss
        mask = create_chirality_mask(emission_axis, excitation_axis)
        self.chirality_mask = torch.FloatTensor(mask)
        
        # Create attention prior
        prior = create_chirality_attention_prior(emission_axis, excitation_axis)
        prior_tensor = torch.FloatTensor(prior)
        self.encoder.set_chirality_prior(prior_tensor)
    
    def forward(self, timepoint_images):
        """
        Forward pass.
        
        Args:
            timepoint_images: (batch, 3, 1, H, W)
        
        Returns:
            reconstructions: (batch, 3, 1, H, W)
            class_logits: (batch, num_classes)
            temporal_weights: (batch, 3)
            spatial_attention_maps: (batch, 3, 1, H', W')
        """
        batch_size = timepoint_images.shape[0]
        num_timepoints = timepoint_images.shape[1]
        
        # Encode each timepoint
        latent_vectors = []
        attention_maps = []
        
        for t in range(num_timepoints):
            z_t, attn_t = self.encoder(timepoint_images[:, t])
            latent_vectors.append(z_t)
            attention_maps.append(attn_t)
        
        # Initialize decoder if needed
        if self.decoder is None and self.encoder.encoded_shape is not None:
            self.decoder = ConvDecoder(
                self.latent_dim,
                self.in_channels,
                self.encoder.encoded_shape,
                self.encoder.input_shape
            ).to(timepoint_images.device)
        
        # Stack latents: (batch, 3, latent_dim)
        latent_vectors = torch.stack(latent_vectors, dim=1)
        attention_maps = torch.stack(attention_maps, dim=1)  # (batch, 3, 1, H', W')
        
        # Temporal attention for classification
        z_agg, temporal_weights = self.temporal_attention(latent_vectors)
        
        # Classify
        class_logits = self.classifier(z_agg)
        
        # Reconstruct
        reconstructions = []
        for t in range(num_timepoints):
            recon_t = self.decoder(latent_vectors[:, t])
            reconstructions.append(recon_t)
        reconstructions = torch.stack(reconstructions, dim=1)
        
        return reconstructions, class_logits, temporal_weights, attention_maps, latent_vectors
    
    def chirality_weighted_recon_loss(self, recon, target):
        """
        Compute reconstruction loss weighted by chirality mask.
        
        Chirality regions contribute more to the loss.
        """
        if self.chirality_mask is None:
            return nn.functional.mse_loss(recon, target)
        
        # Expand mask to match batch dimensions
        mask = self.chirality_mask.to(recon.device)
        
        # Handle dimension matching
        while mask.dim() < recon.dim():
            mask = mask.unsqueeze(0)
        
        # Expand to batch size
        mask = mask.expand_as(recon)
        
        # Weighted MSE
        diff_sq = (recon - target) ** 2
        weighted_loss = (diff_sq * mask).mean()
        
        return weighted_loss


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device, alpha=1.0, beta=1.0, use_chirality_loss=True):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0
    
    classification_criterion = nn.CrossEntropyLoss()
    
    all_temporal_weights = []
    
    for batch in dataloader:
        images, labels, axes_info = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # Set chirality info on first batch
        if model.chirality_mask is None:
            em_ax, ex_ax = axes_info
            model.set_chirality_info(em_ax[0].numpy(), ex_ax[0].numpy())
            if model.chirality_mask is not None:
                model.chirality_mask = model.chirality_mask.to(device)
        
        optimizer.zero_grad()
        
        recon, logits, temp_weights, spatial_attn, latents = model(images)
        
        # Losses
        if use_chirality_loss:
            recon_loss = model.chirality_weighted_recon_loss(recon, images)
        else:
            recon_loss = nn.functional.mse_loss(recon, images)
        
        class_loss = classification_criterion(logits, labels)
        loss = alpha * recon_loss + beta * class_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()
        
        _, pred = logits.max(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
        all_temporal_weights.append(temp_weights.detach().cpu().numpy())
    
    n = len(dataloader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    
    temporal_weights = np.concatenate(all_temporal_weights, axis=0)
    
    return total_loss/n, total_recon_loss/n, total_class_loss/n, acc, temporal_weights


def validate(model, dataloader, device, alpha=1.0, beta=1.0, use_chirality_loss=True):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0
    
    all_probs = []
    all_labels = []
    all_temporal_weights = []
    all_latents = []
    
    classification_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels, axes_info = batch
            images = images.to(device)
            labels = labels.to(device)
            
            recon, logits, temp_weights, spatial_attn, latents = model(images)
            
            # Losses
            if use_chirality_loss and model.chirality_mask is not None:
                recon_loss = model.chirality_weighted_recon_loss(recon, images)
            else:
                recon_loss = nn.functional.mse_loss(recon, images)
            
            class_loss = classification_criterion(logits, labels)
            loss = alpha * recon_loss + beta * class_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            all_temporal_weights.append(temp_weights.cpu().numpy())
            all_latents.append(latents.cpu().numpy())
    
    n = len(dataloader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    temporal_weights = np.concatenate(all_temporal_weights, axis=0)
    latents = np.concatenate(all_latents, axis=0)
    
    return total_loss/n, total_recon_loss/n, total_class_loss/n, acc, auc, temporal_weights, latents


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Chirality-Informed Autoencoder')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing Excel EEM files')
    parser.add_argument('--labels_csv', type=str, required=True,
                       help='CSV with columns: code, group')
    parser.add_argument('--output_dir', type=str, default='./chirality_autoencoder_results')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--use_chirality_loss', action='store_true', default=True,
                       help='Use chirality-weighted reconstruction loss')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
        # Find timepoint directories (robust: folder-based, no filename suffix required)
    def _has_xlsx(folder: str) -> bool:
        try:
            return any(
                f.lower().endswith('.xlsx') and not f.startswith('~$') and not f.startswith('~')
                for f in os.listdir(folder)
            )
        except Exception:
            return False

    data_dir = os.path.normpath(args.data_dir)
    bn = os.path.basename(data_dir).lower()
    if bn in ('out_0h', 'out_6h', 'out_24h', '0h', '6h', '24h'):
        base_dir = os.path.dirname(data_dir)
    else:
        base_dir = data_dir

    tp_suffixes = ['0h', '6h', '24h']
    timepoint_dirs = []

    for suffix in tp_suffixes:
        candidates = [
            os.path.join(base_dir, f'out_{suffix}'),
            os.path.join(base_dir, f'out-{suffix}'),
            os.path.join(base_dir, suffix),
        ]
        chosen = None

        # Prefer explicit timepoint folders
        for cand in candidates:
            if os.path.isdir(cand) and _has_xlsx(cand):
                chosen = cand
                break

        # Fallback: allow all-in-one directory
        if chosen is None and os.path.isdir(base_dir) and _has_xlsx(base_dir):
            chosen = base_dir

        if chosen is None:
            raise FileNotFoundError(
                f"Could not locate timepoint folder for {suffix}. "
                f"Expected e.g. {os.path.join(base_dir, f'out_{suffix}')}"
            )

        timepoint_dirs.append(chosen)

    print(f"Timepoint directories: {timepoint_dirs}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    if 'sample_id' in labels_df.columns and 'code' not in labels_df.columns:
        labels_df['code'] = labels_df['sample_id']
    
    codes = labels_df['code'].astype(str).tolist()
    groups = labels_df['group'].tolist()
    labels = np.array([1 if g == 'ALS' else 0 for g in groups])
    
    print(f"\nLoaded {len(codes)} samples")
    print(f"  ALS: {sum(labels)}, CTRL: {len(labels) - sum(labels)}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    all_latents_collected = []
    all_labels_collected = []
    all_temporal_weights = []
    
    print(f"\n{'='*70}")
    print(f"CHIRALITY-INFORMED AUTOENCODER - {args.n_folds}-FOLD CV")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(codes, labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{args.n_folds}")
        print(f"{'='*70}")
        
        # Split
        train_codes = [codes[i] for i in train_idx]
        train_labels = labels[train_idx].tolist()
        val_codes = [codes[i] for i in val_idx]
        val_labels = labels[val_idx].tolist()
        
        print(f"  Train: {len(train_codes)} ({sum(train_labels)} ALS)")
        print(f"  Val: {len(val_codes)} ({sum(val_labels)} ALS)")
        
        # Datasets
        train_dataset = ExcelImageDataset(train_codes, train_labels, timepoint_dirs)
        val_dataset = ExcelImageDataset(val_codes, val_labels, timepoint_dirs)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Model
        model = ChiralityInformedAutoencoder(
            in_channels=1,
            latent_dim=args.latent_dim,
            num_classes=2
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Training
        best_auc = 0
        best_state = None
        patience_counter = 0
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
        
        for epoch in range(args.epochs):
            train_loss, train_recon, train_class, train_acc, _ = train_epoch(
                model, train_loader, optimizer, device, args.alpha, args.beta, args.use_chirality_loss
            )
            
            val_loss, val_recon, val_class, val_acc, val_auc, temp_weights, latents = validate(
                model, val_loader, device, args.alpha, args.beta, args.use_chirality_loss
            )
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
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
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_fold{fold+1}.pth'))
        
        # Final validation
        _, _, _, val_acc, val_auc, temp_weights, latents = validate(
            model, val_loader, device, args.alpha, args.beta, args.use_chirality_loss
        )
        
        fold_results.append({
            'fold': fold + 1,
            'val_acc': val_acc,
            'val_auc': val_auc,
        })
        
        all_latents_collected.append(latents)
        all_labels_collected.extend(val_labels)
        all_temporal_weights.append(temp_weights)
        
        print(f"  Final: acc={val_acc:.1f}%, AUC={val_auc:.3f}")
        
        # Plot training history
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        
        axes[1].plot(history['train_acc'], label='Train')
        axes[1].plot(history['val_acc'], label='Val')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        
        axes[2].plot(history['val_auc'])
        axes[2].set_title('Val AUC')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'training_fold{fold+1}.png'), dpi=100)
        plt.close()
    
    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    avg_acc = np.mean([r['val_acc'] for r in fold_results])
    std_acc = np.std([r['val_acc'] for r in fold_results])
    avg_auc = np.mean([r['val_auc'] for r in fold_results])
    std_auc = np.std([r['val_auc'] for r in fold_results])
    
    print(f"\nCross-validation results:")
    print(f"  Accuracy: {avg_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  AUC: {avg_auc:.3f} ± {std_auc:.3f}")
    
    # Temporal attention analysis
    all_temp_w = np.vstack(all_temporal_weights)
    mean_temp_w = all_temp_w.mean(axis=0)
    print(f"\nTemporal attention weights (avg):")
    print(f"  0h: {mean_temp_w[0]:.3f}")
    print(f"  6h: {mean_temp_w[1]:.3f}")
    print(f"  24h: {mean_temp_w[2]:.3f}")
    
    # Save results
    summary = {
        'config': vars(args),
        'fold_results': fold_results,
        'avg_accuracy': float(avg_acc),
        'std_accuracy': float(std_acc),
        'avg_auc': float(avg_auc),
        'std_auc': float(std_auc),
        'temporal_weights': {
            '0h': float(mean_temp_w[0]),
            '6h': float(mean_temp_w[1]),
            '24h': float(mean_temp_w[2]),
        }
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save latents for downstream ML
    all_latents = np.vstack(all_latents_collected)
    all_labels_arr = np.array(all_labels_collected)
    
    np.savez(
        os.path.join(args.output_dir, 'extracted_features.npz'),
        latents=all_latents,
        labels=all_labels_arr,
        temporal_weights=all_temp_w,
    )
    
    # Plot temporal attention
    fig, ax = plt.subplots(figsize=(8, 5))
    tp_names = ['0h', '6h', '24h']
    ax.bar(tp_names, mean_temp_w, yerr=all_temp_w.std(axis=0), capsize=5,
           color=['steelblue', 'coral', 'seagreen'], edgecolor='black')
    ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Temporal Attention Weights')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'temporal_attention.png'), dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {args.output_dir}")
    
    # Final summary text
    summary_text = f"""
CHIRALITY-INFORMED AUTOENCODER RESULTS
{'='*50}

Configuration:
  Latent dim: {args.latent_dim}
  Epochs: {args.epochs}
  Learning rate: {args.lr}
  Alpha (recon): {args.alpha}
  Beta (class): {args.beta}
  Chirality-weighted loss: {args.use_chirality_loss}

Cross-validation ({args.n_folds} folds):
  Accuracy: {avg_acc:.1f}% ± {std_acc:.1f}%
  AUC: {avg_auc:.3f} ± {std_auc:.3f}

Temporal attention weights:
  0h:  {mean_temp_w[0]:.3f}
  6h:  {mean_temp_w[1]:.3f}
  24h: {mean_temp_w[2]:.3f}

Output files:
  - model_fold*.pth: Trained models
  - extracted_features.npz: Latent features for downstream ML
  - training_fold*.png: Training curves
  - temporal_attention.png: Attention visualization
  - summary.json: Full results
"""
    
    print(summary_text)
    
    with open(os.path.join(args.output_dir, 'SUMMARY.txt'), 'w') as f:
        f.write(summary_text)


if __name__ == '__main__':
    main()
