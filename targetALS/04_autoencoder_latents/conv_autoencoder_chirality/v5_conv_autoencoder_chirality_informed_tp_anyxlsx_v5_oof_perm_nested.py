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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
import os
import glob
import re
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json

import copy
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

        return images, torch.tensor(label, dtype=torch.long), (emission_axis, excitation_axis), code



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
        if len(batch) == 4:
            images, labels, axes_info, _codes = batch
        else:
            images, labels, axes_info = batch
            _codes = None
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
    """Validate model. Returns losses/metrics plus per-sample outputs."""
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []
    all_codes = []
    all_temporal_weights = []
    all_latents = []

    classification_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, labels, axes_info, codes = batch
            else:
                images, labels, axes_info = batch
                codes = None

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
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            if codes is not None:
                # codes is a list of strings from the Dataset
                all_codes.extend([str(c) for c in codes])
            else:
                all_codes.extend([None] * labels.size(0))

            _, pred = logits.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            all_temporal_weights.append(temp_weights.detach().cpu().numpy())
            all_latents.append(latents.detach().cpu().numpy())

    n = len(dataloader) if len(dataloader) > 0 else 1
    acc = 100.0 * correct / total if total > 0 else 0.0

    try:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    except Exception:
        auc = 0.5

    try:
        ap = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    except Exception:
        ap = 0.5

    temporal_weights = np.concatenate(all_temporal_weights, axis=0) if len(all_temporal_weights) else np.zeros((0, 3))
    latents = np.concatenate(all_latents, axis=0) if len(all_latents) else np.zeros((0, getattr(model, "latent_dim", 0)))

    return (total_loss / n,
            total_recon_loss / n,
            total_class_loss / n,
            acc,
            auc,
            ap,
            temporal_weights,
            latents,
            np.array(all_probs, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_codes, dtype=object))


# ==============================================================================
# MAIN
# ==============================================================================

# ==============================================================================
# EVALUATION HELPERS
# ==============================================================================

def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximizing Youden's J (TPR - FPR)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = int(np.nanargmax(j))
    # roc_curve returns thresholds in descending order; ok
    t = float(thr[best])
    # guard: sometimes inf at start
    if not np.isfinite(t):
        t = 0.5
    return t

def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute a small metrics panel at a given threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal_acc = np.nanmean([sens, spec])

    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "threshold": float(thr),
        "accuracy": float(acc) if np.isfinite(acc) else np.nan,
        "balanced_accuracy": float(bal_acc) if np.isfinite(bal_acc) else np.nan,
        "sensitivity": float(sens) if np.isfinite(sens) else np.nan,
        "specificity": float(spec) if np.isfinite(spec) else np.nan,
        "precision": float(prec) if np.isfinite(prec) else np.nan,
        "npv": float(npv) if np.isfinite(npv) else np.nan,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, seed: int = 123) -> Dict[str, float]:
    """Nonparametric bootstrap CI for AUC using OOF predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return {"auc_ci_low": np.nan, "auc_ci_med": np.nan, "auc_ci_high": np.nan, "n_boot_used": 0}

    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            continue

    if len(aucs) == 0:
        return {"auc_ci_low": np.nan, "auc_ci_med": np.nan, "auc_ci_high": np.nan, "n_boot_used": 0}

    lo, med, hi = np.percentile(aucs, [2.5, 50, 97.5])
    return {"auc_ci_low": float(lo), "auc_ci_med": float(med), "auc_ci_high": float(hi), "n_boot_used": int(len(aucs))}

def plot_oof_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_png: str):
    """Save ROC + PR curves based on OOF predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OOF ROC Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_png = os.path.splitext(out_png)[0] + "_pr.png"
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("OOF Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(pr_png, dpi=120)
    plt.close()

def run_cv_evaluation(
    codes: np.ndarray,
    y: np.ndarray,
    timepoint_dirs: List[str],
    args,
    run_tag: str,
    save_artifacts: bool = True,
    override_epochs: Optional[int] = None,
    override_patience: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run (repeated) CV and return a dict with:
      - oof_df (DataFrame-like list of dicts)
      - fold_results (list)
      - summary (dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    k_folds = args.k_folds
    repeats = args.repeats
    seed = args.seed
    nested = not args.no_nested
    inner_frac = args.inner_val_frac

    epochs = int(override_epochs) if override_epochs is not None else int(args.epochs)
    patience = int(override_patience) if override_patience is not None else int(args.patience)

    fold_results = []
    oof_records = []
    temporal_weights_all = []

    # For downstream: latents aligned to OOF order
    latents_records = []

    for rep in range(repeats):
        cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed + rep)

        for fold, (train_idx, test_idx) in enumerate(cv.split(codes, y), start=1):
            print(f"\n{'='*70}")
            print(f"REPEAT {rep+1}/{repeats} - FOLD {fold}/{k_folds}")
            print(f"{'='*70}")

            train_codes = codes[train_idx]
            train_y = y[train_idx]
            test_codes = codes[test_idx]
            test_y = y[test_idx]

            if nested:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=inner_frac, random_state=seed + rep * 100 + fold)
                inner_train_sub, inner_val_sub = next(splitter.split(train_codes, train_y))
                inner_train_codes = train_codes[inner_train_sub]
                inner_train_y = train_y[inner_train_sub]
                inner_val_codes = train_codes[inner_val_sub]
                inner_val_y = train_y[inner_val_sub]

                early_train_dataset = ExcelImageDataset(inner_train_codes, inner_train_y, timepoint_dirs)
                early_val_dataset = ExcelImageDataset(inner_val_codes, inner_val_y, timepoint_dirs)

                train_loader = DataLoader(early_train_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(early_val_dataset, batch_size=args.batch_size, shuffle=False)

                print(f"  Inner-train: {len(inner_train_codes)} ({int(inner_train_y.sum())} ALS)")
                print(f"  Inner-val:   {len(inner_val_codes)} ({int(inner_val_y.sum())} ALS)")
                print(f"  Outer-test:  {len(test_codes)} ({int(test_y.sum())} ALS)")
            else:
                train_dataset = ExcelImageDataset(train_codes, train_y, timepoint_dirs)
                test_dataset = ExcelImageDataset(test_codes, test_y, timepoint_dirs)

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                print(f"  Train: {len(train_codes)} ({int(train_y.sum())} ALS)")
                print(f"  Val:   {len(test_codes)} ({int(test_y.sum())} ALS)  [NOTE: val used for early stopping]")

            # Model
            model = ChiralityInformedAutoencoder(
                in_channels=1,
                latent_dim=args.latent_dim,
                num_classes=2
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            best_auc = -1.0
            best_state = None
            patience_counter = 0

            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

            for epoch in range(epochs):
                train_loss, train_recon, train_class, train_acc, _ = train_epoch(
                    model, train_loader, optimizer, device, args.alpha, args.beta, args.use_chirality_loss
                )

                val_out = validate(model, val_loader, device, args.alpha, args.beta, args.use_chirality_loss)
                val_loss, val_recon, val_class, val_acc, val_auc, val_ap, _, _, _, _, _ = val_out

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['val_auc'].append(val_auc)

                if val_auc > best_auc:
                    best_auc = float(val_auc)
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, val_auc={val_auc:.3f}, val_ap={val_ap:.3f}")

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            # Evaluate on OUTER test
            test_dataset = ExcelImageDataset(test_codes, test_y, timepoint_dirs)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            test_out = validate(model, test_loader, device, args.alpha, args.beta, args.use_chirality_loss)
            test_loss, test_recon, test_class, test_acc, test_auc, test_ap, temp_w, latents, probs, y_true, out_codes = test_out

            # Threshold selection (no leakage into outer test)
            if nested:
                # Use inner-val to pick threshold
                inner_val_dataset = ExcelImageDataset(inner_val_codes, inner_val_y, timepoint_dirs)
                inner_val_loader = DataLoader(inner_val_dataset, batch_size=args.batch_size, shuffle=False)
                _, _, _, _, _, _, _, _, inner_probs, inner_true, _ = validate(
                    model, inner_val_loader, device, args.alpha, args.beta, args.use_chirality_loss
                )
                thr_you = youden_threshold(inner_true, inner_probs)
            else:
                # Use training set to pick threshold
                train_dataset = ExcelImageDataset(train_codes, train_y, timepoint_dirs)
                train_loader_eval = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
                _, _, _, _, _, _, _, _, train_probs, train_true, _ = validate(
                    model, train_loader_eval, device, args.alpha, args.beta, args.use_chirality_loss
                )
                thr_you = youden_threshold(train_true, train_probs)

            m05 = metrics_at_threshold(y_true, probs, 0.5)
            myou = metrics_at_threshold(y_true, probs, thr_you)

            fold_id = f"r{rep+1}_f{fold}"
            fold_results.append({
                "fold_id": fold_id,
                "repeat": rep + 1,
                "fold": fold,
                "n_test": int(len(test_codes)),
                "test_acc_pct": float(test_acc),
                "test_auc": float(test_auc),
                "test_ap": float(test_ap),
                "thr_youden": float(thr_you),
                "acc_05": m05["accuracy"],
                "bal_acc_05": m05["balanced_accuracy"],
                "sens_05": m05["sensitivity"],
                "spec_05": m05["specificity"],
                "acc_you": myou["accuracy"],
                "bal_acc_you": myou["balanced_accuracy"],
                "sens_you": myou["sensitivity"],
                "spec_you": myou["specificity"],
            })

            print(f"  Outer-test: AUC={test_auc:.3f}, AP={test_ap:.3f}, acc@0.5={m05['accuracy']:.3f}, bal@Youden={myou['balanced_accuracy']:.3f}")

            temporal_weights_all.append(temp_w)

            # Save artifacts
            if save_artifacts:
                # models/plots per fold
                model_path = os.path.join(args.output_dir, f"model_{run_tag}_{fold_id}.pth")
                torch.save(model.state_dict(), model_path)

                # Training curve
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
                plt.savefig(os.path.join(args.output_dir, f"training_{run_tag}_{fold_id}.png"), dpi=120)
                plt.close()

            # OOF records
            for c, yt, pr in zip(out_codes, y_true, probs):
                c_str = str(c)
                oof_records.append({
                    "run_tag": run_tag,
                    "repeat": rep + 1,
                    "fold": fold,
                    "fold_id": fold_id,
                    "code": c_str,
                    "y_true": int(yt),
                    "prob_pos": float(pr),
                    "pred_05": int(pr >= 0.5),
                    "pred_you": int(pr >= thr_you),
                    "thr_you": float(thr_you),
                })

            # Latents aligned to OOF
            for c, yt, pr, z in zip(out_codes, y_true, probs, latents):
                latents_records.append({
                    "code": str(c),
                    "y_true": int(yt),
                    "prob_pos": float(pr),
                    "fold_id": fold_id,
                    "latent": z.astype(np.float32),
                })

    # Aggregate summary
    oof_y = np.array([r["y_true"] for r in oof_records], dtype=int)
    oof_p = np.array([r["prob_pos"] for r in oof_records], dtype=float)

    oof_auc = roc_auc_score(oof_y, oof_p) if len(np.unique(oof_y)) > 1 else 0.5
    oof_ap = average_precision_score(oof_y, oof_p) if len(np.unique(oof_y)) > 1 else 0.5

    # Use per-sample predictions at fixed threshold 0.5 and per-fold Youden threshold
    oof_pred05 = np.array([r["pred_05"] for r in oof_records], dtype=int)
    oof_predyou = np.array([r["pred_you"] for r in oof_records], dtype=int)

    # Metrics computed from OOF predictions
    m05_global = metrics_at_threshold(oof_y, oof_p, 0.5)
    # For "Youden", use already-chosen per-fold thresholds → compute confusion directly:
    cm_you = confusion_matrix(oof_y, oof_predyou, labels=[0, 1])
    tn, fp, fn, tp = cm_you.ravel()
    acc_you = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    sens_you = tp / (tp + fn) if (tp + fn) else np.nan
    spec_you = tn / (tn + fp) if (tn + fp) else np.nan
    bal_you = np.nanmean([sens_you, spec_you])

    ci = bootstrap_auc_ci(oof_y, oof_p, n_boot=args.bootstrap, seed=seed + 999)

    # Temporal attention weights
    if len(temporal_weights_all):
        tw = np.concatenate(temporal_weights_all, axis=0)
        tw_mean = tw.mean(axis=0).tolist()
    else:
        tw_mean = [np.nan, np.nan, np.nan]

    summary = {
        "run_tag": run_tag,
        "nested": nested,
        "k_folds": k_folds,
        "repeats": repeats,
        "n_samples": int(len(codes)),
        "oof_auc": float(oof_auc),
        "oof_ap": float(oof_ap),
        "oof_acc_05": m05_global["accuracy"],
        "oof_bal_acc_05": m05_global["balanced_accuracy"],
        "oof_sens_05": m05_global["sensitivity"],
        "oof_spec_05": m05_global["specificity"],
        "oof_acc_you": float(acc_you) if np.isfinite(acc_you) else np.nan,
        "oof_bal_acc_you": float(bal_you) if np.isfinite(bal_you) else np.nan,
        "oof_sens_you": float(sens_you) if np.isfinite(sens_you) else np.nan,
        "oof_spec_you": float(spec_you) if np.isfinite(spec_you) else np.nan,
        "auc_ci_low": ci["auc_ci_low"],
        "auc_ci_med": ci["auc_ci_med"],
        "auc_ci_high": ci["auc_ci_high"],
        "n_boot_used": ci["n_boot_used"],
        "temporal_attention_mean": tw_mean,
    }

    return {
        "oof_records": oof_records,
        "fold_results": fold_results,
        "summary": summary,
        "latents_records": latents_records,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Chirality-Informed Autoencoder (OOF eval + permutation test)')

    parser.add_argument('--data_dir', type=str, required=True, help='Parent dir containing out_0h/out_6h/out_24h')
    parser.add_argument('--labels_csv', type=str, required=True, help='CSV with columns: code, group')
    parser.add_argument('--output_dir', type=str, default='./chirality_autoencoder_results')

    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--use_chirality_loss', action='store_true', default=True)

    # CV / evaluation
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_nested', action='store_true', help='Disable nested CV (NOT recommended).')
    parser.add_argument('--inner_val_frac', type=float, default=0.2, help='Inner validation fraction for nested CV.')
    parser.add_argument('--bootstrap', type=int, default=2000, help='Bootstrap replicates for OOF AUC CI.')

    # Permutation test (optional; can be slow)
    parser.add_argument('--permute', type=int, default=0, help='Number of label permutations (0 disables).')
    parser.add_argument('--perm_seed', type=int, default=123)
    parser.add_argument('--perm_epochs', type=int, default=40, help='Epochs for permutation runs.')
    parser.add_argument('--perm_patience', type=int, default=10, help='Patience for permutation runs.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Timepoint folders
    # --------------------------------------------------------------------------
    base_dir = os.path.normpath(args.data_dir)
    tp_dirs = [os.path.join(base_dir, "out_0h"),
               os.path.join(base_dir, "out_6h"),
               os.path.join(base_dir, "out_24h")]

    for d in tp_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing timepoint folder: {d}")

    print(f"Timepoint directories: {tp_dirs}")

    # --------------------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------------------
    labels_df = pd.read_csv(args.labels_csv)

    # robust column picking
    cols = [c.lower() for c in labels_df.columns]
    if "code" not in cols:
        raise ValueError("labels_csv must contain a 'code' column")
    if "group" not in cols:
        raise ValueError("labels_csv must contain a 'group' column")

    # map original columns
    code_col = labels_df.columns[cols.index("code")]
    group_col = labels_df.columns[cols.index("group")]

    labels_df[code_col] = labels_df[code_col].astype(str)
    codes = labels_df[code_col].values

    le = LabelEncoder()
    y = le.fit_transform(labels_df[group_col].astype(str).values)

    print(f"\nLoaded {len(codes)} samples")
    if set(le.classes_) >= set(["ALS", "CTRL"]) or set(le.classes_) >= set(["CTRL", "ALS"]):
        # try to print ALS count if ALS exists
        als_label = int(np.where(le.classes_ == "ALS")[0][0]) if "ALS" in le.classes_ else 1
        print(f"  ALS: {int((y==als_label).sum())}, CTRL: {int((y!=als_label).sum())}")
    else:
        # generic
        print("  Class balance:", dict(zip(le.classes_, np.bincount(y))))

    # Ensure binary labels are 0/1 with "ALS" as 1 when present
    if "ALS" in le.classes_ and "CTRL" in le.classes_:
        als_idx = int(np.where(le.classes_ == "ALS")[0][0])
        y = (y == als_idx).astype(int)

    # --------------------------------------------------------------------------
    # Main CV run
    # --------------------------------------------------------------------------
    run_tag = "main"
    res = run_cv_evaluation(codes=codes, y=y, timepoint_dirs=tp_dirs, args=args, run_tag=run_tag, save_artifacts=True)

    oof_records = res["oof_records"]
    fold_results = res["fold_results"]
    summary = res["summary"]
    latents_records = res["latents_records"]

    # Save OOF predictions
    oof_path = os.path.join(args.output_dir, "oof_predictions.csv")
    pd.DataFrame(oof_records).to_csv(oof_path, index=False)

    folds_path = os.path.join(args.output_dir, "fold_metrics.csv")
    pd.DataFrame(fold_results).to_csv(folds_path, index=False)

    # Save latents in a compact NPZ aligned to oof order
    # order by appearance in latents_records
    lat_codes = [r["code"] for r in latents_records]
    lat_y = np.array([r["y_true"] for r in latents_records], dtype=np.int64)
    lat_prob = np.array([r["prob_pos"] for r in latents_records], dtype=np.float32)
    lat_fold = np.array([r["fold_id"] for r in latents_records], dtype=object)
    Z = np.stack([r["latent"] for r in latents_records], axis=0)

    np.savez_compressed(
        os.path.join(args.output_dir, "extracted_features_oof.npz"),
        codes=np.array(lat_codes, dtype=object),
        labels=lat_y,
        prob_pos=lat_prob,
        fold_id=lat_fold,
        latents=Z,
    )

    # Curves
    try:
        plot_oof_roc_pr(
            y_true=np.array([r["y_true"] for r in oof_records]),
            y_prob=np.array([r["prob_pos"] for r in oof_records]),
            out_png=os.path.join(args.output_dir, "oof_roc.png"),
        )
    except Exception as e:
        print("Could not plot OOF curves:", e)

    # Save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("OOF SUMMARY")
    print(f"{'='*70}")
    print(f"Nested CV: {summary['nested']} (disable with --no_nested)")
    print(f"OOF AUC: {summary['oof_auc']:.3f} (95% bootstrap CI {summary['auc_ci_low']:.3f}–{summary['auc_ci_high']:.3f}, n_boot_used={summary['n_boot_used']})")
    print(f"OOF AP:  {summary['oof_ap']:.3f}")
    print(f"OOF balanced acc @0.5: {summary['oof_bal_acc_05']:.3f}")
    print(f"OOF balanced acc @Youden(per-fold): {summary['oof_bal_acc_you']:.3f}")
    print(f"Temporal attention mean: 0h={summary['temporal_attention_mean'][0]:.3f}, 6h={summary['temporal_attention_mean'][1]:.3f}, 24h={summary['temporal_attention_mean'][2]:.3f}")

    print("\nSaved:")
    print(f"  - {oof_path}")
    print(f"  - {folds_path}")
    print(f"  - {os.path.join(args.output_dir, 'extracted_features_oof.npz')}")
    print(f"  - {os.path.join(args.output_dir, 'oof_roc.png')} and oof_roc_pr.png")
    print(f"  - {os.path.join(args.output_dir, 'summary.json')}")

    # --------------------------------------------------------------------------
    # Permutation test (optional)
    # --------------------------------------------------------------------------
    if args.permute and args.permute > 0:
        print(f"\n{'='*70}")
        print(f"PERMUTATION TEST (N={args.permute})")
        print(f"{'='*70}")

        rng = np.random.default_rng(args.perm_seed)
        perm_aucs = []

        observed = float(summary["oof_auc"])

        for i in range(int(args.permute)):
            y_perm = rng.permutation(y)
            perm_res = run_cv_evaluation(
                codes=codes,
                y=y_perm,
                timepoint_dirs=tp_dirs,
                args=args,
                run_tag=f"perm{i+1}",
                save_artifacts=False,
                override_epochs=args.perm_epochs,
                override_patience=args.perm_patience,
            )
            perm_auc = float(perm_res["summary"]["oof_auc"])
            perm_aucs.append(perm_auc)
            if (i + 1) % 10 == 0:
                print(f"  Perm {i+1}/{args.permute}: auc={perm_auc:.3f}")

        perm_aucs_arr = np.array(perm_aucs, dtype=float)
        p_value = (1.0 + np.sum(perm_aucs_arr >= observed)) / (1.0 + len(perm_aucs_arr))

        perm_out = {
            "observed_oof_auc": observed,
            "n_permutations": int(args.permute),
            "p_value": float(p_value),
            "perm_auc_mean": float(np.mean(perm_aucs_arr)) if len(perm_aucs_arr) else np.nan,
            "perm_auc_std": float(np.std(perm_aucs_arr)) if len(perm_aucs_arr) else np.nan,
            "perm_aucs": perm_aucs,
        }

        with open(os.path.join(args.output_dir, "permutation_test.json"), "w") as f:
            json.dump(perm_out, f, indent=2)

        print(f"\nPermutation p-value (AUC >= observed): {p_value:.4f}")
        print(f"Saved: {os.path.join(args.output_dir, 'permutation_test.json')}")


if __name__ == '__main__':
    main()
