#!/usr/bin/env python3
"""
HYBRID CHIRALITY-FOCUSED AUTOENCODER + ML PIPELINE
===================================================

This combines:
1. Lightweight autoencoder FOCUSED on chirality regions (not full matrix)
2. Latent feature extraction for interpretability  
3. Ensemble ML with proper regularization

WHY THIS APPROACH?
- Full-matrix deep autoencoders overfit on 39 samples
- Chirality regions contain the discriminative signal
- Extracting small ROI patches reduces dimensionality
- Shallow autoencoder learns representations without memorizing

ARCHITECTURE:
- Input: Concatenated chirality ROI patches (12 chiralities × small window)
- Encoder: 2 conv layers (no deep hierarchy needed)
- Latent: 64-128 dimensions (matches data complexity)
- Dual loss: Reconstruction + Classification
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from openpyxl import load_workbook
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# CHIRALITY DEFINITIONS
# =============================================================================

CHIRALITIES = [
    {"name": "(8,3)",  "em": 973.98,  "exc": 673.94, "height": 0.480},
    {"name": "(6,5)",  "em": 987.82,  "exc": 577.12, "height": 0.762},
    {"name": "(7,5)",  "em": 1047.81, "exc": 653.32, "height": 0.648},
    {"name": "(10,2)", "em": 1080.60, "exc": 745.92, "height": 0.389},
    {"name": "(9,4)",  "em": 1131.96, "exc": 731.39, "height": 0.553},
    {"name": "(8,4)",  "em": 1130.34, "exc": 599.78, "height": 0.591},
    {"name": "(7,6)",  "em": 1138.19, "exc": 659.79, "height": 0.725},
    {"name": "(8,6)",  "em": 1200.03, "exc": 727.40, "height": 0.318},
    {"name": "(8,7)",  "em": 1288.27, "exc": 740.87, "height": 0.134},
    {"name": "(9,5)",  "em": 1262.98, "exc": 685.15, "height": 0.190},
    {"name": "(10,3)", "em": 1267.70, "exc": 648.97, "height": 0.190},
    {"name": "(10,5)", "em": 1282.97, "exc": 801.23, "height": 0.100},
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_excel_eem(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load EEM data from Excel file."""
    wb = load_workbook(filepath, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    
    header = rows[0]
    excitations = []
    for h in header[1:]:
        if h is None:
            continue
        s = str(h)
        m = re.search(r'Excitation_0*([0-9]+(?:\.[0-9]+)?)', s, re.IGNORECASE)
        if m:
            excitations.append(float(m.group(1)))
    
    excitations = np.array(excitations)
    
    emissions = []
    data_rows = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        try:
            em = float(row[0])
            emissions.append(em)
            row_data = [float(v) if isinstance(v, (int, float)) and v is not None else 0.0 
                       for v in row[1:len(excitations)+1]]
            data_rows.append(row_data)
        except:
            continue
    
    return np.array(data_rows, dtype=np.float32), np.array(emissions), excitations


def extract_chirality_patches(
    data: np.ndarray, 
    emissions: np.ndarray, 
    excitations: np.ndarray,
    patch_em: int = 20,  # ±20 rows around chirality center
    patch_exc: int = 4    # ±4 columns around chirality center
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract small patches around each chirality center.
    
    Returns:
        patches: Array of shape (n_chiralities, 2*patch_em+1, 2*patch_exc+1)
        names: List of chirality names
    """
    patches = []
    names = []
    
    for chir in CHIRALITIES:
        em_idx = np.argmin(np.abs(emissions - chir['em']))
        exc_idx = np.argmin(np.abs(excitations - chir['exc']))
        
        # Extract patch with bounds checking
        em_start = max(0, em_idx - patch_em)
        em_end = min(data.shape[0], em_idx + patch_em + 1)
        exc_start = max(0, exc_idx - patch_exc)
        exc_end = min(data.shape[1], exc_idx + patch_exc + 1)
        
        patch = data[em_start:em_end, exc_start:exc_end]
        
        # Pad if necessary to ensure consistent size
        target_em = 2 * patch_em + 1
        target_exc = 2 * patch_exc + 1
        
        if patch.shape[0] < target_em or patch.shape[1] < target_exc:
            padded = np.zeros((target_em, target_exc), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        
        patches.append(patch)
        names.append(chir['name'])
    
    return np.array(patches), names


class ChiralityPatchDataset(Dataset):
    """Dataset that extracts chirality patches from EEM files."""
    
    def __init__(self, file_lists: List[Dict[str, str]], labels: List[int],
                 patch_em: int = 20, patch_exc: int = 4, normalize: bool = True):
        """
        Args:
            file_lists: List of dicts with timepoint files {'0h': path, '6h': path, '24h': path}
            labels: List of labels (0 or 1)
            patch_em: Patch size in emission direction
            patch_exc: Patch size in excitation direction
        """
        self.file_lists = file_lists
        self.labels = labels
        self.patch_em = patch_em
        self.patch_exc = patch_exc
        self.normalize = normalize
        self._cache = {}
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        files = self.file_lists[idx]
        label = self.labels[idx]
        
        # Load patches for all timepoints
        timepoint_patches = []
        
        for tp in ['0h', '6h', '24h']:
            filepath = files[tp]
            
            if filepath not in self._cache:
                data, emissions, excitations = load_excel_eem(filepath)
                
                # Normalize the EEM
                if self.normalize:
                    data = (data - data.min()) / (data.max() - data.min() + 1e-10)
                
                patches, _ = extract_chirality_patches(
                    data, emissions, excitations, 
                    self.patch_em, self.patch_exc
                )
                self._cache[filepath] = patches
            else:
                patches = self._cache[filepath]
            
            timepoint_patches.append(patches)
        
        # Stack: (3 timepoints, 12 chiralities, H, W)
        patches_tensor = torch.FloatTensor(np.stack(timepoint_patches))
        
        return patches_tensor, torch.tensor(label, dtype=torch.long)


# =============================================================================
# LIGHTWEIGHT AUTOENCODER
# =============================================================================

class ChiralityEncoder(nn.Module):
    """
    Lightweight encoder for chirality patches.
    
    Input: (batch, timepoints, chiralities, H, W)
    Output: (batch, latent_dim)
    """
    
    def __init__(self, n_timepoints: int = 3, n_chiralities: int = 12,
                 patch_h: int = 41, patch_w: int = 9, latent_dim: int = 64):
        super().__init__()
        
        self.n_timepoints = n_timepoints
        self.n_chiralities = n_chiralities
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.latent_dim = latent_dim
        
        # Process each chirality patch with shared conv
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2))  # Fixed output size
        )
        
        # Combine features from all chiralities and timepoints
        combined_dim = 32 * 4 * 2 * n_chiralities * n_timepoints
        
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )
        
        self.combined_dim = combined_dim
        
    def forward(self, x):
        # x: (batch, timepoints, chiralities, H, W)
        batch_size = x.shape[0]
        
        # Reshape to process all patches through shared encoder
        # (batch * timepoints * chiralities, 1, H, W)
        x = x.view(-1, 1, self.patch_h, self.patch_w)
        
        # Encode patches
        x = self.patch_encoder(x)
        
        # Reshape back and flatten
        x = x.view(batch_size, -1)
        
        # Project to latent space
        z = self.fc(x)
        
        return z


class ChiralityDecoder(nn.Module):
    """Decoder to reconstruct chirality patches from latent."""
    
    def __init__(self, n_timepoints: int = 3, n_chiralities: int = 12,
                 patch_h: int = 41, patch_w: int = 9, latent_dim: int = 64):
        super().__init__()
        
        self.n_timepoints = n_timepoints
        self.n_chiralities = n_chiralities
        self.patch_h = patch_h
        self.patch_w = patch_w
        
        # Project from latent to spatial
        combined_dim = 32 * 4 * 2 * n_chiralities * n_timepoints
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, combined_dim)
        )
        
        # Decode each patch
        self.patch_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        batch_size = z.shape[0]
        
        # Project to spatial features
        x = self.fc(z)
        
        # Reshape for deconv
        x = x.view(-1, 32, 4, 2)
        
        # Decode
        x = self.patch_decoder(x)
        
        # Resize to exact patch dimensions
        x = nn.functional.interpolate(x, size=(self.patch_h, self.patch_w), mode='bilinear')
        
        # Reshape to (batch, timepoints, chiralities, H, W)
        x = x.view(batch_size, self.n_timepoints, self.n_chiralities, self.patch_h, self.patch_w)
        
        return x


class ChiralityClassifier(nn.Module):
    """Classification head for latent features."""
    
    def __init__(self, latent_dim: int = 64, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, z):
        return self.classifier(z)


class ChiralityAutoencoder(nn.Module):
    """Complete autoencoder with classification head."""
    
    def __init__(self, n_timepoints: int = 3, n_chiralities: int = 12,
                 patch_h: int = 41, patch_w: int = 9, latent_dim: int = 64,
                 num_classes: int = 2):
        super().__init__()
        
        self.encoder = ChiralityEncoder(
            n_timepoints, n_chiralities, patch_h, patch_w, latent_dim
        )
        self.decoder = ChiralityDecoder(
            n_timepoints, n_chiralities, patch_h, patch_w, latent_dim
        )
        self.classifier = ChiralityClassifier(latent_dim, num_classes)
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, logits, z


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, alpha=1.0, beta=1.0):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon = 0.0
    total_class = 0.0
    correct = 0
    total = 0
    
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        recon, logits, z = model(patches)
        
        recon_loss = recon_criterion(recon, patches)
        class_loss = class_criterion(logits, labels)
        loss = alpha * recon_loss + beta * class_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_class += class_loss.item()
        
        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    return {
        'loss': total_loss / len(loader),
        'recon_loss': total_recon / len(loader),
        'class_loss': total_class / len(loader),
        'accuracy': 100.0 * correct / total
    }


def validate(model, loader, device, alpha=1.0, beta=1.0):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_recon = 0.0
    total_class = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_latents = []
    
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            
            recon, logits, z = model(patches)
            
            recon_loss = recon_criterion(recon, patches)
            class_loss = class_criterion(logits, labels)
            loss = alpha * recon_loss + beta * class_loss
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_class += class_loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_latents.append(z.cpu().numpy())
            
            _, pred = torch.max(logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    latents = np.concatenate(all_latents, axis=0)
    
    return {
        'loss': total_loss / len(loader),
        'recon_loss': total_recon / len(loader),
        'class_loss': total_class / len(loader),
        'accuracy': 100.0 * correct / total,
        'auc': auc,
        'latents': latents,
        'labels': np.array(all_labels)
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def find_sample_files(data_dir: Path, code: str) -> Dict[str, str]:
    """Find all timepoint files for a sample."""
    files = {}
    base = code.replace('.', '_')
    
    for tp in ['0h', '6h', '24h']:
        # Try different patterns
        patterns = [
            f"{base}_{tp}.xlsx",
            f"{code.split('.')[0]}_{tp}.xlsx",
        ]
        
        for pattern in patterns:
            filepath = data_dir / pattern
            if filepath.exists():
                files[tp] = str(filepath)
                break
        
        if tp not in files:
            # Search
            for f in data_dir.glob(f"*_{tp}.xlsx"):
                fname = f.stem.replace('_', '.')
                if fname.startswith(code.split('.')[0]):
                    files[tp] = str(f)
                    break
    
    return files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Chirality Autoencoder + ML')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./hybrid_results')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1.0, help='Reconstruction weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Classification weight')
    parser.add_argument('--patience', type=int, default=20)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("HYBRID CHIRALITY-FOCUSED AUTOENCODER + ML PIPELINE")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    print(f"\nSamples: {len(labels_df)} ({(labels_df['group']=='ALS').sum()} ALS, "
          f"{(labels_df['group']=='CTRL').sum()} CTRL)")
    
    # Find files for all samples
    data_dir = Path(args.data_dir)
    file_lists = []
    labels = []
    valid_codes = []
    
    for _, row in labels_df.iterrows():
        code = str(row['code'])
        group = row['group']
        
        files = find_sample_files(data_dir, code)
        
        if len(files) == 3:
            file_lists.append(files)
            labels.append(1 if group == 'ALS' else 0)
            valid_codes.append(code)
        else:
            print(f"  [SKIP] {code}: Missing timepoints")
    
    labels = np.array(labels)
    print(f"\nValid samples: {len(labels)}")
    
    # Determine patch dimensions from first sample
    test_data, test_em, test_exc = load_excel_eem(file_lists[0]['0h'])
    test_patches, _ = extract_chirality_patches(test_data, test_em, test_exc)
    patch_h, patch_w = test_patches.shape[1], test_patches.shape[2]
    print(f"Patch dimensions: {patch_h} × {patch_w}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    
    fold_results = []
    all_latents = []
    all_labels_list = []
    
    print(f"\n{'='*70}")
    print(f"TRAINING {args.n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(file_lists, labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{args.n_folds}")
        print(f"{'='*70}")
        
        # Split data
        train_files = [file_lists[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_files = [file_lists[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        print(f"Train: {len(train_files)} ({sum(train_labels)} ALS)")
        print(f"Val: {len(val_files)} ({sum(val_labels)} ALS)")
        
        # Datasets
        train_dataset = ChiralityPatchDataset(train_files, train_labels.tolist(),
                                              patch_em=20, patch_exc=4)
        val_dataset = ChiralityPatchDataset(val_files, val_labels.tolist(),
                                            patch_em=20, patch_exc=4)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Model
        model = ChiralityAutoencoder(
            n_timepoints=3, n_chiralities=12,
            patch_h=patch_h, patch_w=patch_w,
            latent_dim=args.latent_dim, num_classes=2
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training
        best_val_auc = 0.0
        best_model_state = None
        no_improve = 0
        
        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, DEVICE,
                                       args.alpha, args.beta)
            val_metrics = validate(model, val_loader, DEVICE, args.alpha, args.beta)
            
            scheduler.step(val_metrics['loss'])
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Train Acc={train_metrics['accuracy']:.1f}%, "
                      f"Val Acc={val_metrics['accuracy']:.1f}%, Val AUC={val_metrics['auc']:.3f}")
            
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final validation
        final_metrics = validate(model, val_loader, DEVICE, args.alpha, args.beta)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': final_metrics['accuracy'],
            'auc': final_metrics['auc'],
            'recon_loss': final_metrics['recon_loss']
        })
        
        # Store latents for downstream ML
        all_latents.append(final_metrics['latents'])
        all_labels_list.extend(final_metrics['labels'])
        
        # Save model
        torch.save(best_model_state, output_dir / f'model_fold{fold+1}.pth')
        
        print(f"\n  FOLD {fold+1} RESULT: Acc={final_metrics['accuracy']:.1f}%, AUC={final_metrics['auc']:.3f}")
    
    # Cross-validation summary
    print(f"\n{'='*70}")
    print("AUTOENCODER CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    accs = [r['accuracy'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    
    print(f"\nAccuracy: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")
    print(f"AUC-ROC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    
    # Now train ML on full data with latent features
    print(f"\n{'='*70}")
    print("LATENT FEATURE ML COMPARISON")
    print(f"{'='*70}")
    
    # Extract latents from all data using last fold model
    full_dataset = ChiralityPatchDataset(file_lists, labels.tolist(),
                                         patch_em=20, patch_exc=4)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    all_z = []
    with torch.no_grad():
        for patches, _ in full_loader:
            patches = patches.to(DEVICE)
            z = model.encoder(patches)
            all_z.append(z.cpu().numpy())
    
    X_latent = np.concatenate(all_z, axis=0)
    y = labels
    
    print(f"\nLatent features: {X_latent.shape[1]} dimensions")
    
    # Compare ML models on latent features
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=5000, C=0.1),
        'SVM_RBF': SVC(kernel='rbf', class_weight='balanced', probability=True, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, 
                                               class_weight='balanced', random_state=SEED),
    }
    
    ml_results = {}
    
    for name, clf in models.items():
        # Cross-validate
        skf_ml = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        preds = []
        probs = []
        trues = []
        
        for train_idx, test_idx in skf_ml.split(X_latent, y):
            X_tr, X_te = X_latent[train_idx], X_latent[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            clf.fit(X_tr, y_tr)
            preds.extend(clf.predict(X_te))
            
            if hasattr(clf, 'predict_proba'):
                probs.extend(clf.predict_proba(X_te)[:, 1])
            else:
                probs.extend(clf.decision_function(X_te))
            
            trues.extend(y_te)
        
        acc = accuracy_score(trues, preds)
        auc = roc_auc_score(trues, probs)
        
        ml_results[name] = {'accuracy': acc, 'auc': auc}
        print(f"  {name}: Acc={acc:.3f}, AUC={auc:.3f}")
    
    # Summary
    summary = {
        'autoencoder_cv': {
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs)),
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'fold_results': fold_results
        },
        'ml_on_latents': {k: {kk: float(vv) for kk, vv in v.items()} 
                        for k, v in ml_results.items()},
        'latent_dim': args.latent_dim,
        'n_samples': len(labels),
        'n_chiralities': len(CHIRALITIES)
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save latent features
    pd.DataFrame(X_latent, columns=[f'z_{i}' for i in range(X_latent.shape[1])]).to_csv(
        output_dir / 'latent_features.csv', index=False
    )
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nAutoencoder (end-to-end):")
    print(f"  Accuracy: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")
    print(f"  AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    
    best_ml = max(ml_results.keys(), key=lambda x: ml_results[x]['auc'])
    print(f"\nBest ML on latent features: {best_ml}")
    print(f"  Accuracy: {ml_results[best_ml]['accuracy']:.3f}")
    print(f"  AUC: {ml_results[best_ml]['auc']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
