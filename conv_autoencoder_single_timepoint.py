"""
conv_autoencoder_single_tp.py
--------------------------------
Single-timepoint version of the original conv_autoencoder_detailed.py.

Key changes vs multi-timepoint version:
- Dataset expects ONE Excel file per subject (e.g., 0h only) from a single directory.
- No attention module (not needed with a single timepoint).
- Encoder/Decoder architecture and joint loss (reconstruction + classification) retained.
- Training bug fixed: gradients are enabled; validate() is defined before main(); stray code removed.

Disk layout expected:
  sample_labels.csv   # columns: code, group
  out_tp/             # SINGLE timepoint directory (configurable)
    ├─ <code1>.xlsx
    ├─ <code2>.xlsx
    └─ ...

Each Excel file must have identical H×W (after dropping the first row and column).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# ----------------------------
# Configuration
# ----------------------------
LABELS_CSV = "sample_labels.csv"     # must have columns: code, group
TIMEPOINT_DIR = "out_0h"             # single directory containing <code>.xlsx
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LATENT_DIM = 256                     # smaller latent is often better with small N
ALPHA = 1.0                          # reconstruction loss weight
BETA = 1.0                           # classification loss weight
N_SPLITS = 5
RANDOM_STATE = 42


# ----------------------------
# Dataset
# ----------------------------
class SingleExcelDataset(Dataset):
    def __init__(self, codes, labels, timepoint_dir, transform=None):
        self.codes = list(codes)
        self.labels = list(labels)
        self.timepoint_dir = timepoint_dir
        self.transform = transform

    def __len__(self):
        return len(self.codes)

    def load_excel_as_image(self, filepath) -> np.ndarray:
        # Read Excel with pandas (openpyxl engine implicitly used).
        df = pd.read_excel(filepath, engine='openpyxl', header=None)

        # Drop first row (header-like) and first column (row labels-like)
        # Keep numeric portion only
        if df.shape[0] < 2 or df.shape[1] < 2:
            raise ValueError(f"Excel file too small after header trim: {filepath}")
        data_block = df.iloc[1:, 1:]

        # Coerce to numeric
        data_block = data_block.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        arr = data_block.to_numpy(dtype=np.float32)

        # Min-Max normalize per file
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        return arr

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = int(self.labels[idx])

        fpath = os.path.join(self.timepoint_dir, f"{code}.xlsx")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing Excel for code '{code}': {fpath}")

        img = self.load_excel_as_image(fpath)   # (H, W) float32 in [0,1]

        # Add channel dim -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.from_numpy(img)      # float32
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor


# ----------------------------
# Model components
# ----------------------------
class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/2 x W/2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/4 x W/4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/8 x W/8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.input_shape = None      # (H, W)
        self.encoded_shape = None    # (C, H', W')
        self.fc = None               # created dynamically

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        if self.input_shape is None:
            _, _, H, W = x.shape
            self.input_shape = (H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (B, 256, H', W')

        if self.encoded_shape is None:
            self.encoded_shape = x.shape[1:]  # (C, H', W')
            flat_size = int(np.prod(self.encoded_shape))
            self.fc = nn.Linear(flat_size, self.latent_dim).to(x.device)

        B = x.shape[0]
        x = x.view(B, -1)
        z = self.fc(x)  # (B, latent_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int,
                 encoded_shape: Tuple[int, int, int], target_shape: Tuple[int, int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.encoded_shape = encoded_shape
        self.target_shape = target_shape  # (H, W)

        C, H, W = encoded_shape
        self.flat_size = C * H * W

        self.fc = nn.Linear(latent_dim, self.flat_size)

        # Transposed convs roughly mirror encoder pooling structure
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.shape[0]
        x = self.fc(z)  # (B, flat_size)
        x = x.view(B, *self.encoded_shape)  # (B, 256, H', W')

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)

        # Ensure final size matches target H, W
        target_H, target_W = self.target_shape
        if x.shape[-2] != target_H or x.shape[-1] != target_W:
            x = torch.nn.functional.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
        return x


class AEClassifier(nn.Module):
    """Single-timepoint autoencoder + classifier."""
    def __init__(self, in_channels: int = 1, latent_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = None  # created lazily after first encoding
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 1, H, W)
        z = self.encoder(x)  # (B, latent_dim)

        # Lazy-create decoder when encoded/input shapes are known
        if self.decoder is None:
            self.decoder = ConvDecoder(
                latent_dim=self.encoder.latent_dim,
                out_channels=1,
                encoded_shape=self.encoder.encoded_shape,
                target_shape=self.encoder.input_shape,
            ).to(x.device)

        recon = self.decoder(z)  # (B, 1, H, W)
        logits = self.classifier(z)  # (B, num_classes)
        return recon, logits, z


# ----------------------------
# Training / Validation
# ----------------------------

def train_epoch(model, dataloader, optimizer, device, alpha=1.0, beta=1.0):
    model.train()
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)  # (B, 1, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        recon, logits, _ = model(images)
        recon_loss = reconstruction_criterion(recon, images)
        class_loss = classification_criterion(logits, labels)
        loss = alpha * recon_loss + beta * class_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()

        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    n_batches = max(1, len(dataloader))
    return (
        total_loss / n_batches,
        total_recon_loss / n_batches,
        total_class_loss / n_batches,
        (100.0 * correct / total) if total > 0 else 0.0
    )


@torch.no_grad()

def validate(model, dataloader, device, alpha=1.0, beta=1.0):
    model.eval()
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_class_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        recon, logits, _ = model(images)
        recon_loss = reconstruction_criterion(recon, images)
        class_loss = classification_criterion(logits, labels)
        loss = alpha * recon_loss + beta * class_loss

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()

        _, pred = torch.max(logits, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    n_batches = max(1, len(dataloader))
    return (
        total_loss / n_batches,
        total_recon_loss / n_batches,
        total_class_loss / n_batches,
        (100.0 * correct / total) if total > 0 else 0.0
    )


# ----------------------------
# Driver
# ----------------------------

def main():
    # Reproducibility (basic)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load labels
    df = pd.read_csv(LABELS_CSV)
    if not {"code", "group"}.issubset(df.columns):
        raise ValueError("sample_labels.csv must contain columns: code, group")

    le = LabelEncoder()
    df["encoded_group"] = le.fit_transform(df["group"].astype(str))

    codes = df["code"].astype(str).values
    labels = df["encoded_group"].values

    # Sanity: verify at least one file exists
    missing = [c for c in codes if not os.path.exists(os.path.join(TIMEPOINT_DIR, f"{c}.xlsx"))]
    if missing:
        raise FileNotFoundError(f"Missing Excel files in '{TIMEPOINT_DIR}' for codes: {missing[:5]}{' ...' if len(missing)>5 else ''}")

    # Peek one file to show H×W
    tmp_path = os.path.join(TIMEPOINT_DIR, f"{codes[0]}.xlsx")
    tmp_arr = pd.read_excel(tmp_path, engine='openpyxl', header=None).iloc[1:,1:].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    print(f"Detected image size (before normalization) H×W = {tmp_arr.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(codes, labels), start=1):
        print(f"\n=== Fold {fold_idx}/{N_SPLITS} ===")

        train_codes, val_codes = codes[train_idx], codes[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_ds = SingleExcelDataset(train_codes, train_labels, TIMEPOINT_DIR)
        val_ds   = SingleExcelDataset(val_codes, val_labels, TIMEPOINT_DIR)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = AEClassifier(in_channels=1, latent_dim=LATENT_DIM, num_classes=len(le.classes_)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = -1.0
        history = {"train": [], "val": []}

        for epoch in range(1, NUM_EPOCHS + 1):
            tr_loss, tr_recon, tr_cls, tr_acc = train_epoch(model, train_loader, optimizer, device, ALPHA, BETA)
            va_loss, va_recon, va_cls, va_acc = validate(model, val_loader, device, ALPHA, BETA)

            history["train"].append({"loss": tr_loss, "recon": tr_recon, "cls": tr_cls, "acc": tr_acc})
            history["val"].append({"loss": va_loss, "recon": va_recon, "cls": va_cls, "acc": va_acc})

            print(f"Epoch {epoch:03d} | "
                  f"Train: loss {tr_loss:.4f} (recon {tr_recon:.4f}, cls {tr_cls:.4f}), acc {tr_acc:.2f}% | "
                  f"Val: loss {va_loss:.4f} (recon {va_recon:.4f}, cls {va_cls:.4f}), acc {va_acc:.2f}%")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), f"best_model_fold{fold_idx}.pth")

        print(f"Best val acc (fold {fold_idx}): {best_val_acc:.2f}%")
        fold_results.append({"fold": fold_idx, "best_val_acc": best_val_acc, "history": history})

    # Summary
    mean_acc = np.mean([fr["best_val_acc"] for fr in fold_results]) if fold_results else float('nan')
    print("\n==== CV Summary ====")
    for fr in fold_results:
        print(f"Fold {fr['fold']}: {fr['best_val_acc']:.2f}%")
    print(f"Mean best val acc: {mean_acc:.2f}%")


if __name__ == "__main__":
    main()
