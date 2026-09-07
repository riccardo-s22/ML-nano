"""
Convolutional Autoencoder with Attention-Based Temporal Aggregation
==================================================================

Regression version to predict ALSFRS total score
------------------------------------------------

This script adapts the previous binary-classification model to
**regress the ALSFRS score** from multi‑timepoint spectroscopic images.

Key changes vs. the classification script
-----------------------------------------
- Targets are **continuous ALSFRS scores** from `ALSFRS.xlsx`.
- Final head is a **regressor** (1 output) instead of classifier.
- Loss is:  total_loss = alpha * reconstruction_MSE + beta * regression_MSE
- Cross‑validation uses standard KFold at the **subject level**.
- We print RMSE, MAE, R² and Pearson r for each epoch (train & val).

Expected files
--------------
- Spectral images per timepoint as Excel:
    TIMEPOINT_DIRS = ["out_0h", "out_6h", "out_24h"]
  Each directory contains `{code}.xlsx`.

- ALSFRS.xlsx with at least the columns:
    "code", " Group", "Total Alsfrsr Score"
  (column names are stripped of whitespace in the script)

You can adapt TIMEPOINT_DIRS and the score column name if needed.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class ExcelImageDataset(Dataset):
    """
    Custom dataset for loading Excel files as image-like arrays
    with multiple timepoints per subject.

    Assumptions
    -----------
    - Each *subject* is identified by a `code` string.
    - For each timepoint directory in `timepoint_dirs`, there is an
      Excel file named `{code}.xlsx`.
    - The Excel sheet layout:
        * Row 1: column headers (e.g., "Emission", "Excitation_...")
        * Col 1: row labels (e.g., emission wavelengths)
        * Remaining cells: numeric data
    - The numeric matrix is normalized to [0, 1] per-file.
    """

    def __init__(
        self,
        codes: List[str],
        targets: List[float],
        timepoint_dirs: List[str],
        transform=None,
    ):
        self.codes = list(codes)
        self.targets = list(targets)
        self.timepoint_dirs = list(timepoint_dirs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.codes)

    @staticmethod
    def load_excel_as_image(filepath: str) -> np.ndarray:
        """Load Excel file at `filepath` and return a 2D float32 array in [0, 1]."""
        wb = load_workbook(filepath, data_only=True)
        ws = wb.active

        data = []
        for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
            # Skip header row
            if row_idx == 0:
                continue

            row_data = []
            for col_idx, cell in enumerate(row):
                # Skip first column (e.g., emission labels)
                if col_idx == 0:
                    continue

                if isinstance(cell, (int, float)) and cell is not None:
                    row_data.append(float(cell))
                else:
                    row_data.append(0.0)

            if row_data:
                data.append(row_data)

        img_array = np.asarray(data, dtype=np.float32)

        if img_array.size == 0:
            raise ValueError(f"No valid numeric data found in {filepath}")

        # Min-max normalization to [0, 1]
        min_val = float(img_array.min())
        max_val = float(img_array.max())
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val)
        else:
            img_array = np.zeros_like(img_array, dtype=np.float32)

        return img_array

    def __getitem__(self, idx: int):
        code = self.codes[idx]
        target = self.targets[idx]

        timepoint_tensors = []
        for tp_dir in self.timepoint_dirs:
            filepath = os.path.join(tp_dir, f"{code}.xlsx")
            img = self.load_excel_as_image(filepath)  # (H, W)

            # Add channel dimension => (1, H, W)
            img = img[np.newaxis, :, :].astype(np.float32)
            tensor_img = torch.from_numpy(img)
            timepoint_tensors.append(tensor_img)

        # Stack to (num_timepoints, 1, H, W)
        images = torch.stack(timepoint_tensors, dim=0)

        if self.transform is not None:
            # Apply same transform to all timepoints
            images = torch.stack([self.transform(tp) for tp in images], dim=0)

        return images, torch.tensor(target, dtype=torch.float32)


# -------------------------------------------------------------------
# Encoder / Decoder
# -------------------------------------------------------------------
class ConvEncoder(nn.Module):
    """
    Shared convolutional encoder.

    Dynamically infers its flatten size on first forward pass so it can
    adapt to arbitrary H×W as long as they are large enough to survive
    three 2×2 pooling operations.
    """

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.flatten_size = None
        self.fc = None
        self.latent_dim = latent_dim

        self.encoded_shape: Tuple[int, int, int] = None
        self.input_shape: Tuple[int, int] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    """
    Convolutional decoder mirroring the encoder, with ConvTranspose2d
    to upsample and a final interpolation to exactly match the original
    H×W.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        encoded_shape: Tuple[int, int, int],
        target_shape: Tuple[int, int],
    ):
        super().__init__()

        self.encoded_channels, self.encoded_h, self.encoded_w = encoded_shape
        self.target_h, self.target_w = target_shape

        self.fc = nn.Linear(
            latent_dim, self.encoded_channels * self.encoded_h * self.encoded_w
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.encoded_channels, self.encoded_h, self.encoded_w)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final_conv(x)

        # Resize to exact target dims
        h, w = x.shape[2], x.shape[3]
        if h != self.target_h or w != self.target_w:
            x = nn.functional.interpolate(
                x, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False
            )

        x = self.sigmoid(x)
        return x


# -------------------------------------------------------------------
# Attention + Regressor
# -------------------------------------------------------------------
class AttentionAggregation(nn.Module):
    """
    Simple additive attention over timepoints.

    Input: (batch, T, latent_dim)
    Output:
        - aggregated: (batch, latent_dim)
        - attention_weights: (batch, T)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, latent_vectors: torch.Tensor):
        # latent_vectors: (B, T, D)
        scores = self.attention_net(latent_vectors).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(scores, dim=1)  # (B, T)

        attn_weights_exp = attn_weights.unsqueeze(-1)  # (B, T, 1)
        aggregated = torch.sum(latent_vectors * attn_weights_exp, dim=1)  # (B, D)

        return aggregated, attn_weights


class Regressor(nn.Module):
    def __init__(self, latent_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, z_agg: torch.Tensor) -> torch.Tensor:
        return self.net(z_agg)


# -------------------------------------------------------------------
# Full model
# -------------------------------------------------------------------
class ConvAutoencoderWithAttention(nn.Module):
    """
    Full dual-objective model:

        timepoint images ──> shared encoder ──> z_t
                                            ├─> decoder (reconstruction per timepoint)
                                            └─> attention over t ─> regressor (ALSFRS)
    """

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, latent_dim)
        self.attention = AttentionAggregation(latent_dim)
        self.decoder = None  # set after first forward
        self.regressor = Regressor(latent_dim)

        self.in_channels = in_channels
        self.latent_dim = latent_dim

    def forward(self, timepoint_images: torch.Tensor):
        """
        Parameters
        ----------
        timepoint_images : (B, T, C, H, W)
        """
        batch_size, num_timepoints = timepoint_images.shape[0], timepoint_images.shape[1]

        latents = []
        for t in range(num_timepoints):
            z_t = self.encoder(timepoint_images[:, t])  # (B, D)
            latents.append(z_t)

        # Initialize decoder once we know encoded / input shapes
        if self.decoder is None:
            self.decoder = ConvDecoder(
                latent_dim=self.latent_dim,
                out_channels=self.in_channels,
                encoded_shape=self.encoder.encoded_shape,
                target_shape=self.encoder.input_shape,
            ).to(timepoint_images.device)

        # (B, T, D)
        latent_tensor = torch.stack(latents, dim=1)

        # Attention aggregation for regression
        z_agg, attn_weights = self.attention(latent_tensor)
        preds = self.regressor(z_agg)  # (B, 1)

        # Per-timepoint reconstruction
        recons = []
        for t in range(num_timepoints):
            recon_t = self.decoder(latent_tensor[:, t])
            recons.append(recon_t)
        reconstructions = torch.stack(recons, dim=1)  # (B, T, C, H, W)

        return reconstructions, preds, attn_weights


# -------------------------------------------------------------------
# Training / Validation loops
# -------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute RMSE, MAE, R^2, Pearson r.
    """
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R^2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Pearson r
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = 0.0

    return rmse, mae, r2, r


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    model.train()

    recon_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()

    total_loss = total_recon = total_reg = 0.0

    all_targets = []
    all_preds = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        recons, preds, _ = model(images)  # preds: (B, 1)

        recon_loss = recon_criterion(recons, images)
        reg_loss = reg_criterion(preds, targets)
        loss = alpha * recon_loss + beta * reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_reg += reg_loss.item()

        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    n_batches = len(dataloader)
    total_loss /= n_batches
    total_recon /= n_batches
    total_reg /= n_batches

    if all_targets:
        y_true = np.concatenate(all_targets, axis=0).flatten()
        y_pred = np.concatenate(all_preds, axis=0).flatten()
        rmse, mae, r2, r = regression_metrics(y_true, y_pred)
    else:
        rmse = mae = r2 = r = 0.0

    return total_loss, total_recon, total_reg, rmse, mae, r2, r


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    model.eval()

    recon_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()

    total_loss = total_recon = total_reg = 0.0

    all_targets = []
    all_preds = []
    all_attn = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)  # (B, 1)

            recons, preds, attn = model(images)

            recon_loss = recon_criterion(recons, images)
            reg_loss = reg_criterion(preds, targets)
            loss = alpha * recon_loss + beta * reg_loss

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_reg += reg_loss.item()

            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_attn.append(attn.detach().cpu().numpy())

    n_batches = len(dataloader)
    total_loss /= n_batches
    total_recon /= n_batches
    total_reg /= n_batches

    if all_targets:
        y_true = np.concatenate(all_targets, axis=0).flatten()
        y_pred = np.concatenate(all_preds, axis=0).flatten()
        rmse, mae, r2, r = regression_metrics(y_true, y_pred)
    else:
        rmse = mae = r2 = r = 0.0

    attn_array = np.concatenate(all_attn, axis=0) if all_attn else np.zeros((0, 0))

    return total_loss, total_recon, total_reg, rmse, mae, r2, r, attn_array


# -------------------------------------------------------------------
# Main training script (subject-level CV)
# -------------------------------------------------------------------
def main():
    # ----------------------------
    # CONFIGURATION
    # ----------------------------
    ALSFRS_FILE = "ALSFRS.xlsx"  # this is your uploaded file

    # For TWO timepoints, put exactly two directories here.
    TIMEPOINT_DIRS = ["out_0h", "out_6h", "out_24h"]

    SCORE_COLUMN = "Total Alsfrsr Score"  # after stripping whitespace
    CODE_COLUMN = "code"

    LATENT_DIM = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    ALPHA = 1.0  # reconstruction loss weight
    BETA = 1.0   # regression loss weight
    N_SPLITS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # LOAD ALSFRS LABELS
    # ----------------------------
    if ALSFRS_FILE.lower().endswith(".xlsx"):
        labels_df = pd.read_excel(ALSFRS_FILE)
    else:
        labels_df = pd.read_csv(ALSFRS_FILE)

    # Strip whitespace in column names to robustly handle " Group"
    labels_df.columns = labels_df.columns.str.strip()

    print(f"Loaded {len(labels_df)} subjects from {ALSFRS_FILE}")
    print("Columns:", list(labels_df.columns))

    if "Group" in labels_df.columns:
        print("Groups:", labels_df["Group"].value_counts().to_dict())

    # Extract codes and ALSFRS scores
    codes = labels_df[CODE_COLUMN].astype(str).values
    scores = labels_df[SCORE_COLUMN].astype(float).values

    # ----------------------------
    # SUBJECT-LEVEL K-FOLD CV
    # ----------------------------
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results_r = []   # store best Pearson r per fold
    fold_results_rmse = []  # store best RMSE per fold

    print("\n" + "=" * 70)
    print("SUBJECT-LEVEL CROSS-VALIDATION (REGRESSION: ALSFRS)")
    print("=" * 70)
    print(f"Total subjects        : {len(codes)}")
    print(f"Timepoints per subject: {len(TIMEPOINT_DIRS)}")
    print(f"Total images          : {len(codes) * len(TIMEPOINT_DIRS)}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(codes), start=1):
        print("\n" + "=" * 70)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("=" * 70)

        train_codes = codes[train_idx]
        val_codes = codes[val_idx]
        train_scores = scores[train_idx]
        val_scores = scores[val_idx]

        print(f"Training subjects   : {len(train_codes)}")
        print(f"Validation subjects : {len(val_codes)}")

        train_dataset = ExcelImageDataset(train_codes, train_scores, TIMEPOINT_DIRS)
        val_dataset = ExcelImageDataset(val_codes, val_scores, TIMEPOINT_DIRS)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Inspect dimensions
        sample_images, _ = next(iter(train_loader))
        num_timepoints, in_channels, height, width = sample_images.shape[1:]
        print("\nData dimensions:")
        print(f"  Timepoints: {num_timepoints}")
        print(f"  Channels  : {in_channels}")
        print(f"  Height    : {height}")
        print(f"  Width     : {width}")

        # Model
        model = ConvAutoencoderWithAttention(
            in_channels=in_channels, latent_dim=LATENT_DIM
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\nModel parameters:")
        print(f"  Total    : {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_r = -1.0  # track best Pearson r
        best_val_rmse = float("inf")

        for epoch in range(1, NUM_EPOCHS + 1):
            (
                tr_loss,
                tr_recon,
                tr_reg,
                tr_rmse,
                tr_mae,
                tr_r2,
                tr_r,
            ) = train_epoch(model, train_loader, optimizer, device, ALPHA, BETA)

            (
                val_loss,
                val_recon,
                val_reg,
                val_rmse,
                val_mae,
                val_r2,
                val_r,
                attn,
            ) = validate(model, val_loader, device, ALPHA, BETA)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"\nEpoch {epoch:03d}/{NUM_EPOCHS} | "
                    f"Train Loss {tr_loss:.3f} (Recon {tr_recon:.3f}, Reg {tr_reg:.3f}) "
                    f"| RMSE {tr_rmse:.3f}, MAE {tr_mae:.3f}, R2 {tr_r2:.3f}, r {tr_r:.3f}\n"
                    f"Val   Loss {val_loss:.3f} (Recon {val_recon:.3f}, Reg {val_reg:.3f}) "
                    f"| RMSE {val_rmse:.3f}, MAE {val_mae:.3f}, R2 {val_r2:.3f}, r {val_r:.3f}"
                )

                if attn.size > 0:
                    mean_attn = attn.mean(axis=0)
                    print("  Mean attention weights per timepoint:")
                    for i, w in enumerate(mean_attn):
                        print(f"    TP{i}: {w:.3f}")

            # Save model if Pearson r improves (or RMSE decreases, whichever you prefer)
            improved = False
            if val_r > best_val_r:
                best_val_r = val_r
                improved = True
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                improved = True

            if improved:
                torch.save(model.state_dict(), f"best_reg_model_fold{fold}.pth")
                print(
                    f"  ✓ New best model saved "
                    f"(val r: {val_r:.3f}, RMSE: {val_rmse:.3f})"
                )

        fold_results_r.append(best_val_r)
        fold_results_rmse.append(best_val_rmse)
        print("\n" + "-" * 70)
        print(f"FOLD {fold} BEST VAL r    : {best_val_r:.3f}")
        print(f"FOLD {fold} BEST VAL RMSE : {best_val_rmse:.3f}")
        print("-" * 70)

    # Summary
    fold_results_r = np.array(fold_results_r, dtype=float)
    fold_results_rmse = np.array(fold_results_rmse, dtype=float)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY (ALSFRS REGRESSION)")
    print("=" * 70)
    print(
        f"Fold Pearson r : {', '.join(f'{a:.3f}' for a in fold_results_r)} "
        f"(mean {fold_results_r.mean():.3f} ± {fold_results_r.std():.3f})"
    )
    print(
        f"Fold RMSE      : {', '.join(f'{a:.3f}' for a in fold_results_rmse)} "
        f"(mean {fold_results_rmse.mean():.3f} ± {fold_results_rmse.std():.3f})"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
