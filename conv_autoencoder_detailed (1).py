"""
Convolutional Autoencoder with Attention-Based Temporal Aggregation

This architecture implements a dual-objective deep learning model for binary classification
of subjects with multiple timepoint observations. The model simultaneously:
1. Learns to reconstruct input images (unsupervised feature learning)
2. Classifies subjects into two groups (supervised classification)

Key Design Principles:
- Shared encoder extracts features useful for both reconstruction and classification
- Attention mechanism learns which timepoints are most discriminative
- Reconstruction acts as regularizer to prevent overfitting on limited data
- Subject-level cross-validation prevents data leakage between timepoints
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
import os
from typing import Tuple, List
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ExcelImageDataset(Dataset):
    """
    Custom dataset for loading Excel files as image-like arrays with multiple timepoints.
    
    Design rationale:
    - Each subject has 3 Excel files (one per timepoint: 0h, 6h, 24h)
    - Excel files are treated as 2D numerical arrays (analogous to images)
    - All 3 timepoints are loaded together to maintain subject-level structure
    - First row (headers) and first column (emission labels) are excluded from data
    """
    
    def __init__(self, codes: List[str], labels: List[int], 
                 timepoint_dirs: List[str], transform=None):
        """
        Args:
            codes: Subject identifiers (filenames without extension)
            labels: Binary group labels for classification
            timepoint_dirs: List of 3 directories containing timepoint data
            transform: Optional data augmentation (not currently used)
        """
        self.codes = codes
        self.labels = labels
        self.timepoint_dirs = timepoint_dirs
        self.transform = transform
        
    def __len__(self):
        return len(self.codes)
    
    def load_excel_as_image(self, filepath: str) -> np.ndarray:
        """
        Load Excel file and convert to numpy array, treating it as image data.
        
        Processing steps:
        1. Skip first row (column headers like "Emission", "Excitation_...")
        2. Skip first column (row labels like emission wavelengths)
        3. Extract only the numeric data matrix
        4. Normalize to [0, 1] range for neural network input
        
        Returns:
            2D numpy array of shape (n_rows, n_cols) normalized to [0, 1]
        """
        wb = load_workbook(filepath, data_only=True)
        ws = wb.active
        
        # Extract numerical data from worksheet, skipping first row (headers)
        data = []
        for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
            # Skip first row (column headers)
            if row_idx == 0:
                continue
            
            row_data = []
            for col_idx, cell in enumerate(row):
                # Skip first column (row labels/emission values)
                if col_idx == 0:
                    continue
                
                # Convert to float, handling None and non-numeric values
                if isinstance(cell, (int, float)) and cell is not None:
                    row_data.append(float(cell))
                else:
                    row_data.append(0.0)
            
            # Only add row if it has data
            if len(row_data) > 0:
                data.append(row_data)
        
        # Convert to numpy array
        img_array = np.array(data, dtype=np.float32)
        
        # Check if we got valid data
        if img_array.size == 0:
            raise ValueError(f"No valid data found in {filepath}")
        
        # Normalize to [0, 1] range for stable neural network training
        min_val = img_array.min()
        max_val = img_array.max()
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val)
        else:
            # If all values are the same, just return zeros
            img_array = np.zeros_like(img_array)
        
        return img_array
    
    def __getitem__(self, idx):
        """
        Load all 3 timepoint images for a single subject.
        
        Returns:
            images: Tensor of shape (3, 1, H, W) - 3 timepoints, 1 channel, H rows, W columns
            label: Integer label (0 or 1) for binary classification
        """
        code = self.codes[idx]
        label = self.labels[idx]
        
        # Load images from all three timepoints (0h, 6h, 24h)
        timepoint_images = []
        for tp_dir in self.timepoint_dirs:
            filepath = os.path.join(tp_dir, f"{code}.xlsx")
            img = self.load_excel_as_image(filepath)
            
            # Add channel dimension (grayscale image has 1 channel)
            # Shape: (H, W) -> (1, H, W)
            if len(img.shape) == 2:
                img = img[np.newaxis, :, :]
            
            timepoint_images.append(torch.FloatTensor(img))
        
        # Stack timepoints: shape becomes (3, C, H, W)
        # This maintains the temporal structure for the attention mechanism
        images = torch.stack(timepoint_images)
        
        if self.transform:
            # Apply same transform to all timepoints to maintain consistency
            images = torch.stack([self.transform(img) for img in images])
        
        return images, torch.tensor(label, dtype=torch.long)


class ConvEncoder(nn.Module):
    """
    Convolutional encoder that extracts hierarchical features from images.
    
    Architecture Philosophy:
    - Convolutional layers learn spatial patterns (edges, textures, structures)
    - Each layer learns increasingly abstract features through the hierarchy
    - MaxPooling progressively reduces spatial dimensions (downsampling)
    - BatchNorm stabilizes training and acts as regularization
    - ReLU introduces non-linearity for learning complex patterns
    
    The encoder is SHARED across all timepoints, ensuring consistent feature extraction.
    This is critical for the attention mechanism to meaningfully compare timepoints.
    """
    
    def __init__(self, in_channels: int, latent_dim: int):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale data)
            latent_dim: Dimension of the compressed latent representation
        """
        super(ConvEncoder, self).__init__()
        
        # First convolutional block
        # Purpose: Learn low-level features (edges, gradients, basic textures)
        # Conv2d: Applies learnable filters that detect spatial patterns
        # BatchNorm2d: Normalizes activations for stable training
        # ReLU: Non-linear activation (outputs max(0, x))
        # MaxPool2d: Downsamples by taking maximum in 2x2 windows
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 32 filters
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half
        )
        
        # Second convolutional block
        # Purpose: Combine low-level features into mid-level patterns
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 filters
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Further spatial reduction
        )
        
        # Third convolutional block
        # Purpose: Learn high-level abstract features
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 filters
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Another spatial reduction
        )
        
        # Fourth convolutional block
        # Purpose: Learn highest-level abstract representations
        # Note: No pooling here - maintains spatial structure before flattening
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 filters
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # These will be initialized dynamically based on input dimensions
        # This allows the model to work with any input size
        self.flatten_size = None
        self.fc = None  # Fully connected layer to compress to latent_dim
        self.latent_dim = latent_dim
        self.encoded_shape = None  # Store shape for decoder initialization
        self.input_shape = None  # Store original input shape for exact reconstruction
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Latent vector of shape (batch, latent_dim)
        
        Process:
        1. Store input dimensions for decoder
        2. Apply convolutional blocks sequentially (creates feature hierarchy)
        3. Flatten spatial features into 1D vector
        4. Project to latent space via fully connected layer
        """
        # Store original input shape for decoder to match exactly
        if self.input_shape is None:
            self.input_shape = (x.shape[2], x.shape[3])
        
        # Apply convolutional hierarchy
        # Each layer extracts increasingly abstract features
        x = self.conv1(x)  # Low-level features
        x = self.conv2(x)  # Mid-level patterns
        x = self.conv3(x)  # High-level structures
        x = self.conv4(x)  # Abstract representations
        
        # Store the encoded shape (for decoder initialization)
        if self.encoded_shape is None:
            self.encoded_shape = (x.shape[1], x.shape[2], x.shape[3])
        
        # Initialize FC layer on first forward pass (now we know dimensions)
        # This dynamic initialization allows flexibility with input sizes
        if self.fc is None:
            self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
        
        # Flatten spatial dimensions: (batch, channels, H, W) -> (batch, channels*H*W)
        x = x.view(x.size(0), -1)
        
        # Project to latent space: (batch, flatten_size) -> (batch, latent_dim)
        # This is the compressed representation used by both decoder and classifier
        x = self.fc(x)
        
        return x


class ConvDecoder(nn.Module):
    """
    Convolutional decoder that reconstructs images from latent representations.
    
    Architecture Philosophy:
    - Mirror of the encoder, but using transpose convolutions for upsampling
    - ConvTranspose2d "reverses" the pooling operations to increase spatial dimensions
    - Final interpolation ensures exact dimension matching with input
    
    Purpose in Dual-Objective Framework:
    - Reconstruction loss forces encoder to preserve information
    - Acts as regularization to prevent overfitting on limited subjects
    - Ensures learned features are information-rich, not just discriminative
    """
    
    def __init__(self, latent_dim: int, out_channels: int, encoded_shape: tuple, target_shape: tuple):
        """
        Args:
            latent_dim: Dimension of latent representation
            out_channels: Number of output channels (1 for grayscale)
            encoded_shape: Shape after encoding (channels, height, width)
            target_shape: Original input dimensions (height, width) for exact reconstruction
        """
        super(ConvDecoder, self).__init__()
        
        # Store shapes for reconstruction
        self.encoded_channels, self.encoded_h, self.encoded_w = encoded_shape
        self.target_h, self.target_w = target_shape
        
        # Project latent vector back to spatial feature maps
        self.fc = nn.Linear(latent_dim, self.encoded_channels * self.encoded_h * self.encoded_w)
        
        # First transpose convolutional block
        # ConvTranspose2d with stride=2 doubles spatial dimensions
        # This "reverses" the MaxPool operations from encoding
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Second transpose convolutional block
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Third transpose convolutional block
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        
        # Sigmoid ensures output is in [0, 1] range (matching normalized input)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        """
        Reconstruct image from latent representation.
        
        Args:
            z: Latent vector of shape (batch, latent_dim)
        
        Returns:
            Reconstructed image of shape (batch, channels, height, width)
        
        Process:
        1. Project latent to spatial features
        2. Apply transpose convolutions to upsample
        3. Interpolate to exact input dimensions
        4. Apply sigmoid for [0, 1] output range
        """
        # Project latent back to feature maps
        x = self.fc(z)
        x = x.view(x.size(0), self.encoded_channels, self.encoded_h, self.encoded_w)
        
        # Apply upsampling hierarchy (mirror of encoder)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final_conv(x)
        
        # Adaptive resizing to match exact target dimensions
        # This handles cases where transpose convolutions don't perfectly reverse pooling
        # (e.g., 71 columns -> after pooling/unpooling might become 64 or 72)
        current_h, current_w = x.shape[2], x.shape[3]
        
        if current_h != self.target_h or current_w != self.target_w:
            # Use bilinear interpolation to resize to exact dimensions
            x = torch.nn.functional.interpolate(
                x, 
                size=(self.target_h, self.target_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply sigmoid to constrain output to [0, 1]
        x = self.sigmoid(x)
        return x


class AttentionAggregation(nn.Module):
    """
    Attention mechanism for aggregating multiple timepoint latent vectors.
    
    Design Philosophy - Approach 3 from our discussion:
    - Learns which timepoints are most important for classification
    - Weights (α1, α2, α3) are LEARNED during training, not fixed
    - Allows model to discover temporal patterns in the data
    - Provides interpretability: we can see which timepoints matter most
    
    How it works:
    1. Small neural network scores each timepoint's importance
    2. Softmax ensures weights sum to 1 (valid probability distribution)
    3. Weighted sum creates aggregated representation for classification
    
    Example learned weights:
    - α1=0.7, α2=0.2, α3=0.1: Early timepoint (0h) most discriminative
    - α1=0.1, α2=0.2, α3=0.7: Late timepoint (24h) most discriminative
    - α1=0.33, α2=0.33, α3=0.34: All timepoints equally important
    """
    
    def __init__(self, latent_dim: int):
        """
        Args:
            latent_dim: Dimension of each latent vector
        """
        super(AttentionAggregation, self).__init__()
        
        # Small neural network that learns to score timepoint importance
        # Input: latent vector (latent_dim)
        # Output: single score indicating importance
        self.attention_net = nn.Sequential(
            nn.Linear(latent_dim, 64),  # Project to smaller space
            nn.Tanh(),  # Non-linear activation
            nn.Linear(64, 1)  # Output single importance score
        )
        
    def forward(self, latent_vectors):
        """
        Compute attention-weighted aggregation of timepoint latent vectors.
        
        Args:
            latent_vectors: Tensor of shape (batch, num_timepoints, latent_dim)
                           e.g., (batch, 3, 512) for 3 timepoints
        
        Returns:
            aggregated: Weighted sum of shape (batch, latent_dim)
            attention_weights: Learned weights of shape (batch, num_timepoints)
                              These tell us which timepoints the model considers important
        
        Process:
        1. Compute importance score for each timepoint
        2. Normalize scores with softmax (makes them sum to 1)
        3. Weighted sum: z_agg = α1·z1 + α2·z2 + α3·z3
        """
        # Compute attention scores for each timepoint
        # Shape: (batch, num_timepoints, 1)
        attention_scores = self.attention_net(latent_vectors)
        attention_scores = attention_scores.squeeze(-1)  # Shape: (batch, num_timepoints)
        
        # Apply softmax to get weights that sum to 1
        # This ensures weights form a valid probability distribution
        attention_weights = torch.softmax(attention_scores, dim=1)  # Shape: (batch, num_timepoints)
        
        # Weighted sum of latent vectors
        # Expand weights for broadcasting: (batch, num_timepoints) -> (batch, num_timepoints, 1)
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        
        # Weighted sum: (batch, num_timepoints, latent_dim) -> (batch, latent_dim)
        aggregated = torch.sum(latent_vectors * attention_weights_expanded, dim=1)
        
        return aggregated, attention_weights


class Classifier(nn.Module):
    """
    Classification head for binary group prediction.
    
    Takes the aggregated latent representation and predicts group membership.
    
    Design elements:
    - Fully connected layers for classification
    - Dropout for regularization (prevents overfitting on limited data)
    - Final layer outputs logits for 2 classes
    - Softmax (applied during loss calculation) provides confidence scores
    """
    
    def __init__(self, latent_dim: int, num_classes: int, dropout_rate: float = 0.3):
        """
        Args:
            latent_dim: Dimension of aggregated latent vector
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Probability of dropping neurons during training
        """
        super(Classifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Project to smaller space
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Regularization to prevent overfitting
            nn.Linear(128, num_classes)  # Output logits for each class
        )
        
    def forward(self, z_agg):
        """
        Predict class from aggregated latent representation.
        
        Args:
            z_agg: Aggregated latent vector of shape (batch, latent_dim)
        
        Returns:
            Class logits of shape (batch, num_classes)
        """
        return self.classifier(z_agg)


class ConvAutoencoderWithAttention(nn.Module):
    """
    Complete dual-objective model integrating all components.
    
    ARCHITECTURE OVERVIEW:
    
    Input: 3 images per subject (timepoints t1, t2, t3)
           ↓
    Shared Encoder (processes each timepoint independently)
           ↓
    3 Latent Vectors (z1, z2, z3)
           ↓
           ├─→ Attention Aggregation → z_agg → Classifier → Group Prediction
           │                                                  (SUPERVISED)
           │
           └─→ Individual Decoders → Reconstructed Images
                                      (UNSUPERVISED)
    
    KEY DESIGN DECISIONS:
    
    1. SHARED ENCODER:
       - Same weights process all timepoints
       - Ensures features are comparable across time
       - Critical for attention mechanism to work
    
    2. DUAL OBJECTIVES:
       - Reconstruction: Forces features to be information-rich
       - Classification: Forces features to be discriminative
       - Both objectives share the encoder (joint optimization)
    
    3. ATTENTION AGGREGATION:
       - Collapses 3 latent vectors into 1 for classification
       - Learns which timepoints matter most
       - Provides interpretability
    
    4. SEPARATE RECONSTRUCTION:
       - Each latent vector reconstructs its own timepoint
       - Maintains correspondence: z1→t1, z2→t2, z3→t3
       - Cleaner than trying to reconstruct all from aggregated vector
    
    WHY THIS DESIGN?
    - With limited data (20 subjects), reconstruction provides additional supervision
    - Attention mechanism handles non-independence between timepoints
    - Dual objectives act as mutual regularizers
    - Architecture respects the temporal structure of the data
    """
    
    def __init__(self, in_channels: int, latent_dim: int, num_classes: int):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            latent_dim: Dimension of latent space
            num_classes: Number of output classes (2 for binary)
        """
        super(ConvAutoencoderWithAttention, self).__init__()
        
        self.encoder = ConvEncoder(in_channels, latent_dim)
        self.attention = AttentionAggregation(latent_dim)
        self.decoder = None  # Will be initialized after first forward pass
        self.classifier = Classifier(latent_dim, num_classes)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
    def forward(self, timepoint_images):
        """
        Forward pass through the complete architecture.
        
        Args:
            timepoint_images: Tensor of shape (batch, num_timepoints, channels, height, width)
                             e.g., (4, 3, 1, 512, 71) for batch of 4 subjects
        
        Returns:
            reconstructions: Reconstructed images (batch, num_timepoints, channels, height, width)
            class_logits: Classification predictions (batch, num_classes)
            attention_weights: Learned timepoint importance (batch, num_timepoints)
        
        STEP-BY-STEP PROCESS:
        
        1. ENCODING (creating latent representations):
           - Process each timepoint through shared encoder
           - Produces 3 separate latent vectors per subject
           - Each latent is a compressed representation of one timepoint
        
        2. ATTENTION AGGREGATION (for classification):
           - Combine 3 latent vectors using learned weights
           - Results in single aggregated representation per subject
           - This aggregated vector is used for classification
        
        3. RECONSTRUCTION (for regularization):
           - Each individual latent vector reconstructs its timepoint
           - Ensures encoder preserves information
           - Acts as unsupervised regularization
        
        4. CLASSIFICATION (supervised prediction):
           - Aggregated latent predicts group membership
           - Outputs logits that become probabilities via softmax
        """
        batch_size, num_timepoints = timepoint_images.shape[0], timepoint_images.shape[1]
        
        # STEP 1: ENCODE EACH TIMEPOINT INDEPENDENTLY
        # This maintains consistency: same encoder for all timepoints
        # Critical for attention mechanism to compare timepoints meaningfully
        latent_vectors = []
        for t in range(num_timepoints):
            z_t = self.encoder(timepoint_images[:, t])  # Shape: (batch, latent_dim)
            latent_vectors.append(z_t)
        
        # Initialize decoder after first encoding (now we know dimensions)
        # Dynamic initialization allows model to work with any input size
        if self.decoder is None and self.encoder.encoded_shape is not None:
            self.decoder = ConvDecoder(
                self.latent_dim, 
                self.in_channels, 
                self.encoder.encoded_shape,
                self.encoder.input_shape  # Pass target shape for exact reconstruction
            ).to(timepoint_images.device)
        
        # Stack latent vectors: (batch, num_timepoints, latent_dim)
        # This groups all timepoints for a subject together
        latent_vectors = torch.stack(latent_vectors, dim=1)
        
        # STEP 2: AGGREGATE WITH ATTENTION (for classification branch)
        # Learns which timepoints are most important for distinguishing groups
        # Output is single vector per subject
        z_agg, attention_weights = self.attention(latent_vectors)
        
        # STEP 3: RECONSTRUCT FROM INDIVIDUAL LATENT VECTORS
        # Each timepoint is reconstructed from its own latent representation
        # This is cleaner than reconstructing all from aggregated vector
        # Maintains correspondence: z1→image1, z2→image2, z3→image3
        reconstructions = []
        for t in range(num_timepoints):
            recon_t = self.decoder(latent_vectors[:, t])
            reconstructions.append(recon_t)
        
        # Stack reconstructions: (batch, num_timepoints, channels, height, width)
        reconstructions = torch.stack(reconstructions, dim=1)
        
        # STEP 4: CLASSIFY FROM AGGREGATED REPRESENTATION
        # Uses the attention-weighted combination of all timepoints
        # Outputs logits that become probabilities via softmax
        class_logits = self.classifier(z_agg)
        
        return reconstructions, class_logits, attention_weights


def train_epoch(model, dataloader, optimizer, device, alpha=1.0, beta=1.0):
    """
    Train the model for one epoch.
    
    TRAINING PROCESS:
    1. Forward pass: Compute reconstructions and predictions
    2. Calculate dual losses (reconstruction + classification)
    3. Backward pass: Compute gradients
    4. Update weights via optimizer
    
    Args:
        model: The ConvAutoencoderWithAttention model
        dataloader: Training data loader
        optimizer: Optimizer (e.g., Adam)
        device: CPU or GPU
        alpha: Weight for reconstruction loss (controls importance)
        beta: Weight for classification loss (controls importance)
    
    Returns:
        Average losses and accuracy for the epoch
    
    LOSS FUNCTION DESIGN:
    L_total = α × L_reconstruction + β × L_classification
    
    - L_reconstruction (MSE): Measures how well we reconstruct images
      Purpose: Forces encoder to preserve information, acts as regularizer
    
    - L_classification (CrossEntropy): Measures classification accuracy
      Purpose: Main supervised objective for group prediction
    
    - α and β: Hyperparameters that balance the two objectives
      Equal weights (α=β=1.0) gives equal importance to both tasks
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    # Loss functions for dual objectives
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    all_attention_weights = []
    
    # Disable gradient computation for validation (saves memory and computation)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass only (no backward pass during validation)
            reconstructions, class_logits, attention_weights = model(images)
            
            # Calculate losses (same as training, but no backprop)
            recon_loss = reconstruction_criterion(reconstructions, images)
            class_loss = classification_criterion(class_logits, labels)
            loss = alpha * recon_loss + beta * class_loss
            
            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect attention weights for analysis
            # These tell us which timepoints the model finds most important
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_recon_loss, avg_class_loss, accuracy, np.concatenate(all_attention_weights)


def main():
    """
    Main training script with subject-level cross-validation.
    
    EXPERIMENTAL DESIGN:
    
    1. DATA STRUCTURE:
       - 20 subjects total
       - 2 groups (binary classification)
       - 3 timepoints per subject (0h, 6h, 24h)
       - Each timepoint is an Excel file with spectroscopy data
    
    2. CROSS-VALIDATION STRATEGY (Critical for preventing data leakage):
       - Subject-level 5-fold CV
       - All 3 timepoints from a subject stay together in same fold
       - Never split a subject's timepoints between train and validation
       - This prevents the model from "memorizing" subject-specific patterns
    
    3. TRAINING APPROACH:
       - Dual-objective optimization (reconstruction + classification)
       - Attention mechanism learns timepoint importance
       - Adam optimizer for adaptive learning rates
       - Early stopping based on validation accuracy
    
    4. EVALUATION:
       - Track both reconstruction quality and classification accuracy
       - Monitor attention weights to understand temporal patterns
       - Report cross-validation performance (mean ± std across folds)
    """
    
    # ========== CONFIGURATION ==========
    LABELS_CSV = 'sample_labels.csv'
    TIMEPOINT_DIRS = ['out_0h', 'out_6h', 'out_24h']  # Directories for each timepoint
    LATENT_DIM = 512  # Dimension of compressed latent representation
    BATCH_SIZE = 4    # Small batch size appropriate for limited data (20 subjects)
    NUM_EPOCHS = 100  # Maximum training epochs
    LEARNING_RATE = 0.001  # Adam optimizer learning rate
    
    # LOSS WEIGHTS (hyperparameters for dual objectives)
    ALPHA = 1.0  # Reconstruction loss weight
    BETA = 1.0   # Classification loss weight
    # Equal weights (1.0, 1.0) give equal importance to both objectives
    # Could adjust if one objective is more important than the other
    
    N_SPLITS = 5  # Number of cross-validation folds
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========== LOAD DATA ==========
    # Load subject codes and group labels from CSV
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(labels_df)} samples")
    print(f"Groups: {labels_df['group'].unique()}")
    
    # Encode string labels to integers (0, 1)
    label_encoder = LabelEncoder()
    labels_df['encoded_group'] = label_encoder.fit_transform(labels_df['group'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Prepare data for cross-validation
    codes = labels_df['code'].astype(str).values
    labels = labels_df['encoded_group'].values
    
    # ========== SUBJECT-LEVEL CROSS-VALIDATION ==========
    # Critical: Stratified K-Fold ensures balanced class distribution in each fold
    # shuffle=True randomizes the split
    # random_state=42 ensures reproducibility
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_results = []
    
    print("\n" + "="*70)
    print("STARTING SUBJECT-LEVEL CROSS-VALIDATION")
    print(f"Total subjects: {len(codes)}")
    print(f"Timepoints per subject: {len(TIMEPOINT_DIRS)}")
    print(f"Total images: {len(codes) * len(TIMEPOINT_DIRS)}")
    print("="*70)
    
    # Iterate through cross-validation folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(codes, labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{N_SPLITS}")
        print(f"{'='*70}")
        
        # ========== SPLIT DATA (SUBJECT-LEVEL) ==========
        # Important: We split by subject, not by individual images
        # This ensures all timepoints from a subject stay together
        train_codes = codes[train_idx]
        train_labels = labels[train_idx]
        val_codes = codes[val_idx]
        val_labels = labels[val_idx]
        
        print(f"Training subjects: {len(train_codes)}")
        print(f"Validation subjects: {len(val_codes)}")
        print(f"Training images: {len(train_codes) * len(TIMEPOINT_DIRS)}")
        print(f"Validation images: {len(val_codes) * len(TIMEPOINT_DIRS)}")
        
        # ========== CREATE DATASETS ==========
        # Each dataset handles loading all 3 timepoints per subject
        train_dataset = ExcelImageDataset(train_codes, train_labels, TIMEPOINT_DIRS)
        val_dataset = ExcelImageDataset(val_codes, val_labels, TIMEPOINT_DIRS)
        
        # ========== CREATE DATALOADERS ==========
        # shuffle=True for training (improves generalization)
        # shuffle=False for validation (consistent evaluation)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # ========== GET IMAGE DIMENSIONS ==========
        # Load first batch to determine dimensions dynamically
        # This allows the model to work with any Excel file dimensions
        sample_images, _ = next(iter(train_loader))
        num_timepoints, in_channels, height, width = sample_images.shape[1:]
        print(f"\nData dimensions:")
        print(f"  Timepoints: {num_timepoints}")
        print(f"  Channels: {in_channels}")
        print(f"  Height: {height} (rows in Excel after removing header)")
        print(f"  Width: {width} (columns in Excel after removing first column)")
        
        # ========== INITIALIZE MODEL ==========
        # Model is initialized fresh for each fold
        # This ensures no information leakage between folds
        model = ConvAutoencoderWithAttention(
            in_channels=in_channels,
            latent_dim=LATENT_DIM,
            num_classes=num_classes
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        # ========== OPTIMIZER ==========
        # Adam optimizer with adaptive learning rates
        # Works well for small datasets and handles sparse gradients
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # ========== TRAINING LOOP ==========
        best_val_acc = 0
        train_history = {'loss': [], 'recon_loss': [], 'class_loss': [], 'accuracy': []}
        val_history = {'loss': [], 'recon_loss': [], 'class_loss': [], 'accuracy': []}
        
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        print(f"Loss weights: α={ALPHA} (reconstruction), β={BETA} (classification)")
        
        for epoch in range(NUM_EPOCHS):
            # TRAIN for one epoch
            train_loss, train_recon, train_class, train_acc = train_epoch(
                model, train_loader, optimizer, device, ALPHA, BETA
            )
            
            # VALIDATE on held-out subjects
            val_loss, val_recon, val_class, val_acc, attention_weights = validate(
                model, val_loader, device, ALPHA, BETA
            )
            
            # Store history for plotting/analysis
            train_history['loss'].append(train_loss)
            train_history['recon_loss'].append(train_recon)
            train_history['class_loss'].append(train_class)
            train_history['accuracy'].append(train_acc)
            
            val_history['loss'].append(val_loss)
            val_history['recon_loss'].append(val_recon)
            val_history['class_loss'].append(val_class)
            val_history['accuracy'].append(val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
                print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, "
                      f"Class: {train_class:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, "
                      f"Class: {val_class:.4f}, Acc: {val_acc:.2f}%")
                
                # ATTENTION WEIGHT ANALYSIS
                # Shows which timepoints the model considers most important
                print(f"  Attention weights (mean across validation subjects):")
                print(f"    Timepoint 0h: {attention_weights[:, 0].mean():.3f}")
                print(f"    Timepoint 6h: {attention_weights[:, 1].mean():.3f}")
                print(f"    Timepoint 24h: {attention_weights[:, 2].mean():.3f}")
                # Interpretation:
                # - Higher weight = more important for classification
                # - Weights sum to 1.0 for each subject
            
            # SAVE BEST MODEL (based on validation accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')
                print(f"  ✓ New best model saved (val acc: {val_acc:.2f}%)")
        
        # Store results for this fold
        fold_results.append({
            'fold': fold + 1,
            'best_val_accuracy': best_val_acc,
            'train_history': train_history,
            'val_history': val_history
        })
        
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1} COMPLETE")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*70}")
    
    # ========== FINAL RESULTS (ACROSS ALL FOLDS) ==========
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    # Calculate mean and standard deviation across folds
    # This gives us a robust estimate of model performance
    all_accuracies = [r['best_val_accuracy'] for r in fold_results]
    avg_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    
    print(f"\nOverall Performance:")
    print(f"  Average Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%")
    print(f"\nIndividual Fold Results:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_accuracy']:.2f}%")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # ========== INTERPRETATION GUIDE ==========
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
The model has been trained with dual objectives:

1. RECONSTRUCTION OBJECTIVE (unsupervised):
   - Forces encoder to learn information-rich features
   - Acts as regularization with limited data
   - Low reconstruction loss = encoder preserves image information

2. CLASSIFICATION OBJECTIVE (supervised):
   - Learns to distinguish between two groups
   - High accuracy = model found discriminative patterns
   - Provides the main performance metric

3. ATTENTION WEIGHTS (interpretability):
   - Show which timepoints are most important
   - High weight on timepoint X = that measurement is more discriminative
   - Can provide biological/clinical insights

4. CROSS-VALIDATION PERFORMANCE:
   - Mean ± std gives robust performance estimate
   - Low std = consistent performance across folds
   - High std = performance varies (may indicate small sample size)

NEXT STEPS:
- Examine attention weights to understand temporal patterns
- Visualize reconstructions to verify feature learning
- Analyze which subjects are misclassified
- Consider adjusting α/β if one objective dominates
    """)


if __name__ == "__main__":
    main()  # Mean Squared Error for reconstruction
    classification_criterion = nn.CrossEntropyLoss()  # Cross-entropy for classification
    
    for images, labels in dataloader:
        # Move data to device (GPU if available)
        images = images.to(device)  # Shape: (batch, 3, 1, H, W)
        labels = labels.to(device)  # Shape: (batch,)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # FORWARD PASS through complete architecture
        reconstructions, class_logits, attention_weights = model(images)
        
        # RECONSTRUCTION LOSS (unsupervised objective)
        # Measures difference between original and reconstructed images
        # Forces encoder to learn information-preserving features
        recon_loss = reconstruction_criterion(reconstructions, images)
        
        # CLASSIFICATION LOSS (supervised objective)
        # Measures accuracy of group predictions
        # Forces encoder to learn discriminative features
        class_loss = classification_criterion(class_logits, labels)
        
        # COMBINED LOSS (weighted sum of both objectives)
        # This is the key to dual-objective learning
        # Both objectives share gradients through the encoder
        loss = alpha * recon_loss + beta * class_loss
        
        # BACKWARD PASS (compute gradients)
        loss.backward()
        
        # UPDATE WEIGHTS using optimizer
        optimizer.step()
        
        # Track metrics for monitoring training progress
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_class_loss += class_loss.item()
        
        # Calculate classification accuracy
        _, predicted = torch.max(class_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate average metrics over epoch
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_class_loss = total_class_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_recon_loss, avg_class_loss, accuracy


def validate(model, dataloader, device, alpha=1.0, beta=1.0):
    """
    Validate the model (same as training but without gradient updates).
    
    Purpose:
    - Evaluate performance on held-out validation data
    - Check for overfitting (validation performance vs training performance)
    - Monitor attention weights to see which timepoints model considers important
    
    Returns:
        Validation metrics and attention weights for analysis
    """
    model.eval()  # Set to evaluation mode (disables dropout, uses running batch norm stats)
    
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    reconstruction_criterion = nn.MSELoss()