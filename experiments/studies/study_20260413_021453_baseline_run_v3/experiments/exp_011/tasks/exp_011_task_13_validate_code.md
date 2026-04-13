# Task exp_011_task_13_validate_code

- **Experiment:** exp_011
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 05:07:20.926635+00:00
- **Completed:** 2026-04-13 05:07:20.935313+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var, defaulting to 1 for fast iteration.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# =============================================================================
# MODEL DEFINITION (Custom 4-block CNN with Residual Connections)
# =============================================================================

class BasicResidualBlock(nn.Module):
    """A simple residual block for feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        # Handle dimension mismatch for residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
             # Identity shortcut if dimensions match
             self.shortcut = nn.Identity()


    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class SpecAugmentCNN(nn.Module):
    """
    Custom 4-block CNN front-end with residual connections, followed by 
    Global Average Pooling head.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Initial convolution layer (Input: 1 channel)
        self.initial_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(16)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # 4 Residual Blocks, increasing channels: 16 -> 32 -> 64 -> 64 (Reduced from 128 to fix OOM)
        self.block1 = BasicResidualBlock(16, 32)
        self.block2 = BasicResidualBlock(32, 64)
        self.block3 = BasicResidualBlock(64, 128)
        # FIX: Reduced final block channels from 128 to 64 to mitigate OOM error on MPS device
        self.block4 = BasicResidualBlock(128, 64) 

        # Final Head: Global Average Pooling followed by Lazy Linear Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # LazyLinear infers in_features from the first forward pass (now expecting 64 features)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. Initial Feature Extraction
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        # 2. 4 Residual Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # 3. Global Average Pooling
        x = self.pool(x) # Shape: (B, 64, 1, 1)
        
        # 4. Flatten and Classify
        x = x.flatten(1) # Shape: (B, 64)
        return self.head(x)

# =============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset handles batch_size, num_workers, etc.
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here.
        model = SpecAugmentCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (1, 1, 128, 313) -> Adjusted to accommodate 64 features
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                # Move each batch to the training device. Float32 only.
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                    pct = 100.0 * (batch_idx + 1) / n_train_batches
                    print(
                        f"  epoch {epoch + 1} [{pct:5.1f}%] "
                        f"batch {batch_idx + 1}/{n_train_batches} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # .cpu() before .numpy()
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            # Concatenate results from all validation batches
            probs = np.concatenate(all_probs, axis=0)
            targs

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 192: expected 'except' or 'finally' block
