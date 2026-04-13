# Task exp_007_task_12_validate_code

- **Experiment:** exp_007
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 03:54:08.808340+00:00
- **Completed:** 2026-04-13 03:54:08.815274+00:00

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

# === Hardcoded hyperparameters from the proposal (NOT a hyperparams dict) ===
# LR is set from the proposal: 0.0008
LR = 0.0008
# Dropout is set from the proposal: 0.25
DROPOUT_RATE = 0.25
# EPOCHS is read from an env var.
EPOCHS = 1
# Augmentation settings are read from the proposal
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A simple residual block for feature extraction in CNN stacks.
    Uses BatchNorm, ReLU, and Dropout, following the functional style (F.relu).
    """
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_channels = out_channels
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut_conv(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        out += residual
        out = F.relu(out)
        return out


class DeepCnnModel(nn.Module):
    """
    Custom 5-block residual CNN stack (32->64->128->256 channels)
    for the BirdCLEF task.
    """
    def __init__(self, num_classes, dropout_rate):
        super().__init__()
        
        # 1. Initial Feature Extraction (Input: 1 channel)
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        
        # 2. 5 Residual Blocks (Channel progression: 32 -> 64 -> 128 -> 256 channels)
        self.block1 = ResidualBlock(32, 32, dropout_rate)      # 32 -> 32
        self.block2 = ResidualBlock(32, 64, dropout_rate)      # 32 -> 64
        self.block3 = ResidualBlock(64, 128, dropout_rate)     # 64 -> 128
        self.block4 = ResidualBlock(128, 256, dropout_rate)    # 128 -> 256
        self.block5 = ResidualBlock(256, 256, dropout_rate)    # 256 -> 256 (Final block maintains max channels)
        
        # 3. Global Pooling and Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle variable spatial dimensions robustly
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Initial Pass
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Stacked Residual Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Pooling
        x = self.pool(x)
        
        # Flatten and Classify
        x = x.flatten(1) # Collapse (B, C, 1, 1) -> (B, C)
        return self.head(x)

# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / persistent_workers / prefetch_factor
        # are read from BIRDCLEF_* env vars (sourced from config.yaml).
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = DeepCnnModel(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Dummy input shape: (Batch=4, Channels=1, Height=128, Width=313)
        with torch.no_grad():
            dummy = torch.zeros(4, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimization setup using proposal LR
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log progress every 10% or at the end of a batch.
        log_every = max(1, n_train_batches // 10) 

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
                    # Compute probabilities via sigmoid, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 183: expected 'except' or 'finally' block
