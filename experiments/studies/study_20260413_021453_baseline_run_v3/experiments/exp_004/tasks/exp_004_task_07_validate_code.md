# Task exp_004_task_07_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 03:04:45.690135+00:00
- **Completed:** 2026-04-13 03:04:45.695286+00:00

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

# === Hyperparameters from the proposal ===
# Note: lr is handled by the skeleton's LR=1e-3, so we ignore the 0.001 proposal value.
# We must read EPOCHS from the environment variable.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Update augmentation based on proposal: time_shift=false, noise_injection=false, mixup=0.0, specaugment=true
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """A simple residual block structure for feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        
        # Handle dimension mismatch for the shortcut path
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out += residual
        return out

class DeepCNNResidual(nn.Module):
    """
    5-block residual CNN stack (32->64->128->128->256 channels) followed by GlobalAvgPool.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Block 1: 1 -> 32
        self.block1 = ResidualBlock(1, 32)
        
        # Block 2: 32 -> 64
        self.block2 = ResidualBlock(32, 64)
        
        # Block 3: 64 -> 128
        self.block3 = ResidualBlock(64, 128)
        
        # Block 4: 128 -> 128 (maintaining channels)
        self.block4 = ResidualBlock(128, 128)
        
        # Block 5: 128 -> 256
        self.block5 = ResidualBlock(128, 256)
        
        # Global pooling to flatten spatial dimension
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head using LazyLinear for dynamic input size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. Feature extraction through 5 blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # 2. Global Pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        
        # 3. Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        
        # 4. Classification
        logits = self.head(x)
        return logits

# ===========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Load data using the specified augmentation settings
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model.
        model = DeepCNNResidual(num_classes=num_classes)
        model = model.to(device)

        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input matching the expected shape (B=1, C=1, H=128, W=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log progress every 10% or at the end of the epoch
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
                # Explicitly delete tensors to help manage memory on constrained devices like MPS
                del x, y, logits, loss 

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Compute sigmoid, move to CPU, convert to numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
                    # Explicitly delete tensors to help manage memory on constrained devices like MPS
                    del x, y
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)
            
            # Manually clear the list contents to allow memory freeing after concatenation
            all_probs.clear()
            all_targs.clear()
            del probs, targs

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Using average_precision_score as a proxy for the competition metric
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )
            
            # Explicitly clear epoch scope variables
            del epoch_losses
            
        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 230: '{' was never closed
