# Task exp_010_task_18_validate_code

- **Experiment:** exp_010
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 04:54:10.421584+00:00
- **Completed:** 2026-04-13 04:54:10.425538+00:00

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
# Note: The skeleton provided default values, but the proposal overwrites
# augmentation settings. We must use the proposal's settings for augmentation
# but keep the structure of the other constants.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {
    "time_shift": False,
    "noise_injection": True,
    "mixup": 0.2,
    "specaugment": True
}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """A simple residual block for spectrogram feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        
        # First convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # The error indicates a size mismatch during addition.
        # When adding residual, the dimensions must match.
        # If out_channels != in_channels, we must project the residual path.
        if out.shape != residual.shape:
            # Project the residual path to match the output of the block
            residual = nn.Conv2d(residual.shape[1], out.shape[1], kernel_size=1, padding=0)(residual)
        
        return out + residual

class DeepCnnModel(nn.Module):
    """
    5-block residual CNN stack (32->64->128 channels) with BatchNorm 
    and Global AvgPool head.
    Input: (B, 1, C_in, T)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Initial convolution: Must take 1 input channel
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Stacking 5 blocks, increasing channels: 32 -> 64 -> 128
        # Block 1: 32 -> 32
        self.block1 = ResidualBlock(in_channels=32, out_channels=32)
        
        # Block 2: 32 -> 64 (Channel change: 32 -> 64)
        self.block2 = ResidualBlock(in_channels=32, out_channels=64)
        
        # Block 3: 64 -> 64
        self.block3 = ResidualBlock(in_channels=64, out_channels=64)
        
        # Block 4: 64 -> 128 (Channel change: 64 -> 128)
        self.block4 = ResidualBlock(in_channels=64, out_channels=128)
        
        # Block 5: 128 -> 128
        self.block5 = ResidualBlock(in_channels=128, out_channels=128)

        # Global pooling to collapse (T) dimension
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head: Use LazyLinear for robustness against input size changes
        self.head = nn.LazyLinear(self.num_classes)

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        
        # Stacking blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Global pooling and flattening
        x = self.pool(x)
        x = x.flatten(1)   # (B, C, 1, 1) -> (B, C)
        
        # Final classification
        return self.head(x)


# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ===============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset handles augmentation arguments passed here
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model and move to device
        model = DeepCnnModel(num_classes=num_classes).to(device)
        
        # Fix: Perform dummy forward pass to initialize LazyLinear parameters 
        # and ensure all components are on the correct device type (float32).
        # Input shape must match the expected tensor shape: (B, 1, C_in, T)
        dummy = torch.zeros(1, 1, 128, 313, device=device)
        with torch.no_grad():
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
                    # Compute probabilities, move to CPU, then numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

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

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **validation:** passed
- **code_bytes:** 10024
