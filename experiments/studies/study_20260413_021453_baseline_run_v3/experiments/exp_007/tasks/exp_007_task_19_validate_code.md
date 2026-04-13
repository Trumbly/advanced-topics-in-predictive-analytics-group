# Task exp_007_task_19_validate_code

- **Experiment:** exp_007
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 04:06:16.650618+00:00
- **Completed:** 2026-04-13 04:06:16.658041+00:00

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
# Overriding default LR from skeleton based on proposal: 0.0008
LR = 0.0008
# Weight decay is 0.0, but Adam accepts it.
WEIGHT_DECAY = 0.0
# Dropout is 0.25
DROPOUT = 0.25

# Read epochs from environment variable
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
AUGMENTATION = {"time_shift": True, "noise_injection": True} # Using only specified ones

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """Standard residual block for feature extraction."""
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shortcut = nn.Sequential()
        
        # Handle dimension mismatch for residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block(x)
        x = self.dropout(x)
        return x + residual

class DeepCNN(nn.Module):
    """
    Custom 5-block residual CNN stack (32->64->128->256 channels).
    Uses LazyLinear for the final classifier head.
    """
    def __init__(self, num_classes, dropout_rate):
        super().__init__()
        
        # Initial convolution layer (Input channels = 1)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        
        # 5 Residual Blocks Stack
        self.blocks = nn.Sequential(
            # Block 1: 32 -> 64
            ResidualBlock(32, 64, dropout_rate), 
            # Block 2: 64 -> 128
            ResidualBlock(64, 128, dropout_rate),
            # Block 3: 128 -> 256
            ResidualBlock(128, 256, dropout_rate),
            # Block 4: 256 -> 256 (Staying at 256)
            ResidualBlock(256, 256, dropout_rate),
            # Block 5: 256 -> 256 (Staying at 256)
            ResidualBlock(256, 256, dropout_rate)
        )
        
        # Final pooling and classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle dynamic feature map size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Pass through residual blocks
        x = self.blocks(x)
        
        # Pool spatial dimensions (H, W) to (1, 1)
        x = self.pool(x)
        
        # Flatten from (B, C, 1, 1) to (B, C)
        x = torch.flatten(x, 2)
        
        # Final linear classification
        return self.head(x)

# ============= END MODULE-SCOPE =============

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # data_loader reads batch_size, num_workers, etc. from env vars
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
        model = DeepCNN(num_classes=num_classes, dropout_rate=DROPOUT)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Dummy input shape matches the expected (B, 1, 128, 313)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup using proposal LR and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # Per-class pos_weight
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
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
                    # Sigmoid for probabilities, move to CPU for sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                # Only calculate if the class has at least one positive label
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score is used as a proxy/approximation for AP@k mean
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
- **code_bytes:** 9551
