# Task exp_007_task_13_validate_code

- **Experiment:** exp_007
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 03:56:34.156971+00:00
- **Completed:** 2026-04-13 03:56:34.158376+00:00

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
# Note: We override the skeleton's default LR and use the proposal's values.
# The skeleton's general structure for EPOCHS must be maintained.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 0.0008  # From proposal: lr=0.0008
WEIGHT_DECAY = 0.0 # From proposal: weight_decay=0.0
DROPOUT = 0.25 # From proposal: dropout=0.25
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A general residual block for spectrogram processing.
    Uses Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU structure.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential() # Placeholder for skip connection

    def forward(self, x):
        identity = x
        
        # Update shortcut if dimensions change
        if identity.shape[1] != self.conv1.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(identity.shape[1], self.conv1.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.conv1.out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.dropout(out)
        
        out += self.shortcut(identity)
        out = self.relu2(out) # Final activation after addition
        return out

class DeepCNN(nn.Module):
    """
    Custom 5-block residual CNN stack (32->64->128->256 channels).
    Uses LazyLinear for the classification head.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_rate = DROPOUT
        
        # Initial projection layer (Input: 1 -> 32)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 5 Residual Blocks following the channel progression: 32 -> 64 -> 128 -> 256
        # Block 1: 32 -> 64
        self.block1 = ResidualBlock(32, 64, dropout_rate=self.dropout_rate)
        # Block 2: 64 -> 128
        self.block2 = ResidualBlock(64, 128, dropout_rate=self.dropout_rate)
        # Block 3: 128 -> 256
        self.block3 = ResidualBlock(128, 256, dropout_rate=self.dropout_rate)
        # Block 4: 256 -> 256 (maintaining depth)
        self.block4 = ResidualBlock(256, 256, dropout_rate=self.dropout_rate)
        # Block 5: 256 -> 256 (maintaining depth)
        self.block5 = ResidualBlock(256, 256, dropout_rate=self.dropout_rate)

        # Global pooling and classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle dynamic feature map size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Sequential application of residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        
        # Classification head
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
        model = DeepCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # This is CRITICAL for parameter counting.
        with torch.no_grad():
            # Dummy input must match the input shape: (Batch, Channel, Mel, Time)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup using proposal LR and Weight Decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
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
                    # `.cpu()` before `.numpy()`
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                # Only calculate if there is at least one positive label in the validation set for this class
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score calculates AP, which is the required metric here.
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
                "loss": curves

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 250: '{' was never closed
