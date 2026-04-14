# Task exp_004_task_06_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-14 04:27:40.846987+00:00
- **Completed:** 2026-04-14 04:27:40.853855+00:00

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

# === Hardcoded hyperparameters from the proposal ===
# Note: LR and AUGMENTATION are updated based on the JSON proposal.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "7"))
LR = 0.0008  # Overriding skeleton default 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A standard residual block for 2D spectrogram inputs, using BatchNorm and ReLU.
    Handles channel dimension mismatch via projection if needed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. First convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # 2. Second convolution path
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # 3. Dropout and residual connection
        out = self.dropout(out)
        
        # Residual connection: must ensure shapes match for addition
        if out.shape != x.shape:
            # If shapes mismatch, we must project the input 'x' to match 'out' dimensions
            # Since we control the channel progression, the residual connection should ideally
            # happen after appropriate pooling/projection if dimensions change drastically.
            # For this specific fix, we assume the input x needs to be projected if channels mismatch.
            if out.shape[1] != x.shape[1]:
                # Use a simple projection layer for the residual connection if channels change
                projection = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=1, padding=0).to(x.device)
                x = projection(x)
                
        return out + x

class AttentionPooling(nn.Module):
    """
    Global Attention Pooling module.
    Takes (B, C, H, W) -> (B, C) by attending to spatial dimensions.
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # Global Average Pooling is the baseline feature map
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature map size after pooling: (B, C, 1, 1)
        # We concatenate this with the average feature map (B, 1, 1, 1)
        # to create a combined feature vector for attention scoring.
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. Global Average Pooling (Feature map -> (B, C, 1, 1))
        avg = self.avg_pool(x)
        
        # 2. Self-Attention mechanism (using the feature map itself)
        # We average the feature map across spatial dimensions to get the channel mean (B, C, 1, 1)
        attn_map = avg
        
        # Concatenate the two (effectively, we use avg twice, but the conv layer handles the dimension increase)
        # The output of self.attention_conv is (B, 1, 1, 1)
        attn_out = self.attention_conv(torch.cat([attn_map, attn_map], dim=1))
        
        # 3. Apply softmax-like scoring by squeezing the feature map to (B, C)
        # We use the attention output as a weight map and apply it across the channel dimension.
        # Since we are after linear pooling, we simply average the attention output.
        attention_weights = torch.sigmoid(attn_out.view(x.size(0), 1, 1, 1)).squeeze(1) # (B, 1, 1, 1) -> (B, 1)
        
        # Final weighted feature vector: (B, C, 1, 1) * (B, 1, 1, 1)
        # This scales the entire feature map by the computed attention weight.
        output = avg * attention_weights.view(1, -1, 1, 1)
        return output

class ResidualCNN(nn.Module):
    """
    Deep CNN architecture with 4 residual blocks, BN, and Attention Pooling.
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # Initial convolution layer (Input: 1 channel)
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)
        self.relu_init = nn.ReLU(inplace=True)
        
        # 4 Residual Blocks: 1->32 -> 32->64 -> 64->128 -> 128->256
        # Note: The residual connection forces the feature map size to be preserved.
        self.res1 = ResidualBlock(1, 32, dropout_rate=dropout_rate) # Input: 1, Output: 32
        self.res2 = ResidualBlock(32, 64, dropout_rate=dropout_rate) # Input: 32, Output: 64
        self.res3 = ResidualBlock(64, 128, dropout_rate=dropout_rate) # Input: 64, Output: 128
        self.res4 = ResidualBlock(128, 256, dropout_rate=dropout_rate) # Input: 128, Output: 256
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(dropout_rate=dropout_rate)
        
        # Classifier Head: Uses LazyLinear to handle dynamic feature size
        # The input size will be 256 (the output of res4)
        self.classifier_head = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # Pass through residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Attention Pooling
        x = self.attention_pool(x)
        
        # Final projection and flattening: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1) 
        
        # Classifier head
        logits = self.classifier_head(x)
        return logits


# ========================================================================
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
        # Load data using the specified augmentation config
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
        model = ResidualCNN(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Input shape: (B, 1, 128, 313)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        # Calculate and print parameter count
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup: Using specified LR and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        PATIENCE = 3
        best_auc = 0.0
        epochs_no_improve = 0
        
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
                    # Get logits, apply sigmoid, move to CPU, convert to numpy
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

            # --- Metric 2: cmap@5 (average precision) ---
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
            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

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
- **code_bytes:** 13188
