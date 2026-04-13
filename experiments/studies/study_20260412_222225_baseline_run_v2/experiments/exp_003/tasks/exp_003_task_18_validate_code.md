# Task exp_003_task_18_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 22:54:40.028589+00:00
- **Completed:** 2026-04-12 22:54:40.032107+00:00

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

# === Hyperparameters (using explicit constants as per original structure) ===
EPOCHS = 1
LR = 1e-3
# Note: The original code used AUGMENTATION dictionary, keeping it for compatibility
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class AttentionBlock(nn.Module):
    """
    Implements a simplified self-attention pooling block.
    Takes features (B, C, H, W), pools globally, and then applies
    a small attention mechanism (e.g., a simple multiplicative gate)
    before the final linear layer.
    """
    def __init__(self, channel_dim):
        super().__init__()
        self.channel_dim = channel_dim
        # Global pooling collapses (H, W) -> (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple attention mechanism: 1x1 convolution on the pooled feature
        # to generate weights, followed by a sigmoid activation.
        # We use a single output channel for the attention weight vector.
        self.attention_conv = nn.Conv2d(channel_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Pooling: (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Attention Weight Generation: (B, 1, 1, 1)
        attn_weights_raw = self.attention_conv(pooled)
        attn_weights = self.sigmoid(attn_weights_raw)
        
        # 3. Apply weights: (B, C, 1, 1) * (B, 1, 1, 1) -> (B, C, 1, 1)
        # We broadcast the weight across all C channels.
        weighted_features = pooled * attn_weights
        
        # The output is now a feature vector (B, C, 1, 1) ready for the linear head
        return weighted_features


class CnnAttentionModel(nn.Module):
    """
    3-stage approach: 
    1. Small 3-conv CNN front-end (extract local features) 
    2. Self-Attention Pooling Block (aggregate global context) 
    3. Linear head for multi-label output
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. Small 3-conv CNN front-end ---
        # Input channels = 1 (single spectrogram channel)
        # Padding=1 and kernel=3 preserves spatial dimensions H and W.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # out: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # out: 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # out: 128
        
        # --- 2. Self-Attention Pooling Block ---
        # The feature dimension exiting the CNN stack is 128 channels.
        self.attention_block = AttentionBlock(channel_dim=128)
        
        # --- 3. Linear Head ---
        # Use LazyLinear to handle input feature size inferred from the 
        # feature map after the attention block and pooling.
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        
        # Stage 1: CNN Feature Extraction
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) # Output shape: (B, 128, H_out, W_out)

        # Stage 2: Attention Pooling
        x = self.attention_block(x) # Output shape: (B, 128, 1, 1)

        # Squeeze the spatial dimensions (1, 1) and flatten to (B, 128)
        x = x.view(x.size(0), -1) # Flatten (B, C, 1, 1) -> (B, C)

        # Stage 3: Linear Head
        logits = self.head(x) # (B, num_classes)
        return logits


# ==============================================================================
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
        # load_precomputed_dataset populates num_classes correctly.
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
        model = CnnAttentionModel(num_classes=num_classes)
        
        # FIX: Initialize LazyLinear parameters before calling sum(p.numel())
        # by running a dummy forward pass.
        dummy_input = torch.randn(1, 1, 224, 224).to(device, dtype=torch.float32)
        with torch.no_grad():
            _ = model(dummy_input)
            
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log progress every 10% or on the last batch
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
                    # Compute probabilities and move results back to CPU/Numpy
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
                    # Using average_precision_score as a proxy for AP@k mean
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

        final_auc = curves["roc_auc_macro"][-1]
        final_cmap5 = curves["cmap_at_5"][-1]
        final_f1 = curves["f1_macro"][-1]
        final_loss = curves["loss"][-1]

        results = {
            "metrics": {
                "roc_auc_macro": final_auc,
                "cmap_at_5": final_cmap5,
                "f1_macro": final_f1,
                "loss": final_loss,
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_auc:.4f} "
            f"cmap@5={final_cmap5:.4f} "
            f"f1={final_f1:.4f}",
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
- **code_bytes:** 10133
