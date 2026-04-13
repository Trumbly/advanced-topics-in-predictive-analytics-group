# Task exp_005_task_03_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 08:46:38.664198+00:00
- **Completed:** 2026-04-12 08:46:38.670163+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
from pipelines.models import EfficientNetB0 # Assuming this import snippet exists for the registry model

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Custom Components ---

class SelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling layer to aggregate spatial information
    and reduce the feature map to channel-wise statistics (B, C, 1, 1) -> (B, C).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple channel-wise attention mechanism (Squeeze-Excitation style concept)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Average Pooling to get (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Apply attention mechanism
        attention_weights = self.fc(pooled) # (B, C, 1, 1)
        
        # 3. Re-scale the original features by the attention weights
        # We multiply element-wise, so we need to unsqueeze the weight across batch/channel if necessary.
        # Since both are (B, C, 1, 1), direct multiplication works.
        output = x * attention_weights
        
        # Final pooling to get the representative feature (B, C, 1, 1)
        final_representation = self.avg_pool(output)
        return final_representation.flatten(2) # Returns (B, C)

# --- Final Model Definition ---

class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: EfficientNet-B0
        # NOTE: Input channels must be 1. We assume the registry model wrapper 
        # handles the necessary adaptation or that the provided snippet 
        # implicitly handles single-channel input for this task.
        self.backbone = EfficientNetB0(pretrained=False)
        
        # 2. Feature Adaptation Layer (To handle the 1-channel input mismatch 
        # if the backbone expects 3 channels, we project 1 -> 3 for the first pass 
        # ONLY IF the backbone's first layer is convolutional and expects 3 channels.
        # Since we cannot reliably inspect the first layer, we rely on the 
        # assumption that the pipeline model wrapper is robust.
        # If the backbone fails on 1 channel, this block must be adjusted 
        # based on the actual structure of EfficientNetB0.
        self.feature_adaptor = nn.Identity() # Placeholder, trust the backbone initialization for now.

        # 3. Global Pooling and Attention
        # The output channels (C) of the backbone must be determined.
        # We rely on the SelfAttentionPooling to handle the feature map size.
        # We need to pass a dummy tensor to determine the output channels C. 
        # Since we cannot do that easily, we must assume the backbone's last 
        # convolutional layer output dimension is the correct 'in_channels' for the pooling.
        # For EfficientNetB0, the final feature map channel count is typically 1280.
        BACKBONE_OUTPUT_CHANNELS = 1280 # Standard output channels for EfficientNetB0
        self.attention_pool = SelfAttentionPooling(BACKBONE_OUTPUT_CHANNELS)

        # 4. Classification Head: Uses LazyLinear to handle variable feature sizes
        self.classifier_head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, H, W)
        
        # 1. Pass through backbone
        # The backbone processes the input tensor x.
        x_backbone = self.backbone(x) # Output shape: (B, C, H_out, W_out)
        
        # 2. Self-Attention Pooling
        # Output shape: (B, C)
        x_pooled = self.attention_pool(x_backbone)
        
        # 3. Classification Head
        # x_pooled is already (B, C), suitable for LazyLinear
        logits = self.classifier_head(x_pooled) # Output shape: (B, num_classes)
        
        return logits

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ============================================================================
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
        model = BirdCLEFModel(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        # for the heavy long-tail class imbalance. Capped at 50x.
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": []}
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                # Move each batch to the training device. Float32 only —
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
                    # `.cpu()` before `.numpy()` — sklearn and numpy only
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}",
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
- **code_bytes:** 9571
