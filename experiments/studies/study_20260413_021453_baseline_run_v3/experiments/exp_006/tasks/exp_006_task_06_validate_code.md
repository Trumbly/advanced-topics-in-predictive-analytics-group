# Task exp_006_task_06_validate_code

- **Experiment:** exp_006
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 03:37:15.446605+00:00
- **Completed:** 2026-04-13 03:37:15.450627+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# Note: We override the skeleton's LR for consistency with the proposal (0.001)
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT_RATE = 0.15
# Epochs must read from env var
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
AUGMENTATION = {"time_shift": True, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class MobileNetV3SpectroModel(nn.Module):
    """
    MobileNetV3-Small backbone adapted for 1-channel spectrogram input,
    followed by global pooling and a classification head.
    """
    def __init__(self, num_classes, dropout_rate=0.15):
        super().__init__()
        
        # 1. Load the standard MobileNetV3 backbone
        # We use V3-Small as specified in the proposal.
        self.backbone = models.mobilenet_v3_small(pretrained=False)
        
        # 2. Adapt the input layer: Change input channels from 3 to 1
        # The first layer is typically nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        original_conv = self.backbone.features[0]
        
        # FIX: The error occurs because the first layer might be a specialized module
        # (like Conv2dNormActivation) which does not expose standard attributes like .kernel_size.
        # We must check if the layer supports direct attribute access, and if not,
        # we must adapt the structure or rely on the known structure/behavior.
        # Given the error points to 'kernel_size', we assume the replacement layer
        # must mimic the structure of the original module's attributes.
        if hasattr(original_conv, 'out_channels'):
            self.backbone.features[0] = nn.Conv2d(
                in_channels=1, out_channels=original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, stride=original_conv.stride, 
                padding=original_conv.padding, bias=original_conv.bias is not None
            )
        else:
            # Fallback for modules that don't expose standard attributes (e.g., Conv2dNormActivation)
            # We use the known parameters for MobileNetV3-Small's first conv layer (3x3, stride 2)
            self.backbone.features[0] = nn.Conv2d(
                in_channels=1, out_channels=original_conv.out_channels, 
                kernel_size=3, stride=2, padding=1, bias=True
            )
        
        # 3. The rest of the backbone layers remain the same.
        
        # 4. Global Pooling and Classification Head
        # Use AdaptiveAvgPool2d to collapse the spatial dimensions (H, W) to (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use LazyLinear to handle the dynamically sized input from the backbone
        self.head = nn.LazyLinear(num_classes)
        
        # Apply dropout after the backbone features for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Pass through the adapted backbone
        x = self.backbone(x)
        # Pool to (B, C, 1, 1)
        x = self.global_pool(x)
        # Flatten to (B, C)
        x = x.flatten(1)
        # Apply dropout
        x = self.dropout(x)
        # Final classification logits
        logits = self.head(x)
        return logits


# ============================================================================
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
        # load_precomputed_dataset handles the actual data loading based on env vars
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
        model = MobileNetV3SpectroModel(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, M=128, T=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # --- Training Setup ---
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # Per-class pos_weight from the DatasetProfile — critical
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
                
                # Forward pass: logits (B, C)
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
                    # Get raw logits, apply sigmoid for probabilities, move to CPU/numpy
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
                    # Note: For the competition metric, we use average_precision_score 
                    # as a stand-in for the per-class AP/MAP calculation.
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
- **code_bytes:** 10041
