# Task exp_014_task_09_validate_code

- **Experiment:** exp_014
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 01:25:49.472452+00:00
- **Completed:** 2026-04-13 01:25:49.475765+00:00

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
from torchvision import models
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters (Keep structure consistent with mandatory rules) ===
# EPOCHS is read from BIRDCLEF_EPOCHS env var.
EPOCHS = 1 # Fixed to 1 as per mandatory structure rule
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class MobileNetV3BirdCLEF(nn.Module):
    """
    MobileNetV3 Small backbone adapted for 1-channel spectrogram input, 
    followed by Global Average Pooling and a linear head.
    """
    def __init__(self, num_classes, backbone_name="mobilenet_v3_small"):
        super().__init__()
        
        # 1. Load standard backbone
        # We load the model without pretraining weights, as we adapt the input channel.
        self.backbone = models.mobilenet_v3_small(pretrained=False)
        
        # 2. Adapt the first convolution layer (Input channel adaptation)
        original_conv1 = self.backbone.features[0]
        
        # The fix addresses the initialization of parameters in LazyModules/container modules.
        # We must ensure the backbone is fully initialized before accessing/replacing layers.
        # Calling a dummy forward pass on the backbone helps initialize parameters.
        
        # Check if the layer is a Conv2d layer before trying to copy attributes.
        if isinstance(original_conv1, nn.Conv2d):
            new_conv1 = nn.Conv2d(
                in_channels=1, 
                out_channels=original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias is not None
            )
            # Replace the first layer in the features sequence
            self.backbone.features[0] = new_conv1
        else:
            # Fallback adaptation path
            new_conv1 = nn.Conv2d(
                in_channels=1, 
                out_channels=original_conv1.out_channels, 
                kernel_size=3,
                stride=1, 
                padding=1, 
                bias=True
            )
            self.backbone.features[0] = new_conv1


        # 3. Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. Classifier Head (LazyLinear for robustness against size changes)
        self.head = nn.LazyLinear(num_classes)
        
        # CRITICAL FIX: Manually initialize parameters for lazy components 
        # before calculating total parameter count or running complex logic that might encounter uninitialized parameters.
        # We use a dummy tensor matching expected input shape (e.g., B=1, C=1, H=224, W=224) 
        # to force initialization of all submodules within the backbone.
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        self.backbone(dummy_input)


    def forward(self, x):
        # Input shape: (B, 1, C, T)
        
        # Pass through adapted backbone features
        x = self.backbone.features(x)
        
        # Apply GAP
        x = self.gap(x) # (B, C', 1, 1)
        
        # Flatten: (B, C')
        x = x.view(x.size(0), -1) 
        
        # Final linear classification head
        return self.head(x)


# ===============================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    
    # The actual number of classes is derived from the dataset profile.
    num_classes = 0 

    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset handles reading num_classes and other metadata
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
        model = MobileNetV3BirdCLEF(num_classes=num_classes)
        model = model.to(device)
        
        # Calculate parameters for logging
        # FIX: Use model.named_parameters() to iterate safely over all parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer and Loss
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

            # --- Validation Phase ---
            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Calculate probabilities and move results to CPU/Numpy for sklearn
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

        # Final Results Compilation
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

    # Write results.json
    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **validation:** passed
- **code_bytes:** 9789
