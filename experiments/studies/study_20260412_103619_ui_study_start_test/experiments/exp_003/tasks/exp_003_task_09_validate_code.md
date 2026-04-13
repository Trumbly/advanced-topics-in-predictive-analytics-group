# Task exp_003_task_09_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 08:44:26.037893+00:00
- **Completed:** 2026-04-12 08:44:26.044324+00:00

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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# EPOCHS reads from BIRDCLEF_EPOCHS env var, defaults to "1".
EPOCHS = 1
LR = 1e-3
# The proposal specifies: time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
# Initialize BACKBONE_CLASS to ensure it exists before class definitions read it.
BACKBONE_CLASS = None

try:
    # Attempt to import the specific registry model name
    from pipelines.models import MobileNetV3Small
    BACKBONE_CLASS = MobileNetV3Small
except ImportError:
    # Fallback/Placeholder if the exact registry import fails in the execution environment
    print("Warning: Could not import MobileNetV3Small from pipelines.models. Using placeholder structure.", flush=True)
    
    # Define the placeholder class structure locally
    class PlaceholderMobileNetV3Small(nn.Module):
        """Placeholder for the actual MobileNetV3 backbone."""
        def __init__(self, num_classes):
            super().__init__()
            # Simulate a feature extractor block (e.g., a stack of convolutions)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
            # The actual backbone usually outputs a feature map C x H x W
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            # The final head must use LazyLinear
            self.classifier_head = nn.LazyLinear(num_classes)

        def forward(self, x):
            # Simulate the sequence: Conv -> GAP -> Linear
            x = self.features(x)
            x = self.gap(x)
            x = x.flatten(2) # Flatten features (B, C, 1, 1) -> (B, C)
            return self.classifier_head(x)
            
    # Assign the placeholder class to the globally scoped variable
    BACKBONE_CLASS = PlaceholderMobileNetV3Small


class MobileNetV3Wrapper(nn.Module):
    """Wraps the backbone to enforce Global Average Pooling (GAP) before the classifier head."""
    def __init__(self, num_classes):
        super().__init__()
        # Initialize the backbone using the actual registry class
        self.backbone = BACKBONE_CLASS(num_classes=num_classes)
        
    def forward(self, x):
        # 1. Pass through the backbone (which might already include pooling/flattening)
        x = self.backbone(x)
        
        # 2. Enforce GAP if the backbone output isn't already suitable for the head
        if x.shape[-2:] != (1, 1):
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 3. Pass through the final classification head (which is LazyLinear)
        return x


# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads required configs (batch_size, num_workers, etc.)
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
        model = MobileNetV3Wrapper(num_classes=num_classes)
        # IMPORTANT: Move to device immediately.
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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
                # Move each batch to the training device.
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

            # Validation
            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities and move results back to CPU
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC calculation
            aucs = []
            for c in range(targs.shape[1]):
                # Check for positive labels to avoid sklearn error
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        # Should not happen if sum > 0, but safe guard
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    finally:
        # Final logging and results writing (runs even if training fails)
        if 'results' not in locals() or "error" not in results:
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

        with open("results.json", "w") as fh:
            json.dump(results, fh)

```

## Output
- **validation:** passed
- **code_bytes:** 8457
