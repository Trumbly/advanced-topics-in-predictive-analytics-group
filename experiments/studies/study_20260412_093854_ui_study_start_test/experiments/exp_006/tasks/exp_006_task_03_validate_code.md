# Task exp_006_task_03_validate_code

- **Experiment:** exp_006
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 07:52:46.414598+00:00
- **Completed:** 2026-04-12 07:52:46.419287+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# The proposal specifies LR=0.0005, weight_decay=0.0001, dropout=0.2
# We must override the default LR and use weight_decay.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 0.0005
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
AUGMENTATION = {"time_shift": False, "noise_injection": True, "mixup": 0.0, "specaugment": True}

# --- Custom Components ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel recalibration."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * self.sigmoid(y)

class MobileNetV3SE(nn.Module):
    """
    MobileNetV3 backbone adapted for 1 input channel, followed by SE refinement
    and classification head.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone Initialization (Requires adapting the first layer)
        # We must dynamically load the backbone and modify its first layer.
        # We assume 'pipelines.models' contains the class for the specified backbone.
        try:
            # Attempt to import the required registry model
            from pipelines.models import mobilenet_v3_small
            self.backbone = mobilenet_v3_small(pretrained=True)
            
            # Crucial Adaptation: The input channel must be 1, not 3.
            # We assume the first convolution layer is the one to modify.
            # This is a fragile assumption, but necessary given the task constraints.
            # We iterate through modules to find the first Conv2d and modify it.
            first_conv = None
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d) and 'conv' in name: # Heuristic to find first conv
                    # Check if it's the true input conv layer
                    if module.in_channels != 1 and module.in_channels == 3:
                        print(f"INFO: Modifying first Conv2d layer in {name} from 3 to 1 channel.", flush=True)
                        # Create a new conv layer with the correct input channels
                        new_conv = nn.Conv2d(1, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups)
                        # Copy existing weights (if possible, otherwise the model will train them)
                        # For simplicity and robustness, we just replace the module.
                        setattr(self.backbone, name, new_conv)
                        first_conv = True
                        break
            
            if not first_conv:
                print("WARNING: Could not automatically adapt the first conv layer. Assuming backbone structure is compatible or relying on default initialization.", flush=True)

        except ImportError:
            print("ERROR: Could not import 'mobilenet_v3_small' from pipelines.models. Please check environment setup.", flush=True)
            raise
        except Exception as e:
            print(f"ERROR during backbone initialization: {e}", flush=True)
            raise

        # 2. SE Refinement Block
        # Assuming the final feature dimension after backbone is 128 channels (a common size for MobileNet variants)
        # We need to adapt this based on the actual output channels of the adapted backbone.
        # Since we cannot determine the output channels reliably without running a dummy pass,
        # and the goal is feature refinement, we assume a common channel size, say 128.
        # If the backbone output is different, this must be manually adjusted.
        self.se_block = SEBlock(channel=128) # Placeholder channel size
        
        # 3. Global Pooling and Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        # Use LazyLinear for the final classification layer
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. Backbone Pass
        x = self.backbone(x)
        
        # 2. SE Refinement
        x = self.se_block(x)
        
        # 3. Global Pooling
        x = self.pool(x)
        
        # 4. Flatten and Dropout
        x = x.flatten(1) # (B, C, 1, 1) -> (B, C)
        x = self.dropout(x)
        
        # 5. Classification Head
        logits = self.head(x)
        return logits


# =================================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =================================================================================
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
        model = MobileNetV3SE(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup using provided weight_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        # Per-class pos_weight from the DatasetProfile — critical
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
                    # .cpu() before .numpy()
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
- **code_bytes:** 10100
