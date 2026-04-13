# Task exp_006_task_07_execute_training

- **Experiment:** exp_006
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-12 07:53:41.708503+00:00
- **Completed:** 2026-04-12 07:53:43.760613+00:00

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
from pipelines.models import TorchvisionAdapter

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# The proposal specifies LR=0.0005, weight_decay=0.0001, dropout=0.2
# We must override the default LR and use weight_decay.
EPOCHS = 1
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
        
        # 1. Backbone Initialization using TorchvisionAdapter
        # We use the adapter as per instructions to fix the ImportError.
        # We initialize it with num_classes=None initially, as the adapter is for the backbone.
        # We assume the adapter handles setting up the backbone correctly for 1 channel input
        # if the backbone is designed for 3 channels (it should handle the required adaptation).
        self.backbone = TorchvisionAdapter("mobilenet_v3_small", num_classes=None)
        
        # NOTE: The adapter handles the backbone structure. We must ensure the output channels
        # of the backbone match the SEBlock input (assumed 128) and the head input.
        
        # 2. SE Refinement Block
        # We keep the placeholder channel size, assuming the backbone output matches this.
        self.se_block = SEBlock(channel=128) 
        
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
- **exit_code:** 0
- **duration_seconds:** 2.0506541669601575
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_093854_ui_study_start_test/exp_006
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_093854_ui_study_start_test/exp_006/results.json
- **timed_out:** False
