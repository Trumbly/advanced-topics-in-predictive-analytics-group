# Task exp_001_task_07_execute_training

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-12 07:40:19.635977+00:00
- **Completed:** 2026-04-12 07:43:01.775927+00:00

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

# === Hyperparameters from the proposal ===
# EPOCHS is hardcoded to 1 as per instructions.
EPOCHS = 1
LR = 0.001
WEIGHT_DECAY = 0.0
DROPOUT_RATE = 0.1
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnSmallV1(nn.Module):
    """
    Custom CNN architecture inspired by cnn_small_v1 baseline,
    adapted for (B, 1, C, T) input shape.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Block 1: Input channels = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: Input channels = 32 (from conv1 out)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: Input channels = 64 (from conv2 out)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classifier Head: Adaptive pooling followed by Lazy Linear
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Feature pooling and flattening
        x = self.pool(x)
        x = x.flatten(1) # (B, C, 1, 1) -> (B, C)
        
        # Dropout and Head
        x = self.dropout(x)
        logits = self.head(x)
        return logits

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size, num_workers etc. from env vars
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
        model = CnnSmallV1(num_classes=num_classes)
        model = model.to(device)
        
        # FIX: Initialize LazyLinear parameters before calculating total parameters
        # by running a dummy forward pass.
        dummy_input = torch.randn(1, 1, 32, 224).to(device, dtype=torch.float32)
        _ = model(dummy_input)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup using weight_decay from proposal
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # Per-class pos_weight from the DatasetProfile
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
                    # Calculate probabilities and move to CPU/Numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                # Check if there are any positive labels in this class across the validation set
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        # Handle case where all labels for a class are 0 or 1 (not possible in this setup, but safe)
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
- **duration_seconds:** 162.13861445803195
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_093854_ui_study_start_test/exp_001
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_093854_ui_study_start_test/exp_001/results.json
- **timed_out:** False
