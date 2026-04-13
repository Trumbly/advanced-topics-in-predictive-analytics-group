# Task exp_004_task_07_execute_training

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-12 19:01:22.496544+00:00
- **Completed:** 2026-04-12 19:36:35.877581+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
# The orchestrator resolves `training.device` from config.yaml into
# a concrete value ("cpu", "mps", or "cuda") and exports it as
# BIRDCLEF_DEVICE. Just trust the env var — do NOT call
# torch.cuda.is_available() or torch.backends.mps.is_available().
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
#
# EPOCHS is ALSO read from an env var. During the smoke phase the
# orchestrator sets BIRDCLEF_EPOCHS=1 (fast-iteration mode). During
# the optional PROMOTION phase at the end of a study, the orchestrator
# re-runs top-K smoke-phase experiments with a higher
# BIRDCLEF_EPOCHS value (e.g. 5) to get a realistic final score.
# Your code must ALWAYS read this env var — do NOT hardcode a
# literal `EPOCHS = 1` next to it. The default of 1 keeps fast
# iteration working when no env var is set.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.5, "specaugment": True}

# === Model definition at MODULE scope ===
# If you are using a custom nn.Module, define the CLASS here:
class CNNGRUModel(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        
        # Calculate the flattened feature size after conv layers
        # Input shape: (batch, 1, 128, 313)
        # After conv1: (batch, 64, 128, 313)
        # After pool1: (batch, 64, 64, 157)
        # After conv2: (batch, 128, 64, 157)
        # After pool2: (batch, 128, 32, 79)
        # After conv3: (batch, 256, 32, 79)
        # After pool3: (batch, 256, 16, 40)
        # Flatten to: (batch, 256 * 16 * 40) = (batch, 16384)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, dropout=dropout if 2 > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Apply conv layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Adaptive pooling to get fixed size
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        # Reshape for GRU: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Apply GRU
        gru_out, _ = self.gru(x)
        
        # Use the last time step output
        x = gru_out[:, -1, :]  # (batch, 256)
        
        # Apply dropout and head
        x = self.dropout(x)
        x = self.head(x)
        
        return x

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# Only this guarded block actually trains. Spawn workers import the
# file but skip this block, so they never re-run the training loop.
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
        model = CNNGRUModel(num_classes=num_classes, dropout=0.4)
        model = model.to(device)

        # Initialize LazyLinear (if used) by running ONE dummy batch
        # through the model. Without this, model.parameters() raises
        # "UninitializedParameter" when you call .numel() or pass
        # the parameters to the optimizer.
        first_x, _ = next(iter(train_loader))
        with torch.no_grad():
            model(first_x.to(device, dtype=torch.float32))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        # for the heavy long-tail class imbalance. Capped at 50x.
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
                # Move each batch to the training device. Float32 only —
                # MPS has limited float64 support so do NOT call .double().
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
                    # accept CPU tensors, not MPS/CUDA tensors.
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
            # For each class with positives, take the top-5 predictions by
            # score and compute average precision, then mean across classes.
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
- **exit_code:** 0
- **duration_seconds:** 2113.3649526670342
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_174848_multiple_epochs_test/exp_004
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_174848_multiple_epochs_test/exp_004/results.json
- **timed_out:** False
