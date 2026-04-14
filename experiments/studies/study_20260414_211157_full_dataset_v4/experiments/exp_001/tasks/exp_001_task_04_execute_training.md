# Task exp_001_task_04_execute_training

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-14 21:14:17.035497+00:00
- **Completed:** 2026-04-14 22:44:37.026849+00:00

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
# The orchestrator resolves `training.device` from config.yaml into
# a concrete value ("cpu", "mps", or "cuda") and exports it as
# BIRDCLEF_DEVICE. Just trust the env var — do NOT call
# torch.cuda.is_available() or torch.backends.mps.is_available().
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var so the orchestrator can override it.
# Default 4 — early stopping (patience=2) handles convergence.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
LR = 1e-3
# Use the augmentation structure from the proposal
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.0, "specaugment": False}

# === Model definition at MODULE scope ===
from pipelines.models import TorchvisionAdapter

# The proposal uses "efficientnet_b0", which maps directly to the registry model.
# We don't need to define a class, we just need the adapter call in the __main__ block.
# We will use the adapter to wrap the backbone.

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

        # --- Model Instantiation (Using Registry Adapter) ---
        # Backbone: efficientnet_b0
        # We must use TorchvisionAdapter for 1-channel input.
        # The adapter handles the 1->3 channel expansion and resize-to-224.
        
        # Initialize the adapter which wraps the backbone
        backbone = TorchvisionAdapter("efficientnet_b0", num_classes=num_classes)
        
        # --- Custom Head Adaptation ---
        # The EfficientNet-B0 output must be adapted to (B, num_classes).
        # The adapter outputs (B, C, H, W) where H=W=1. We need to flatten and classify.
        
        class EfficientNetHead(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                # Use LazyLinear for the final classification head,
                # as the input size (C) is determined by the backbone's output channels.
                self.head = nn.LazyLinear(num_classes)

            def forward(self, x):
                # 1. Pass through backbone (results in B, C, 1, 1)
                x = self.backbone(x)
                # 2. Flatten spatial dimensions: (B, C, 1, 1) -> (B, C)
                x = x.flatten(1)
                # 3. Pass through the final head
                return self.head(x)

        model = EfficientNetHead(backbone=backbone)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # The dummy input must match the expected shape (B, 1, 128, 313).
        with torch.no_grad():
            # The input size is fixed at (1, 128, 313) based on the loader's default shape.
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
        PATIENCE = 2
        best_f1 = 0.0
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
                    # `.cpu()` before `.numpy()`
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

            # --- Metric 3: Macro F1 with per-class threshold optimization ---
            best_thresholds = np.full(targs.shape[1], 0.5)
            for c in range(targs.shape[1]):
                if targs[:, c].sum() == 0:
                    continue
                best_f1_c = 0.0
                for thr in np.arange(0.05, 1.0, 0.05):
                    preds_c = (probs[:, c] >= thr).astype(np.float32)
                    f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                    if f1_c > best_f1_c:
                        best_f1_c = f1_c
                        best_thresholds[c] = thr
            preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["f1_macro"].append(val_f1)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                flush=True,
            )
            scheduler.step()

            # Early stopping on F1 (primary metric)
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                # Save best model weights + thresholds for submission
                torch.save(model.state_dict(), "best_model.pt")
                np.save("best_thresholds.npy", best_thresholds)
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
                "f1_macro": curves["f1_macro"][-1],
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: f1={curves['f1_macro'][-1]:.4f} "
            f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
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
- **duration_seconds:** 5419.960328750021
- **workdir:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260414_211157_full_dataset_v4/exp_001
- **results_json_path:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260414_211157_full_dataset_v4/exp_001/results.json
- **timed_out:** False
