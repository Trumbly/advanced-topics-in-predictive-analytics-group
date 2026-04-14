# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-14 02:38:03.465289+00:00
- **Completed:** 2026-04-14 02:38:03.488469+00:00

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
from pipelines.models import TorchvisionAdapter

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal ===
# Note: LR and Dropout are modified from the skeleton defaults.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "7"))
LR = 0.0005 # Updated from 1e-3 to 0.0005
WEIGHT_DECAY = 0.0 # From proposal
DROPOUT_RATE = 0.1 # From proposal
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class MobileNetAdapter(nn.Module):
    """
    Wrapper around TorchvisionAdapter for MobileNetV3-Small,
    adding dropout and the final classification head.
    """
    def __init__(self, pretrained_model_name, num_classes, dropout_rate):
        super().__init__()
        # Use the registry adapter to handle 1->3 channel expansion and resizing
        self.backbone = TorchvisionAdapter(pretrained_model_name, num_classes=num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        # Use LazyLinear to handle the dynamically sized feature map from the backbone
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # The backbone processes the input (1, C, H, W) -> (B, Features)
        x = self.backbone(x)
        # Apply dropout before the final linear layer
        x = self.dropout(x)
        # Final classification layer
        return self.head(x)

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
        # load_precomputed_dataset reads required env vars (batch_size, etc.)
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
        model = MobileNetAdapter(
            pretrained_model_name="mobilenet_v3_small",
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE
        )
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape must match the expected (1, 128, 313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        PATIENCE = 3
        best_auc = 0.0
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
            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
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
- **code_bytes:** 8350
