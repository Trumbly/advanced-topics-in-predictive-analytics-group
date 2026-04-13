# Task exp_003_task_06_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 14:11:45.514141+00:00
- **Completed:** 2026-04-12 14:11:45.517500+00:00

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
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters from the proposal ===
# Note: EPOCHS is read from an env var, not hardcoded.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": True, "mixup": 0.0, "specaugment": False}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    Implements the proposed architecture: CNN -> Temporal Pooling -> GRU -> Linear Head.
    Input: (Batch, 1, Mel, Time)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. CNN Front-end (Input: 1 channel, Output: 32 channels)
        # Conv2d(in, out, kernel_size, padding)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pool over Mel dimension (dim=2) to keep Time dimension (dim=3)
        # Input: (B, 32, Mel, Time) -> Output: (B, 32, 1, Time)
        self.mel_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. GRU Component
        # Input feature size is 32 (the output of the CNN).
        # batch_first=True -> Input/Output shape: (Batch, Sequence, Features)
        self.gru = nn.GRU(input_size=32, hidden_size=128, batch_first=True)
        
        # 3. Output Head (Aggregating time dimension and projecting to num_classes)
        # Pool the final GRU sequence output over the time dimension (dim=1)
        # Input: (B, Time, 128) -> Output: (B, 128)
        self.time_pool = nn.AdaptiveAvgPool1d(1)
        # Final linear projection using LazyLinear for robustness
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, Mel, Time)
        
        # CNN Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Pool over Mel dimension (dim=2)
        # Shape: (B, 32, 1, Time)
        x = self.mel_pool(x)
        
        # Reshape for GRU: (B, C, 1, T) -> (B, 32, T)
        # We squeeze dimension 2 (the pooled dimension)
        x = x.squeeze(2) # Shape: (B, 32, T)
        
        # GRU Input requires (B, T, F) for batch_first=True
        # Shape: (B, 32, T) -> (B, T, 32)
        x = x.permute(0, 2, 1)

        # GRU: (B, T, 32) -> (B, T, 128)
        gru_out = self.gru(x)
        
        # Time Pooling: (B, T, 128) -> (B, 128)
        # We pool over the time dimension (dim=1)
        pooled_out = self.time_pool(gru_out)
        
        # Final Linear Projection
        logits = self.fc(pooled_out)
        return logits

# ==============================================================================
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
        # Data loading uses the configuration set by the orchestrator/env vars
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
        model = CnnGruHybridModel(num_classes=num_classes)
        model = model.to(device)
        
        # FIX: Initialize parameters for LazyLinear before counting parameters
        # Create a dummy input tensor (Batch=1, 1, Mel, Time)
        # We use a dummy batch size of 1.
        dummy_x = torch.randn(1, 1, 1, 1).to(device, dtype=torch.float32)
        _ = model(dummy_x)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

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

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Sigmoid for probabilities, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            # Concatenate results
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
                    # average_precision_score calculates the area under the precision-recall curve,
                    # which is equivalent to mean average precision (AP) when computing for one class.
                    # We average these APs across classes.
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
- **code_bytes:** 9230
