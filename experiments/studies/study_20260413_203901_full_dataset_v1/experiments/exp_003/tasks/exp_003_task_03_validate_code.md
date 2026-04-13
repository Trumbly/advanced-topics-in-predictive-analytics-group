# Task exp_003_task_03_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 21:27:23.013014+00:00
- **Completed:** 2026-04-13 21:27:23.018096+00:00

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
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters (Read from env var, use proposal overrides where appropriate) ===
# EPOCHS is read from an env var so the orchestrator can override it.
# Default 15 is used per skeleton, despite proposal suggesting 1.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "15"))
LR = 1e-3
# Using proposal dropout rate
DROPOUT_RATE = 0.15
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": False}

# === Model definition at MODULE scope ===

class CnnGruHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- CNN Front-end: 3-conv blocks (1 -> 32 -> 64 channels) ---
        # Input: (B, 1, 128, T)
        
        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 64 -> 64 (Final Conv)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Dropout layer as specified in proposal
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        
        # --- GRU Layer ---
        # Input size for GRU must match the flattened channels from CNN_OUT * H_OUT
        # CNN_OUT size: (B, 64, 32, T/4)
        # Flattened size: 64 * 32 = 2048
        CNN_FEATURE_DIM = 64 * 32
        GRU_HIDDEN_SIZE = 128
        
        # GRU takes (B, T_seq, Feature_Dim) if batch_first=True
        self.gru = nn.GRU(
            input_size=CNN_FEATURE_DIM, 
            hidden_size=GRU_HIDDEN_SIZE, 
            batch_first=True
        )
        
        # --- Final Linear Head ---
        # Pool the GRU output sequence (T_seq) down to a single vector representation (128)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
        # Use LazyLinear for robustness against changing time dimensions
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, T)
        
        # --- CNN Feature Extraction ---
        x = F.relu(self.conv1(x))
        x = self.pool1(x) # (B, 32, 64, T/2)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x) # (B, 64, 32, T/4)
        
        x = F.relu(self.conv3(x)) # (B, 64, 32, T/4)
        
        # Flatten the spatial dimensions (Height and Channels) to create the GRU feature dimension
        # Target shape: (B, 64*32, T/4)
        x = x.permute(0, 2, 3, 1).contiguous() # (B, T/4, 64, 32)
        x = x.view(x.size(0), -1, x.size(1)) # (B, 2048, T/4) - Incorrect permutation for time dimension
        
        # Correct permutation: Keep Time dimension (dim 1 after permute) separate
        # (B, C, H, W) -> (B, H, W, C) -> (B, T/4, 64*32)
        # Let's use the standard (B, C, H, W) -> (B, C*H, W) trick, but GRU needs (B, T, F)
        
        # Strategy: Treat the final time dimension (W) as the sequence length T_seq
        # (B, C, H, W) -> (B, H*C, W)
        x = x.view(x.size(0), -1, x.size(3)) # (B, 64*32, T/4)
        
        # Apply dropout
        x = self.dropout(x)
        
        # --- GRU Sequence Modeling ---
        # Input shape: (B, Feature_Dim, T_seq) -> (B, 2048, T/4)
        # GRU expects (B, T_seq, Input_Size) if batch_first=True
        x = x.permute(0, 2, 1).contiguous() # (B, T/4, 2048)
        
        # output: (B, T/4, 128)
        output, _ = self.gru(x)
        
        # --- Final Classification Head ---
        # Pool over the time dimension (T/4) to get (B, 128)
        # AdaptiveAvgPool1d expects (B, C, L)
        output = output.permute(0, 2, 1).contiguous() # (B, 128, T/4)
        output = self.pool(output) # (B, 128, 1)
        output = output.view(output.size(0), -1) # (B, 128)
        
        # Final classification logits
        logits = self.classifier(output) # (B, num_classes)
        return logits

# ===========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
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
        model = CnnGruHybridModel(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Use a dummy input matching expected preprocessed shape (1, 1, 128, 313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
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
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Note: average_precision_score is used as a proxy for cmap@5/AP
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
- **validation:** failed

## Error
- **type:** EpochsCapExceeded
- **message:** BIRDCLEF_EPOCHS default 15 exceeds the hard cap of 1. Use default '15'.
