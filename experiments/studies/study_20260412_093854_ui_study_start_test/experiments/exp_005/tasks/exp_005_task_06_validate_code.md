# Task exp_005_task_06_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 07:51:26.907902+00:00
- **Completed:** 2026-04-12 07:51:26.913300+00:00

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
# EPOCHS is read from an env var. Default of 1 keeps fast iteration working.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Model Definition ---
class CnnGruHybrid(nn.Module):
    """
    Architecture: 3-conv CNN front-end (spectral features) -> GRU (temporal) -> Linear projection.
    Input shape: (B, 1, N_mels=128, T=313)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- CNN Front-end (Spectral Feature Extraction) ---
        # Input: (B, 1, 128, 313)
        
        # Block 1: 1 -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Block 3: 128 -> 256 channels (C3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # --- Temporal Preparation ---
        # Pool across the Mel dimension (dim=2, index 2) to collapse 128 to 1.
        # Shape: (B, 256, 128, T) -> (B, 256, 1, T)
        self.pool_mel = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- GRU Layer (Temporal Processing) ---
        # Input size must match the channel count from the last CNN block (256).
        # Hidden size = 256.
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=False)
        
        # --- Final Projection ---
        # After GRU, output is (B, T, 256). Pool across time (dim=1) to get (B, 256).
        self.pool_time = nn.AdaptiveAvgPool1d(1)
        # Final linear layer uses LazyLinear to handle variable feature sizes.
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, T)
        
        # 1. CNN Feature Extraction
        x = self.relu1(self.conv1(x))    # (B, 64, 128, T)
        x = self.relu2(self.conv2(x))    # (B, 128, 128, T)
        x = self.relu3(self.conv3(x))    # (B, 256, 128, T)
        
        # 2. Temporal Preparation (Pool Mel dim)
        x = self.pool_mel(x)            # (B, 256, 1, T)
        
        # Permute from (B, C, 1, T) -> (B, 256, T)
        x = x.squeeze(2) # (B, 256, T)
        
        # 3. GRU Processing
        # GRU expects (seq_len, batch, input_size) if batch_first=False (default)
        # Our current shape is (B, 256, T). We need (T, B, 256).
        # Permute to (B, T, 256) -> (T, B, 256)
        gru_input = x.permute(2, 0, 1) # (T, B, 256)
        gru_output = self.gru(gru_input) # (T, B, 256)
        
        # 4. Aggregation and Classification
        # (T, B, 256) -> (B, 256)
        # Needs to be (B, C, L) for AvgPool1d.
        gru_output = gru_output.permute(1, 2, 0) # (B, 256, T)
        pooled = self.pool_time(gru_output)     # (B, 256, 1)
        
        # Flatten (B, 256, 1) -> (B, 256)
        x = pooled.flatten(1)
        
        # Final linear projection
        logits = self.classifier(x) # (B, num_classes)
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
    curves = {"loss": [], "roc_auc_macro": []} # FIX: Initialize 'curves' dictionary here
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size/num_workers etc. from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

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
                    # Calculate probabilities (sigmoid) and move to CPU/numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for i in range(num_classes):
                if np.any(targs[:, i] == 1):
                    try:
                        auc = roc_auc_score(targs[:, i], probs[:, i])
                        aucs.append(auc)
                    except ValueError:
                        # Handle case where only one class is present in the validation set
                        pass
            
            macro_auc = np.mean(aucs) if aucs else 0.0
            curves["roc_auc_macro"].append(macro_auc)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}", flush=True)
        results["error"] = str(e)
    finally:
        # Ensure results are saved even if an error occurred
        results["final_auc"] = np.mean(curves["roc_auc_macro"]) if curves["roc_auc_macro"] else 0.0
        results["epochs_completed"] = EPOCHS
        
        print("\nSaving results...", flush=True)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Done.", flush=True)

```

## Output
- **validation:** passed
- **code_bytes:** 8193
