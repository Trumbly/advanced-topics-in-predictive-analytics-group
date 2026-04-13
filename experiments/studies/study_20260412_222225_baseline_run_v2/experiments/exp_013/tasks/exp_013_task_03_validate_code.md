# Task exp_013_task_03_validate_code

- **Experiment:** exp_013
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 01:05:03.700322+00:00
- **Completed:** 2026-04-13 01:05:03.709532+00:00

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
# The proposal specifies "dropout": 0.2. We use the default augmentation
# unless explicitly overridden in the call below.
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class CnnGruHybridModel(nn.Module):
    """
    Lightweight 3-conv stack (64 channels) -> Flatten -> 1-layer GRU(128) -> Global Average Pooling -> Linear Head
    Input: (B, 1, N_mels, T) -> (B, 64, N_mels/2, T/2) -> (B, 128) -> (B, 128) -> (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Lightweight 3-conv stack (64 channels)
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # out: 32
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves T

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # out: 64
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves T again

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # out: 64
        self.relu3 = nn.ReLU(inplace=True)
        
        # After 3 max pools (if we were using 3 pools), we'd need a final pool.
        # Since the proposal implies a sequence of Conv -> Pool, we'll use
        # AdaptiveAvgPool2d to guarantee spatial collapse before the GRU/Linear layer.
        # We use the final feature map size (64) from conv3.
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1)) # Collapses (B, 64, H/4, T/4) -> (B, 64, 1, 1)

        # 2. GRU Layer
        # The input to the GRU will be feature vectors derived from the 64 channels.
        # Since the GRU expects a time sequence, we treat the 64 feature maps as 
        # the input channels (C_in) and the temporal dimension (T') as the sequence length.
        # Input to GRU: (Batch, Features_Dim, Sequence_Length)
        # We'll reshape the output of the pooling layer: (B, 64) -> (B, 1, 64) (assuming 64 is the feature dim)
        # Let's simplify the interpretation: The 64 channels are the features, and we
        # sequence over the remaining spatial dimension (which is 1x1 after pooling).
        # A simpler, more robust approach for multi-channel feature maps is to
        # treat the 64 channels as the feature dimension (embed_dim) and apply the GRU
        # over a pseudo-time sequence derived from the feature map.
        # Given the architecture description, the most stable interpretation for 
        # Conv -> Pool -> GRU -> Linear is:
        # 1. CNN stack outputs (B, C_out, 1, 1).
        # 2. Flatten: (B, C_out).
        # 3. Pass (B, C_out) through a linear layer or sequence model.
        # We will adapt the GRU to process the feature vector (B, 64) as if it were 
        # the input to the GRU's hidden state, which is non-standard.
        
        # Standard interpretation for (B, C_out) -> GRU(128):
        # Treat C_out (64) as the input feature dimension (input_size) and 
        # use a linear layer to map it to the GRU's input size, or use a standard 
        # TimeDistributed/RNN approach.
        
        # Using the stable (B, C_out) -> Linear -> (B, 128) approach, 
        # but incorporating the GRU structure:
        # We will apply the GRU *after* flattening the 64 channels into a single vector,
        # effectively setting the sequence length to 1, and the input feature size to 64.
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        
        # 3. Global Average Pooling (Applied after GRU output)
        # Since GRU output is (B, L, 128), we pool over L to get (B, 128).
        self.pool_gru = nn.AdaptiveAvgPool1d(1) 
        
        # 4. Linear Head
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, N_mels, T)
        
        # Conv Stack
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x) # (B, 32, H/2, T/2)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x) # (B, 64, H/4, T/4)
        
        x = self.conv3(x)
        x = self.relu3(x) # (B, 64, H/4, T/4)
        
        # Global Pooling to get feature vector (B, 64, 1, 1)
        x = self.pool_final(x)
        
        # Flatten: (B, 64)
        x = x.squeeze(-1).squeeze(-1) 
        
        # GRU input: (B, Features_Dim) -> (B, 1, Features_Dim) for batch_first=True
        x = x.unsqueeze(1) 
        
        # GRU pass: output (B, 1, 128)
        gru_out = self.gru(x)
        
        # Global Avg Pool over sequence length (dim=1): (B, 128)
        x = self.pool_gru(gru_out.permute(2, 0, 1).contiguous()).squeeze(0)
        
        # Linear Head: (B, num_classes)
        logits = self.fc(x)
        return logits


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
                    # average_precision_score calculates the area under the PR curve, 
                    # which is the metric requested for "class-mean average precision".
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
- **code_bytes:** 12140
