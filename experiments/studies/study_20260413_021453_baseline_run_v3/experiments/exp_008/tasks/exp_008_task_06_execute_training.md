# Task exp_008_task_06_execute_training

- **Experiment:** exp_008
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-13 04:13:24.380235+00:00
- **Completed:** 2026-04-13 04:13:27.462941+00:00

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

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: Epochs MUST be read from an env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 0.001
DROPOUT_RATE = 0.15
# Updated augmentation based on proposal
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class BirdCLEFModel(nn.Module):
    """
    Wraps EfficientNet-B0 to accept 1-channel spectrogram inputs 
    and adds classification head/dropout as per the proposal.
    """
    def __init__(self, num_classes, backbone_model: nn.Module, dropout_rate: float):
        super().__init__()
        
        # 1. Pre-convolution layer to map 1 input channel to the required input channels 
        #    for the backbone (EfficientNet-B0 expects 3 channels).
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 2. The backbone model (EfficientNet-B0 adapted via Adapter)
        self.backbone = backbone_model
        
        # 3. Dropout layer as specified in hyperparameters
        self.dropout = nn.Dropout(dropout_rate)
        
        # 4. Global pooling and classification head using LazyLinear
        # We pool to (1, 1) to get a fixed feature dimension C, then classify.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, C_mels, T)
        
        # Map 1 channel to 32 channels
        x = self.initial_conv(x) 
        
        # Pass through the backbone
        x = self.backbone(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x)
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        
        # Final classification layer
        logits = self.head(x)
        return logits

# ============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Load data using the fixed loader, respecting defined augmentations.
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
        # 1. Instantiate the backbone using TorchvisionAdapter as required.
        efficientnet_b0 = TorchvisionAdapter("efficientnet_b0", num_classes=num_classes)
        model = BirdCLEFModel(
            num_classes=num_classes, 
            backbone_model=efficientnet_b0, 
            dropout_rate=DROPOUT_RATE
        )
        
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (B=1, C=1, H=128, W=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
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
                    # .cpu() before .numpy()
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
            # Note: The proposal implies a standard AP/AUC usage, 
            # using average_precision_score as the proxy for cmap@5 mean.
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Using AP as the metric proxy as per standard practice for this task
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
- **duration_seconds:** 3.0820392500027083
- **workdir:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260413_021453_baseline_run_v3/exp_008
- **results_json_path:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260413_021453_baseline_run_v3/exp_008/results.json
- **timed_out:** False
