# Task exp_016_task_04_validate_code

- **Experiment:** exp_016
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 06:10:34.027410+00:00
- **Completed:** 2026-04-13 06:10:34.030160+00:00

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
# Note: These values are overridden by env vars in the run block.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# --- Custom Components ---

class SelfAttention(nn.Module):
    """
    A simplified Time-Aware Self-Attention block applied to feature maps.
    Input shape: (B, C, H, W)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(dim, dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scale factor

    def forward(self, x):
        B, C, H, W = x.size()
        
        # 1. Calculate Q, K, V
        Q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1) # (B, N, D/4)
        K = self.key_conv(x).view(B, -1, H * W) # (B, D/4, N)
        V = self.value_conv(x).view(B, -1, H * W) # (B, D, N)

        # 2. Calculate Attention Map (Dot product Q * K)
        attention_map = torch.bmm(Q, K).transpose(1, 2) # (B, N, N)
        
        # 3. Apply Softmax and Weight
        attention_weights = F.softmax(attention_map, dim=-1) * (self.dim**-0.5)
        
        # 4. Output weighted sum
        out = torch.bmm(attention_weights, V) # (B, D, N)
        
        # 5. Reshape and scale: (B, D, N) -> (B, D, 1, 1) * (H, W) -> (B, D, H, W)
        out = out.permute(0, 2, 1).view(B, self.dim, H, W)
        
        # 6. Residual connection and scaling
        return self.gamma * out + x

# === Model definition at MODULE scope ===
class CnnAttentionModel(nn.Module):
    """
    MobileNetV3-style backbone followed by Time-Aware Self-Attention.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- Backbone (Mimicking MobileNetV3/Small CNN structure) ---
        # Input: (B, 1, 128, 313)
        
        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers to reduce spatial dimensions (simulating depth)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU(inplace=True)
        
        # --- Attention Block ---
        # Operating on features with 128 channels
        self.attention = SelfAttention(dim=128)

        # --- Classifier Head ---
        # Adaptive pooling ensures robustness regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Using LazyLinear to handle feature dimension inference
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. Backbone Feature Extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x) # (B, 32, 64, 156) approx
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x) # (B, 64, 32, 78) approx
        
        x = self.relu(self.bn3(self.conv3(x)))
        # No pooling here, keep spatial map size for attention
        
        # 2. Attention Application
        x = self.attention(x)
        
        # 3. Classification Head
        x = self.pool(x) # (B, 128, 1, 1)
        x = x.view(x.size(0), -1) # Flatten to (B, 128)
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
        model = CnnAttentionModel(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Without this, p.numel() crashes on uninitialized parameters.
        with torch.no_grad():
            # Dummy input shape matching (B, 1, 128, 313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
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

            # --- Metric 2: cmap@5 ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Using average_precision_score as proxy for cmap@5 as per standard practice
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
            f"f1={curves['f1

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 253: unterminated string literal (detected at line 253)
