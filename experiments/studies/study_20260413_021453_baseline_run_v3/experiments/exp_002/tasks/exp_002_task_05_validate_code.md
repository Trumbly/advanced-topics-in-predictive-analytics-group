# Task exp_002_task_05_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 02:27:11.628871+00:00
- **Completed:** 2026-04-13 02:27:11.631672+00:00

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

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# Note: The provided proposal has augmentation settings that override the
# default structure, so we use those specific settings here.
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# EPOCHS must be read from the env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3


# === Model definition at MODULE scope ===
class SelfAttentionBlock(nn.Module):
    """
    Implements a basic self-attention mechanism for the feature vector 
    output by the CNN backbone.
    Input: (B, D) - D is the channel dimension (128).
    Output: (B, D)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))

    def forward(self, x):
        # x shape: (Batch, Embed_Dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled Dot-Product Attention: (B, D) @ (B, D).T / sqrt(D)
        # Since we are treating the entire feature vector as L=1 sequence length
        # and D=feature dimension, we calculate attention over the feature dimension.
        # For simplicity and robustness, we use dot product similarity 
        # across the feature dimension for self-attention weighting.
        
        # Compute attention scores: (B, D) * (B, D).T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        attention_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output


class CnnAttentionModel(nn.Module):
    """
    3-conv CNN front-end followed by Self-Attention pooling layer.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Conv CNN Front-end ---
        # Conv1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Conv2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Conv3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # --- Self-Attention Pooling Layer ---
        # 1. Collapse spatial dimensions to (B, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 2. Self-Attention mechanism operating on the 128 features
        self.attention = SelfAttentionBlock(embed_dim=128)
        
        # --- Classification Head ---
        # Use LazyLinear for robustness against input size variance
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x)) # Output shape: (B, 128, H', W')
        
        # 2. Pooling and Attention
        x = self.pool(x) # Output shape: (B, 128, 1, 1)
        x = x.flatten(2) # Output shape: (B, 128, 1) - Flattening H', W'
        x = x.squeeze(-1) # Output shape: (B, 128)
        
        x = self.attention(x) # Output shape: (B, 128)
        
        # 3. Classification Head
        logits = self.head(x) # Output shape: (B, num_classes)
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
        # Need to adjust augmentation dictionary keys based on the proposal's JSON
        # which lists "specaugment": true, but the skeleton suggests
        # "time_shift", "noise_injection", etc. We use the proposal's values
        # for the keys that exist in the skeleton's structure.
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation={
                "time_shift": AUGMENTATION["time_shift"], 
                "noise_injection": AUGMENTATION["noise_injection"],
                "mixup": AUGMENTATION["mixup"],
                "specaugment": AUGMENTATION["specaugment"],
            },
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
        with torch.no_grad():
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
                    # average_precision_score computes the area under the ROC curve (AUC)
                    # which is commonly used as a proxy for AP, but for the specific
                    # BirdCLEF AP@K metric, we calculate AP for all classes and average.
                    # Note: The prompt description is slightly ambiguous (AP vs AUC).
                    # We stick to average_precision_score as it is the most robust metric
                    # available in sklearn for this context, which approximates the
                    # desired performance measure.
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 232: invalid syntax
