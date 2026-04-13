# Task exp_016_task_05_validate_code

- **Experiment:** exp_016
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 06:12:59.058282+00:00
- **Completed:** 2026-04-13 06:12:59.060201+00:00

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

# === Device selection ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# NOTE: EPOCHS is read from env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Update augmentation based on proposal: specaugment=True, others=False
AUGMENTATION = {"specaugment": True, "time_shift": False, "noise_injection": False, "mixup": 0.0}

# Import the specific registry model required by the proposal
try:
    from pipelines.models import MobileNetV3Small
except ImportError:
    # Fallback/Placeholder if environment is not fully set up for testing
    print("Warning: Could not import MobileNetV3Small. Using a dummy backbone for structure check.", flush=True)
    class MobileNetV3Small(nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy backbone: 1 -> 64 channels
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return x

# =============================================================================
# Custom Attention Block Definition (Time-Aware Self-Attention)
# Input: (B, C, T) - Feature map from backbone
# Output: (B, C, T) - Context-aware feature map
# =============================================================================
class TimeAwareAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Q, K, V projections
        self.query_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.key_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (B, C, T)
        B, C, T = x.size()
        
        # 1. Project Q, K, V
        Q = self.query_proj(x).view(B, C, T).permute(0, 2, 1) # (B, T, C)
        K = self.key_proj(x).view(B, C, T).permute(0, 2, 1) # (B, T, C)
        V = self.value_proj(x).view(B, C, T).permute(0, 2, 1) # (B, T, C)

        # 2. Calculate Attention Scores: (B, T, C) @ (B, C, T) -> (B, T, T)
        # We use torch.bmm for batched matrix multiplication
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (C ** 0.5)
        
        # 3. Apply Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 4. Apply weights to V
        context_vector = torch.bmm(attention_weights, V) # (B, T, C)
        
        # 5. Reshape back to (B, C, T) and project output
        context_vector = context_vector.permute(0, 2, 1) # (B, C, T)
        output = self.output_proj(context_vector)
        
        return self.dropout(output)

# =============================================================================
# Main Model Definition
# =============================================================================
class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print("Initializing BirdCLEFModel...")
        
        # 1. Backbone: MobileNetV3Small (Registry Model)
        self.backbone = MobileNetV3Small()
        
        # 2. Attention Block: Operates on the feature maps from the backbone
        # The output channels of the backbone are used as the embedding dimension.
        # We rely on the backbone's final feature depth.
        # Since we are using adaptive pooling later, we estimate the feature depth
        # by checking the initial convolution output size. Assuming 64 channels for simplicity
        # based on typical mobile backbone output before global pooling.
        # A more robust solution would involve hooking into the backbone's final layer.
        # Here we assume the feature maps retain a depth of 128 after standard processing.
        self.attention = TimeAwareAttention(embed_dim=128) 
        
        # 3. Global Pooling and Classifier Head
        # Use AdaptiveAvgPool2d to collapse spatial dimensions (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use LazyLinear for the final classification head
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, 313)
        
        # 1. Backbone Feature Extraction
        x = self.backbone(x) # Output shape: (B, C_out, H_out, W_out)
        
        # 2. Attention Block
        x = self.attention(x) # Output shape: (B, C_out, H_out, W_out)
        
        # 3. Global Pooling
        x = self.pool(x) # Output shape: (B, C_out, 1, 1)
        
        # 4. Flatten and Classify
        x = x.flatten(1) # Output shape: (B, C_out)
        logits = self.head(x) # Output shape: (B, num_classes)
        
        return logits

# =============================================================================
# RUNTIME section
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
        model = BirdCLEFModel(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Dummy input shape matches the expected input: (B, 1, 128, 313)
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

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 239: unterminated string literal (detected at line 239)
