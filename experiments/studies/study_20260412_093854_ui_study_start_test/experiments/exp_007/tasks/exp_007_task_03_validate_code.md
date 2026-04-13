# Task exp_007_task_03_validate_code

- **Experiment:** exp_007
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-12 07:55:42.011528+00:00
- **Completed:** 2026-04-12 07:55:42.013032+00:00

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

# === Hardcoded hyperparameters from the proposal (NOT a hyperparams dict) ===
# Note: Overriding skeleton defaults based on proposal JSON.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1")) # Uses env var, default 1
LR = 5e-4 # From proposal: lr: 0.0005
AUGMENTATION = {"time_shift": False, "noise_injection": True, "mixup": 0.5, "specaugment": True}

# --- Model Components ---

class ResidualBlock(nn.Module):
    """A standard residual connection block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initial Conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism operating over feature maps (H, W).
    This treats the spatial dimensions (H*W) as the sequence length.
    """
    def __init__(self, in_channels, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = nn.Dropout(dropout_rate)
        
        # 1x1 convolutions to project features into Q, K, V subspaces
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        
        # Final output projection
        self.g = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Project: (B, C, H, W) -> (B, C/8, H, W)
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)
        
        # 1. Calculate Attention Scores (Q * K_transpose)
        # We flatten H*W into a vector length L = H*W.
        # Reshape to (B, L, C/8)
        Q_flat = Q.view(B, -1, Q.size(-1))
        K_flat = K.view(B, -1, K.size(-1))
        V_flat = V.view(B, -1, V.size(-1))
        
        # Attention map: (B, L, C/8) @ (B, C/8, L) -> (B, L, L)
        attention_map = torch.bmm(Q_flat, K_flat.transpose(1, 2)) * (Q.size(-1) ** -0.5)
        
        # Softmax over the last dimension (L)
        attention_map = F.softmax(attention_map, dim=-1)
        
        # 2. Apply attention map to Values
        # Output: (B, L, L) @ (B, L, C/8) -> (B, L, C/8)
        out_flat = torch.bmm(attention_map, V_flat)
        
        # 3. Reshape back to feature map dimensions
        out = out_flat.view(B, self.in_channels, H, W)
        
        # Final linear projection
        out = self.dropout(self.g(out))
        return out


class AttentionCNN(nn.Module):
    """
    3-Conv CNN front-end using residual blocks, followed by Self-Attention.
    """
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # --- CNN Front-end ---
        # Input: (B, 1, 128, 313)
        self.conv_block1 = ResidualBlock(in_channels=1, out_channels=32, dropout_rate=dropout_rate)
        self.conv_block2 = ResidualBlock(in_channels=32, out_channels=64, dropout_rate=dropout_rate)
        self.conv_block3 = ResidualBlock(in_channels=64, out_channels=128, dropout_rate=dropout_rate)
        
        self.cnn_features = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3
        )
        
        # --- Attention ---
        self.attention = SelfAttention(in_channels=128, dropout_rate=dropout_rate)
        
        # --- Head ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle unknown feature map size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.cnn_features(x) # (B, 128, H', W')
        
        # 2. Self-Attention
        x = self.attention(x) # (B, 128, H', W')
        
        # 3. Pool and Flatten
        x = self.pool(x) # (B, 128, 1, 1)
        x = x.flatten(1) # (B, 128)
        
        # 4. Linear Head
        logits = self.head(x) # (B, num_classes)
        return logits

# --- Module Scope END ---

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads config/config.yaml for batch_size, etc.
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
        model = AttentionCNN(num_classes=num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": []}
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
                    # Sigmoid for probability, move to CPU for sklearn/numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0: # Only calculate AUC if there's at least one positive label
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 243: '(' was never closed
