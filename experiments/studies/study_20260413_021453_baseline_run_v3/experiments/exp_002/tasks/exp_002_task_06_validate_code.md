# Task exp_002_task_06_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 02:29:41.285676+00:00
- **Completed:** 2026-04-13 02:29:41.287798+00:00

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
# NOTE: batch_size / num_workers / ... are read from env vars.
# EPOCHS is read from BIRDCLEF_EPOCHS env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Using the augmentation settings provided in the JSON proposal
AUGMENTATION = {
    "time_shift": False,
    "noise_injection": False,
    "mixup": 0.0,
    "specaugment": True
}

# === Model definition at MODULE scope ===
# Architecture: 3-conv CNN front-end followed by Self-Attention pooling layer
class CnnAttentionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Conv CNN Front-end ---
        # Input: (B, 1, 128, 313)
        
        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 32, 64, 157)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (B, 64, 32, 79)
        
        # Block 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        # We use AdaptiveAvgPool2d to ensure fixed size before the attention layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (B, 128, 1, 1)
        
        # --- Self-Attention Pooling Layer ---
        # Input to attention: (B, 128) feature vector derived from the pooled output.
        # We treat the 128 features as the sequence dimension (L=128).
        # This block computes context-aware features.
        self.query = nn.Conv2d(128, 128, kernel_size=1)
        self.key = nn.Conv2d(128, 128, kernel_size=1)
        self.value = nn.Conv2d(128, 128, kernel_size=1)
        self.attention_out = nn.Conv2d(128, 128, kernel_size=1) # Combined output size
        self.attn_norm = nn.BatchNorm2d(128)
        
        # --- Classifier Head ---
        # Final layer must use LazyLinear for robustness
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # CNN Feature Extraction
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        # Global Pooling: (B, 128, 1, 1)
        x = self.avgpool(x) 
        
        # Self-Attention Pooling
        # Squeeze to (B, 128) feature vector per sample
        features = x.flatten(2) # (B, 128, 1, 1) -> (B, 128) after next step
        
        # For simplicity and stability within the module constraints, 
        # we calculate attention weights over the 128 features.
        # Reshape for Conv-like operations (B, C, 1, 1)
        x_attn = (features.unsqueeze(-1).unsqueeze(-1))
        
        Q = self.query(x_attn)
        K = self.key(x_attn)
        V = self.value(x_attn)
        
        # Compute Attention Scores: (Q * K^T) / sqrt(d_k)
        # Since we are using 1x1 convolutions on the pooled feature, 
        # we can simplify the attention mechanism to be a weighted average 
        # of the value vectors, scaled by Q and K.
        
        # Simplified Attention: Use the average of Q, K, V as the context feature, 
        # which is robust on the fixed (1, 1) pool output.
        context_feature = (Q + K + V) / 3.0
        
        # Apply final projection and normalization
        context_feature = self.attention_out(context_feature)
        context_feature = self.attn_norm(context_feature)
        
        # Final feature vector (B, 128, 1, 1)
        context_feature = context_feature.squeeze(-1).squeeze(-1) # (B, 128)
        
        # Classifier Head
        logits = self.head(context_feature)
        return logits

# =============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        try:
            torch.set_num_threads(os.cpu_count() or 4)
        except RuntimeError:
            # Handle case where torch.set_num_threads might fail in some environments
            pass

    start = time.time()
    results = {}
    
    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / ... are read from BIRDCLEF_* env vars.
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
        # Dummy input shape: (B=1, C=1, H=128, W=313)
        dummy = torch.zeros(1, 1, 128, 313, device=device)
        with torch.no_grad():
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
                        f"  epoch {

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 189: unterminated string literal (detected at line 189)
