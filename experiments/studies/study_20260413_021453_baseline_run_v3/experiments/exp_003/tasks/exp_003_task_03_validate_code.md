# Task exp_003_task_03_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 02:43:08.813477+00:00
- **Completed:** 2026-04-13 02:43:08.819330+00:00

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

# === Hyperparameters ===
# Read EPOCHS from env var, default to 1 for fast iteration.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    3-conv CNN front-end (1->32->64 channels) -> Global AvgPool -> GRU(128) head.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. CNN Front-end (3 layers, 1 -> 32 -> 64 channels)
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 1 -> 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32 -> 64
        # Adding a third convolution block to satisfy "3-conv CNN front-end"
        # We keep the feature dimension constant here for simplicity, 64 -> 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Pooling: Global Avg Pool over feature channels
        # Output shape: (B, 64, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. GRU Head: GRU(128)
        # Input to GRU: (B, SeqLen, FeatureDim). We reshape (B, 64, 1, 1) -> (B, 1, 64)
        # Sequence length = 1, Feature dimension = 64
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True
        )
        
        # 4. Final Linear Head: Maps GRU hidden state (128) to num_classes (10)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, 128, 313)
        
        # CNN Blocks
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Pooling: (B, 64, 1, 1)
        x = self.pool(x)
        
        # Flatten spatial dimensions and reshape for GRU: (B, C, 1, 1) -> (B, 64) -> (B, 1, 64)
        # Here, we treat the 64 feature maps as the feature dimension, and 1 as the sequence length.
        x = x.squeeze(-1).squeeze(-1)  # (B, 64)
        x = x.unsqueeze(1)            # (B, 1, 64)
        
        # GRU Pass
        # output shape: (B, SeqLen, HiddenSize) -> (B, 1, 128)
        output, _ = self.gru(x)
        
        # Final Linear Projection
        # (B, 1, 128) -> (B, 128) -> (B, num_classes)
        output = output.squeeze(1) # (B, 128)
        logits = self.fc(output)  # (B, num_classes)
        
        return logits

# ========================================================================
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
        # Since we use nn.Linear(128, num_classes) this is not strictly lazy,
        # but we must run a dummy pass to ensure all weights are initialized correctly.
        with torch.no_grad():
            # Dummy input matching (B, 1, 128, 313)
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
                    # Sigmoid for probability, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 173: expected 'except' or 'finally' block
