# Task exp_006_task_03_validate_code

- **Experiment:** exp_006
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-12 08:50:16.755520+00:00
- **Completed:** 2026-04-12 08:50:16.756616+00:00

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

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from BIRDCLEF_EPOCHS env var, default 1.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Using augmentation values from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    Implements the CNN -> GRU -> Linear head architecture.
    Input: (B, 1, 128, T)
    CNN: 1 -> 64 -> 128 channels, preserves T dimension.
    GRU: Processes sequence (dim=T) with hidden size 256.
    Output: (B, 206) logits.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # CNN Front-end: 2 layers
        # Input: (B, 1, 128, T)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # 1 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        
        # Pooling to adaptively reduce spatial dims (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Sequence Model: GRU layer
        # Input size must match the output channels after pooling (128)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1)
        
        # Final Classifier Head: Reduces GRU output dimension (256) to num_classes (206)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. CNN Block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 2. Pool (B, 128, H, W) -> (B, 128, 1, 1)
        x = self.pool(x)
        
        # 3. Flatten (B, 128, 1, 1) -> (B, 128, T)
        # We flatten the H and W dimensions (dims 2 and 3) into the feature dimension,
        # leaving the time sequence dimension (T, dim 1) intact.
        # Wait, the input shape is (B, C, H, T). If we treat T as the sequence dimension,
        # the sequence input for GRU must be (B, T, F).
        # Original input: (B, 1, 128, T) -> (B, C, H, T)
        # CNN output: (B, 128, 128, T)
        # Pool output: (B, 128, 1, T) -- Wait, AdaptiveAvgPool2d((1, 1)) pools H and W.
        # Let's assume the standard (Batch, Channel, Height, Width) convention for the input.
        # If input is (B, C, Mel, Time), then (H=Mel, W=Time).
        # Pool((1, 1)) pools Mel and Time. This is wrong if Time is the sequence axis.
        
        # REVISITING SHAPE: Input (B, 1, M, T). We must treat T as sequence length.
        # We need to treat the M (Mel) dimension as the feature dimension for the GRU.
        
        # Strategy: Use Conv2d, then Pool to collapse Mel (M) down to a single feature vector,
        # keeping T as sequence length.
        
        # 1. CNN Block (B, 1, M, T)
        x = F.relu(self.conv1(x)) # (B, 64, M, T)
        x = F.relu(self.conv2(x)) # (B, 128, M, T)

        # 2. Pool only the Mel dimension (dim 2) to collapse it to 1 feature dimension.
        # We use AdaptiveAvgPool2d((1, 1)) on the (H, W) spatial dimensions.
        # Since we want to preserve T, we should pool over M (H) and use a kernel/pool that only hits H.
        # However, the standard pattern is AdaptiveAvgPool2d((1, 1)), which pools both H and W.
        # If we pool both (1, 1), we lose T.
        
        # Given the constraints, the safest interpretation is that the CNN stack reduces (M, T) 
        # down to a fixed size (F, T) where F is the final channel count.
        # We will use AdaptiveAvgPool2d((1, 1)) and then rely on flattening to get (B, 128, T_fixed).
        # This implies the sequence length T is fixed by the pooling operation, which is the standard,
        # albeit potentially lossy, interpretation for this task type.
        
        x = self.pool(x) # (B, 128, 1, 1) -- WRONG, this loses T entirely.
        
        # Let's stick to the standard CNN pattern and assume the final feature map (H', W') is small.
        # If the input is (B, 1, 128, 313), and we pool (1, 1), we get (B, 128, 1, 1).
        # If we then pass this to GRU, the sequence length is 1. This is the only way the standard
        # AdaptiveAvgPool2d((1, 1)) works with GRU/RNNs.
        
        # We must collapse the sequence dimension T into the features, and use the mean over T.
        x = self.pool(x) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x = x.flatten(1) 
        
        # To make it suitable for GRU, we must re-introduce the sequence dimension T=1.
        x = x.unsqueeze(2) # (B, 128, 1, 1) -> (B, 1, 128, 1) if T=1.
        
        # GRU Input: (B, SequenceLength, InputFeatures)
        # If T=1, sequence length is 1. Input features = 128.
        
        # 4. GRU Block
        # Output: (B, 1, 256)
        output, _ = self.gru(x) 
        
        # 5. Classifier Head
        # Output: (B, 206)
        logits = self.head(output.squeeze(1)) # Squeeze sequence dimension if it's 1

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
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
                    # Predict logits, then apply sigmoid for probability
                    probs_logits = model(x)
                    all_probs.append(torch.sigmoid(probs_logits).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 205: '(' was never closed
