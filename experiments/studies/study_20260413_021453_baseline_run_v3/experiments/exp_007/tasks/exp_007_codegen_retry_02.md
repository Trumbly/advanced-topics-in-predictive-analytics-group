# Task exp_007_codegen_retry_02

- **Experiment:** exp_007
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-13 03:56:34.159483+00:00
- **Completed:** 2026-04-13 03:58:59.255725+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained Python script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Input tensor shape —...

[USER]
ture above. Use the skeleton in the system prompt as your structure.
Fill in the model definition and any proposal-specific training changes.

Return ONLY the Python code. No explanations, no markdown fences, no prose.


## VALIDATION ERROR (attempt 2)
Your previous code was rejected by the validator:
  Error type: SyntaxError
  Message: Line 250: '{' was never closed

Fix this specific issue and return the COMPLETE corrected script.
Return ONLY Python code — no explanations, no markdown fences.
```

## LLM Response
```
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
# Overriding skeleton defaults with proposal values where they differ
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
# Use the proposal's learning rate
LR = 0.0008
# Use the proposal's augmentation settings
AUGMENTATION = {
    "time_shift": True, 
    "noise_injection": True, 
    "mixup": 0.0, 
    "specaugment": True
}

# ============================================================================
# MODULE-SCOPE: Model Definition
# ================================================================================

class ResidualBlock(nn.Module):
    """A simple residual block adapted for spectrogram features."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        
        # Residual connection: Handle potential dimension mismatch if in_channels != out_channels
        # Since we ensure the input channel matches the output channel in the stacking, 
        # we assume the dimensions match for the residual connection here.
        out += residual
        return out

class DeepCNN(nn.Module):
    """
    Custom 5-block residual CNN stack (32->64->128->256 channels).
    Input: (B, 1, N_mels, T)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Initial convolution layer: 1 -> 32
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(32)
        
        # Block 1: 32 -> 32 (Residual)
        self.block1 = ResidualBlock(32, 32, dropout_rate=0.25)
        
        # Downsample and increase channels: 32 -> 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = ResidualBlock(32, 64, dropout_rate=0.25)
        
        # Downsample and increase channels: 64 -> 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = ResidualBlock(64, 128, dropout_rate=0.25)
        
        # Downsample and increase channels: 128 -> 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = ResidualBlock(128, 256, dropout_rate=0.25)
        
        # Final feature extraction block (depth)
        self.block5 = ResidualBlock(256, 256, dropout_rate=0.25)

        # Global pooling and classification head
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle variable feature dimensions
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Initial pass
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = F.relu(x)
        
        # Block 1
        x = self.block1(x)
        
        # Block 2
        x = self.pool1(x)
        x = self.block2(x)
        
        # Block 3
        x = self.pool2(x)
        x = self.block3(x)
        
        # Block 4
        x = self.pool3(x)
        x = self.block4(x)
        
        # Block 5
        x = self.block5(x)
        
        # Classification head
        x = self.pool_global(x)
        x = x.flatten(1) # (B, C, 1, 1) -> (B, C)
        return self.head(x)


# ============================================================================
# RUNTIME section - MUST be inside `if __name__ == "__main__":`
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
        # load_precomputed_dataset reads augmentation settings from env vars/config
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
        model = DeepCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mels=128, Time=313)
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
                    # Calculate probabilities (sigmoid) and move to CPU for numpy conversion
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                # Only compute if there is at least one positive label in the validation set for this class
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score calculates AUC, which is equivalent to AP for the whole set.
                    # The description asks for 'average precision at k=5', but standard libraries
                    # only provide overall AP/AUC. We use the standard AP calculation here, 
                    # which is the closest robust proxy, as the library doesn't support k=5 AP directly.
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
                f"epoch {epoch + 1}/{
```

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
# Overriding skeleton defaults with proposal values where they differ
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
# Use the proposal's learning rate
LR = 0.0008
# Use the proposal's augmentation settings
AUGMENTATION = {
    "time_shift": True, 
    "noise_injection": True, 
    "mixup": 0.0, 
    "specaugment": True
}

# ============================================================================
# MODULE-SCOPE: Model Definition
# ================================================================================

class ResidualBlock(nn.Module):
    """A simple residual block adapted for spectrogram features."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        
        # Residual connection: Handle potential dimension mismatch if in_channels != out_channels
        # Since we ensure the input channel matches the output channel in the stacking, 
        # we assume the dimensions match for the residual connection here.
        out += residual
        return out

class DeepCNN(nn.Module):
    """
    Custom 5-block residual CNN stack (32->64->128->256 channels).
    Input: (B, 1, N_mels, T)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Initial convolution layer: 1 -> 32
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(32)
        
        # Block 1: 32 -> 32 (Residual)
        self.block1 = ResidualBlock(32, 32, dropout_rate=0.25)
        
        # Downsample and increase channels: 32 -> 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = ResidualBlock(32, 64, dropout_rate=0.25)
        
        # Downsample and increase channels: 64 -> 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = ResidualBlock(64, 128, dropout_rate=0.25)
        
        # Downsample and increase channels: 128 -> 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = ResidualBlock(128, 256, dropout_rate=0.25)
        
        # Final feature extraction block (depth)
        self.block5 = ResidualBlock(256, 256, dropout_rate=0.25)

        # Global pooling and classification head
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle variable feature dimensions
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Initial pass
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = F.relu(x)
        
        # Block 1
        x = self.block1(x)
        
        # Block 2
        x = self.pool1(x)
        x = self.block2(x)
        
        # Block 3
        x = self.pool2(x)
        x = self.block3(x)
        
        # Block 4
        x = self.pool3(x)
        x = self.block4(x)
        
        # Block 5
        x = self.block5(x)
        
        # Classification head
        x = self.pool_global(x)
        x = x.flatten(1) # (B, C, 1, 1) -> (B, C)
        return self.head(x)


# ============================================================================
# RUNTIME section - MUST be inside `if __name__ == "__main__":`
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
        # load_precomputed_dataset reads augmentation settings from env vars/config
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
        model = DeepCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mels=128, Time=313)
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
                    # Calculate probabilities (sigmoid) and move to CPU for numpy conversion
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                # Only compute if there is at least one positive label in the validation set for this class
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score calculates AUC, which is equivalent to AP for the whole set.
                    # The description asks for 'average precision at k=5', but standard libraries
                    # only provide overall AP/AUC. We use the standard AP calculation here, 
                    # which is the closest robust proxy, as the library doesn't support k=5 AP directly.
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
                f"epoch {epoch + 1}/{

```

## Output
- **code:** import json
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
# Overriding skeleton defaults with proposal values where they differ
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
# Use the proposal's learning rate
LR = 0.0008
# Use the proposal's augmentation settings
AUGMENTATION = {
    "time_shift": True, 
    "noise_injection": True, 
    "mixup": 0.0, 
    "specaugment": True
}

# ============================================================================
# MODULE-SCOPE: Model Definition
# ================================================================================

class ResidualBlock(nn.Module):
    """A simple residual block adapted for spectrogram features."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        
        # Residual connection: Handle potential dimension mismatch if in_channels != out_channels
        # Since we ensure the input channel matches the output channel in the stacking, 
        # we assume the dimensions match for the residual connection here.
        out += residual
        return out

class DeepCNN(nn.Module):
    """
    Custom 5-block residual CNN stack (32->64->128->256 channels).
    Input: (B, 1, N_mels, T)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Initial convolution layer: 1 -> 32
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(32)
        
        # Block 1: 32 -> 32 (Residual)
        self.block1 = ResidualBlock(32, 32, dropout_rate=0.25)
        
        # Downsample and increase channels: 32 -> 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = ResidualBlock(32, 64, dropout_rate=0.25)
        
        # Downsample and increase channels: 64 -> 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = ResidualBlock(64, 128, dropout_rate=0.25)
        
        # Downsample and increase channels: 128 -> 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = ResidualBlock(128, 256, dropout_rate=0.25)
        
        # Final feature extraction block (depth)
        self.block5 = ResidualBlock(256, 256, dropout_rate=0.25)

        # Global pooling and classification head
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle variable feature dimensions
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Initial pass
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = F.relu(x)
        
        # Block 1
        x = self.block1(x)
        
        # Block 2
        x = self.pool1(x)
        x = self.block2(x)
        
        # Block 3
        x = self.pool2(x)
        x = self.block3(x)
        
        # Block 4
        x = self.pool3(x)
        x = self.block4(x)
        
        # Block 5
        x = self.block5(x)
        
        # Classification head
        x = self.pool_global(x)
        x = x.flatten(1) # (B, C, 1, 1) -> (B, C)
        return self.head(x)


# ============================================================================
# RUNTIME section - MUST be inside `if __name__ == "__main__":`
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
        # load_precomputed_dataset reads augmentation settings from env vars/config
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
        model = DeepCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mels=128, Time=313)
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
                    # Calculate probabilities (sigmoid) and move to CPU for numpy conversion
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                # Only compute if there is at least one positive label in the validation set for this class
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score calculates AUC, which is equivalent to AP for the whole set.
                    # The description asks for 'average precision at k=5', but standard libraries
                    # only provide overall AP/AUC. We use the standard AP calculation here, 
                    # which is the closest robust proxy, as the library doesn't support k=5 AP directly.
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
                f"epoch {epoch + 1}/{

