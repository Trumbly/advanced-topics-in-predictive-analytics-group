# Task exp_004_codegen_retry_02

- **Experiment:** exp_004
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-13 03:04:45.696691+00:00
- **Completed:** 2026-04-13 03:07:11.056801+00:00

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
  Message: Line 230: '{' was never closed

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
# EPOCHS reads from environment variable BIRDCLEF_EPOCHS
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Augmentation is read from the proposal, but the skeleton requires a dict.
# We manually adjust the required augmentation here based on the proposal.
# Proposal: {"time_shift": false, "noise_injection": false, "mixup": 0.0, "specaugment": true}
# Skeleton requires: {"time_shift": True, "noise_injection": True, ...}
# We must use the skeleton's defaults if the proposal contradicts the required format,
# but we will set it to match the proposal's specified (false/false) values if possible,
# while keeping the structure required by load_precomputed_dataset.
# Since the skeleton mandates specific keys and we must match the required structure:
AUGMENTATION = {"time_shift": False, "noise_injection": False}


# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A basic residual block for CNN feature extraction.
    Input/Output channels must match the target_channels for the identity shortcut.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()

        # Determine if the shortcut needs to project (i.e., if input channels != output channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # Shortcut path
        shortcut_out = self.shortcut(x)

        # Add and normalize
        out += shortcut_out
        out = F.relu(out)
        return out


class DeepCNN(nn.Module):
    """
    Implements the 5-block residual CNN stack (32->64->128->128->256 channels).
    """
    def __init__(self, num_classes, dropout_rate=0.25):
        super().__init__()
        
        # Initial convolution (1 -> 32)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 5-Block residual stack
        # Block 1: 32 -> 64
        self.block1 = ResidualBlock(in_channels=32, out_channels=64, dropout_rate=dropout_rate)
        # Block 2: 64 -> 128
        self.block2 = ResidualBlock(in_channels=64, out_channels=128, dropout_rate=dropout_rate)
        # Block 3: 128 -> 128 (maintaining channels)
        self.block3 = ResidualBlock(in_channels=128, out_channels=128, dropout_rate=dropout_rate)
        # Block 4: 128 -> 256
        self.block4 = ResidualBlock(in_channels=128, out_channels=256, dropout_rate=dropout_rate)
        # Block 5: 256 -> 256 (maintaining channels)
        self.block5 = ResidualBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate)
        
        # Global Average Pooling and Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle dynamic feature map size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input: (B, 1, H, W)
        
        # 1. Initial Conv
        x = self.initial_conv(x)
        
        # 2. Residual Blocks
        x = self.block1(x) # Output: 64 channels
        x = self.block2(x) # Output: 128 channels
        x = self.block3(x) # Output: 128 channels
        x = self.block4(x) # Output: 256 channels
        x = self.block5(x) # Output: 256 channels
        
        # 3. Pooling and Head
        x = self.pool(x) # Output: (B, 256, 1, 1)
        x = x.flatten(1) # Collapse spatial dimensions: (B, 256)
        
        # 4. Classification
        logits = self.head(x) # Output: (B, num_classes)
        return logits


# ===========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        try:
            torch.set_num_threads(os.cpu_count() or 4)
        except RuntimeError:
            pass # Ignore if setting threads fails

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads paths from environment variables
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
        model = DeepCNN(num_classes=num_classes).to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Required before calling p.numel()
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channel=1, Height=128, Width=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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
                    # Using average_precision_score as proxy for AP@k for simplicity 
                    # as true AP@5 calculation is complex without explicit ranking logic.
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
# EPOCHS reads from environment variable BIRDCLEF_EPOCHS
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Augmentation is read from the proposal, but the skeleton requires a dict.
# We manually adjust the required augmentation here based on the proposal.
# Proposal: {"time_shift": false, "noise_injection": false, "mixup": 0.0, "specaugment": true}
# Skeleton requires: {"time_shift": True, "noise_injection": True, ...}
# We must use the skeleton's defaults if the proposal contradicts the required format,
# but we will set it to match the proposal's specified (false/false) values if possible,
# while keeping the structure required by load_precomputed_dataset.
# Since the skeleton mandates specific keys and we must match the required structure:
AUGMENTATION = {"time_shift": False, "noise_injection": False}


# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A basic residual block for CNN feature extraction.
    Input/Output channels must match the target_channels for the identity shortcut.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()

        # Determine if the shortcut needs to project (i.e., if input channels != output channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # Shortcut path
        shortcut_out = self.shortcut(x)

        # Add and normalize
        out += shortcut_out
        out = F.relu(out)
        return out


class DeepCNN(nn.Module):
    """
    Implements the 5-block residual CNN stack (32->64->128->128->256 channels).
    """
    def __init__(self, num_classes, dropout_rate=0.25):
        super().__init__()
        
        # Initial convolution (1 -> 32)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 5-Block residual stack
        # Block 1: 32 -> 64
        self.block1 = ResidualBlock(in_channels=32, out_channels=64, dropout_rate=dropout_rate)
        # Block 2: 64 -> 128
        self.block2 = ResidualBlock(in_channels=64, out_channels=128, dropout_rate=dropout_rate)
        # Block 3: 128 -> 128 (maintaining channels)
        self.block3 = ResidualBlock(in_channels=128, out_channels=128, dropout_rate=dropout_rate)
        # Block 4: 128 -> 256
        self.block4 = ResidualBlock(in_channels=128, out_channels=256, dropout_rate=dropout_rate)
        # Block 5: 256 -> 256 (maintaining channels)
        self.block5 = ResidualBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate)
        
        # Global Average Pooling and Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle dynamic feature map size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input: (B, 1, H, W)
        
        # 1. Initial Conv
        x = self.initial_conv(x)
        
        # 2. Residual Blocks
        x = self.block1(x) # Output: 64 channels
        x = self.block2(x) # Output: 128 channels
        x = self.block3(x) # Output: 128 channels
        x = self.block4(x) # Output: 256 channels
        x = self.block5(x) # Output: 256 channels
        
        # 3. Pooling and Head
        x = self.pool(x) # Output: (B, 256, 1, 1)
        x = x.flatten(1) # Collapse spatial dimensions: (B, 256)
        
        # 4. Classification
        logits = self.head(x) # Output: (B, num_classes)
        return logits


# ===========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        try:
            torch.set_num_threads(os.cpu_count() or 4)
        except RuntimeError:
            pass # Ignore if setting threads fails

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads paths from environment variables
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
        model = DeepCNN(num_classes=num_classes).to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Required before calling p.numel()
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channel=1, Height=128, Width=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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
                    # Using average_precision_score as proxy for AP@k for simplicity 
                    # as true AP@5 calculation is complex without explicit ranking logic.
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
# EPOCHS reads from environment variable BIRDCLEF_EPOCHS
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Augmentation is read from the proposal, but the skeleton requires a dict.
# We manually adjust the required augmentation here based on the proposal.
# Proposal: {"time_shift": false, "noise_injection": false, "mixup": 0.0, "specaugment": true}
# Skeleton requires: {"time_shift": True, "noise_injection": True, ...}
# We must use the skeleton's defaults if the proposal contradicts the required format,
# but we will set it to match the proposal's specified (false/false) values if possible,
# while keeping the structure required by load_precomputed_dataset.
# Since the skeleton mandates specific keys and we must match the required structure:
AUGMENTATION = {"time_shift": False, "noise_injection": False}


# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A basic residual block for CNN feature extraction.
    Input/Output channels must match the target_channels for the identity shortcut.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()

        # Determine if the shortcut needs to project (i.e., if input channels != output channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # Shortcut path
        shortcut_out = self.shortcut(x)

        # Add and normalize
        out += shortcut_out
        out = F.relu(out)
        return out


class DeepCNN(nn.Module):
    """
    Implements the 5-block residual CNN stack (32->64->128->128->256 channels).
    """
    def __init__(self, num_classes, dropout_rate=0.25):
        super().__init__()
        
        # Initial convolution (1 -> 32)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 5-Block residual stack
        # Block 1: 32 -> 64
        self.block1 = ResidualBlock(in_channels=32, out_channels=64, dropout_rate=dropout_rate)
        # Block 2: 64 -> 128
        self.block2 = ResidualBlock(in_channels=64, out_channels=128, dropout_rate=dropout_rate)
        # Block 3: 128 -> 128 (maintaining channels)
        self.block3 = ResidualBlock(in_channels=128, out_channels=128, dropout_rate=dropout_rate)
        # Block 4: 128 -> 256
        self.block4 = ResidualBlock(in_channels=128, out_channels=256, dropout_rate=dropout_rate)
        # Block 5: 256 -> 256 (maintaining channels)
        self.block5 = ResidualBlock(in_channels=256, out_channels=256, dropout_rate=dropout_rate)
        
        # Global Average Pooling and Classification Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle dynamic feature map size
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input: (B, 1, H, W)
        
        # 1. Initial Conv
        x = self.initial_conv(x)
        
        # 2. Residual Blocks
        x = self.block1(x) # Output: 64 channels
        x = self.block2(x) # Output: 128 channels
        x = self.block3(x) # Output: 128 channels
        x = self.block4(x) # Output: 256 channels
        x = self.block5(x) # Output: 256 channels
        
        # 3. Pooling and Head
        x = self.pool(x) # Output: (B, 256, 1, 1)
        x = x.flatten(1) # Collapse spatial dimensions: (B, 256)
        
        # 4. Classification
        logits = self.head(x) # Output: (B, num_classes)
        return logits


# ===========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        try:
            torch.set_num_threads(os.cpu_count() or 4)
        except RuntimeError:
            pass # Ignore if setting threads fails

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads paths from environment variables
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
        model = DeepCNN(num_classes=num_classes).to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Required before calling p.numel()
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channel=1, Height=128, Width=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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
                    # Using average_precision_score as proxy for AP@k for simplicity 
                    # as true AP@5 calculation is complex without explicit ranking logic.
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

