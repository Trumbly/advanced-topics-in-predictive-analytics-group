# Task exp_007_task_18_validate_code

- **Experiment:** exp_007
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 04:04:04.839681+00:00
- **Completed:** 2026-04-13 04:04:04.845386+00:00

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
# Overriding skeleton default LR with proposed value 0.0008
LR = 0.0008
# Dropout rate from proposal
DROPOUT_RATE = 0.25
# Epochs are read from env var, default falls back to 1 if not set.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
AUGMENTATION = {"time_shift": True, "noise_injection": True}


# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """A residual block for Conv2d feature extraction."""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        
        if in_c != out_c:
            # If input and output channels differ, adjust the shortcut path
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out += residual
        out = self.dropout(out)
        return out

class DeepCNN(nn.Module):
    """Custom 5-block residual CNN stack (32->64->128->256 channels)."""
    def __init__(self, num_classes, dropout_rate):
        super().__init__()
        
        # Initial convolution layer (Input: 1 channel)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 32 -> 64 (Block 1)
        self.block1 = ResidualBlock(32, 64, dropout_rate=dropout_rate)
        
        # 64 -> 128 (Block 2)
        self.block2 = ResidualBlock(64, 128, dropout_rate=dropout_rate)
        
        # 128 -> 256 (Block 3)
        self.block3 = ResidualBlock(128, 256, dropout_rate=dropout_rate)
        
        # 256 -> 256 (Block 4 - Final feature extraction block)
        self.block4 = ResidualBlock(256, 256, dropout_rate=dropout_rate)

        # Final pooling and classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Use LazyLinear to handle variable spatial dimensions
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # B, 1, 128, 313
        x = self.initial_conv(x)
        
        # 32 -> 64
        x = self.block1(x)
        
        # 64 -> 128
        x = self.block2(x)
        
        # 128 -> 256
        x = self.block3(x)
        
        # 256 -> 256
        x = self.block4(x)
        
        # Pool: (B, 256, 1, 1)
        x = self.pool(x)
        
        # Flatten: (B, 256)
        x = torch.flatten(x, 2)
        
        # Final Linear layer
        return self.head(x)


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
        # FIX: Reducing batch size to mitigate MPS OOM errors.
        # We pass a small, explicit batch_size override (e.g., 64) to the loader.
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
            batch_size=64 # Reduced batch size to prevent Out of Memory on MPS
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = DeepCNN(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Using a dummy tensor matching the input shape (B, 1, 128, 313)
            dummy = torch.zeros(

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 154: '(' was never closed
