# Task exp_010_task_03_validate_code

- **Experiment:** exp_010
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 00:01:37.113369+00:00
- **Completed:** 2026-04-13 00:01:37.114339+00:00

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

# === Hardcode hyperparameters from the proposal ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# are read from BIRDCLEF_* env vars (sourced from config.yaml).
#
# EPOCHS is read from an env var, defaulting to 1.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Augmentation settings from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnAttentionModel(nn.Module):
    """
    Implements the proposed 3-block CNN stack -> Attention (simulated via advanced pooling) -> Linear Head.
    The architecture processes the input (B, 1, Mel, Time) through 3 Conv layers
    with intermediate feature expansion, ending with 64 channels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 3-block Conv Stack (Targeting 64 output channels) ---
        # Input: (B, 1, 128, 313)
        # Block 1: 1 -> 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Block 2: 64 -> 128 (Intermediate expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Block 3: 128 -> 64 (Final projection to 64 channels)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Activation functions (Using functional API for simplicity)
        self.relu = nn.ReLU(inplace=True)

        # --- Attention / Feature Aggregation ---
        # We pool over the Mel dimension (H) to collapse the feature space,
        # keeping the temporal dimension (T) intact for pseudo-attention.
        # Input to pool: (B, 64, H, T). Output: (B, 64, 1, T).
        self.pool = nn.AdaptiveAvgPool2d((1, None)) # Pool only over Mel dimension (H)
        
        # After pooling/reshaping, the feature vector is (B, 64, T).
        # We flatten the 64 channels and the T time frames together for the linear head.
        # The final feature dimension size is 64 * T_new. This is too large for LazyLinear.
        # Instead, we pool the time dimension as well, yielding a fixed feature size:
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) # Pool over H and T dimensions
        
        # --- Linear Head ---
        # Use LazyLinear to handle the inferred input size (64 channels)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Shape: (B, 64, H, T)

        # 2. Global Pooling (Collapses H and T to 1x1)
        x = self.global_pool(x)
        # Shape: (B, 64, 1, 1)

        # 3. Flatten and Linear Projection
        #

```

## Output
- **validation:** passed
- **code_bytes:** 3222
