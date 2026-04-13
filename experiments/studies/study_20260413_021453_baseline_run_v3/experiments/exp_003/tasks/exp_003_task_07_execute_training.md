# Task exp_003_task_07_execute_training

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-13 02:50:24.258738+00:00
- **Completed:** 2026-04-13 02:50:26.097941+00:00

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
# EPOCHS is read from BIRDCLEF_EPOCHS env var.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# =============================================================================
# MODEL DEFINITION: CNN-GRU Hybrid
# Architecture: 3-conv CNN front-end (1->32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head
# Input: (B, 1, 128, 313)
# =============================================================================

class CnnGruHybridModel(nn.Module):
    """
    CNN-GRU Hybrid model for spectrogram processing.
    Uses 3 consecutive Conv blocks to extract features, pools spatially,
    and passes the resulting feature sequence through a GRU.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. CNN Front-end (1 -> 32 -> 64) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # We use a third block to adhere to the "3-conv" description, 
        # but keep the final output channels at 64 as requested.
        # This block helps deepen the feature extraction.
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- 2. Pooling and Feature Flattening ---
        # Adaptive Average Pooling over the spatial dimensions (H, W)
        # to collapse the feature map, keeping only the depth/channels.
        # Output shape: (B, 64, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- 3. GRU Head ---
        # Input size to GRU must match the channel count after pooling (64).
        # The GRU processes the feature vector (64) across the time steps (T_reduced).
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Final linear layer to map the GRU's hidden state dimension (128) 
        # to the required number of classes.
        self.fc = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # x shape: (B, 1, 128, 313)
        
        # CNN Stack
        x = self.relu(self.conv1(x)) # (B, 32, 128, 313)
        x = self.relu(self.conv2(x)) # (B, 64, 128, 313)
        x = self.relu(self.conv3(x)) # (B, 64, 128, 313)
        
        # Pool spatial dimensions (H=128 -> 1, W=313 -> 1)
        # Result: (B, 64, 1, 1)
        x = self.pool(x)

        # Flatten spatial dimensions to prepare for GRU input
        # (B, 64, 1, 1) -> (B, 64, 1)
        x = x.squeeze(2).squeeze(2) 

        # For the GRU to process time, we need to expand the feature dimension
        # back to (B, C, T_reduced) where T_reduced is the original time steps.
        # Since we pooled down to 1x1, we must unsqueeze the feature dimension 
        # to represent the sequence length T_frames.
        # We assume the feature vector (64) should be used for *every* time step.
        # (B, 64) -> (B, 64, 1) -> (B, 64, T_frames) -- This requires T_frames.
        
        # Since the original input has 313 frames, we must unsqueeze the 
        # feature vector 313 times.
        # (B, 64) -> (B, 64, 1) -> (B, 64, 313)
        T_frames = x.size(2) # This is incorrect, x is (B, 64)
        
        # Re-evaluating the pooling step: If we use AdaptiveAvgPool2d((1, 1)), 
        # we lose all temporal information needed for the GRU.
        # To keep time, we must pool only the Height dimension:
        
        # Resetting the pooling strategy to retain time dimension (T=313)
        # We pool (B, 64, 128, 313) -> (B, 64, 1, 313)
        x = self.conv3(x)
        x = self.relu(x)
        
        # Pool Height dimension (128 -> 1)
        x = self.pool(x) # (B, 64, 1, 1) -- Wait, this is still wrong. The pool is 2D.
        
        # Let's use AvgPool2d on the Height dimension only, assuming a kernel size of 1
        # and only pooling the height. This is complex with pure Conv2d.
        
        # Easiest robust solution: Pool the spatial dimension H=128 down to 1, 
        # keeping T=313. This requires a manual reshape or specialized pool.
        
        # Standard approach: Use AdaptiveAvgPool2d((1, 1)) which pools both.
        # If the intent is GRU, the most common pattern is to use a CNN, then 
        # Flatten+Linear (if time is irrelevant) OR CNN+(Time Pool) -> GRU.
        
        # STICKING TO THE SKELETON'S ASSUMPTION: Global pool applied, 
        # but we must feed the GRU. We will assume the CNN output (B, 64, 1, 1) 
        # is replicated across the time dimension (313) for the GRU input.
        
        # (B, 64, 1, 1) -> (B, 64, 1) -> (B, 64, 313)
        # This requires knowing the original T

```

## Output
- **exit_code:** 0
- **duration_seconds:** 1.837830208009109
- **workdir:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260413_021453_baseline_run_v3/exp_003
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** NoResultsFile
- **message:** Script exited with status 0 but did not produce a results.json file in the working directory. Make sure your code writes `results.json` at the end of training.
