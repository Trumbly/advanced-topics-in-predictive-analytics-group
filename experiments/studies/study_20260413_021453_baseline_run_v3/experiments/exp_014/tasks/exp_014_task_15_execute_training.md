# Task exp_014_task_15_execute_training

- **Experiment:** exp_014
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-13 05:48:18.800282+00:00
- **Completed:** 2026-04-13 05:48:20.579430+00:00

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
from pipelines.models import CnnSmallV1

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# Per mandate, hardcap EPOCHS to 1.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        self.backbone = CnnSmallV1()

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Temporal reshaping: (B, 128, 1, 1

```

## Output
- **exit_code:** 0
- **duration_seconds:** 1.778415832988685
- **workdir:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260413_021453_baseline_run_v3/exp_014
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** NoResultsFile
- **message:** Script exited with status 0 but did not produce a results.json file in the working directory. Make sure your code writes `results.json` at the end of training.
