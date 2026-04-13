# Task exp_006_task_18_validate_code

- **Experiment:** exp_006
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-13 03:43:46.229664+00:00
- **Completed:** 2026-04-13 03:43:46.230427+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pipelines.models import TorchvisionAdapter
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
import torch.nn.functional as F

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# Note: We override the skeleton's LR for consistency with the proposal (0.001)
LR = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT_RATE = 0.15
# Epochs must read from env var
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
AUGMENTATION = {"time_shift": True, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class MobileNetV3SpectroModel(nn.Module):
    """
    MobileNetV3-Small backbone adapted for 1-channel spectrogram input,
    followed by global pooling and a classification head.
    """
    def __init__(self, num_classes, dropout_rate=0.15):
        super().__init__()
        
        # 1. Load the standard MobileNetV3 backbone using the adapter
        self.backbone = TorchvisionAdapter("mobilenet_v3_small", pretrained=False)
        
        # 2. Adapt the input layer: Change input channels from 3 to 1
        # The adapter handles the structure, so we modify the first conv layer directly.
        original_conv = self.backbone.features[0]
        
        # Get the number of output channels from the original layer
        out_channels = original_conv.out_channels
        
        # Replace the first layer with nn.Conv2d manually, as its internal structure
        # is causing attribute errors when accessing .kernel_size or .padding.
        self.backbone.features[0] = nn.Conv2d(
            in_channels=1, out_channels=out_channels, 
            kernel_size=3, stride=2, padding=1, bias=True
        )
        
        # 3. The rest of the backbone layers remain the same.
        
        # 4. Global Pooling and Classification Head
        # Use AdaptiveAvgPool2d to collapse the spatial dimensions (H, W) to (1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use LazyLinear to handle the dynamically sized input from the backbone
        self.head = nn.LazyLinear(num_classes)
        
        # Apply dropout

```

## Output
- **validation:** passed
- **code_bytes:** 2507
