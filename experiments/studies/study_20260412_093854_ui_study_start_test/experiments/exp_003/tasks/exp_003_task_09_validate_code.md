# Task exp_003_task_09_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-12 07:46:04.096918+00:00
- **Completed:** 2026-04-12 07:46:04.098100+00:00

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

# === Hyperparameters ===
# EPOCHS is read from BIRDCLEF_EPOCHS env var, default 1.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 0.0005 # From proposal
AUGMENTATION = {"time_shift": True, "noise_injection": True} # Using subset of proposal augmentations + defaults

# === Registry Model Import (Simulated based on prompt instructions) ===
# In a real environment, this would be imported from pipelines.models
# We simulate the import for a known backbone structure.
try:
    # Attempt to import the specified registry model
    from pipelines.models import mobilenet_v3_small
    BACKBONE_MODEL_CLASS = mobilenet_v3_small
except ImportError:
    print("Warning: Could not import 'mobilenet_v3_small' from pipelines.models. Using a placeholder structure.", file=os.sys.stderr)
    # Fallback placeholder if the actual library isn't present for execution testing
    class MockMobileNetV3Small(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate the feature extractor part, assuming it handles (B, 1, H, W)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.features(x)
    BACKBONE_MODEL_CLASS = MockMobileNetV3Small

# === Custom Model Definition ===
# Adapter wrapper around the backbone to add the final classification head.
class MobileNetAdapter(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        
        # 1. Global pooling to collapse spatial dimensions (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2. LazyLinear handles the feature size dynamically, making it robust.
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, N_mels, T)
        
        # Pass through the feature extraction backbone
        x = self.backbone(x) 
        
        # Pool: (B, C, H, W) -> (B, C, 1, 1)
        x = self.pool(x) 
        
        # Flatten: (B, C, 1, 1) -> (B, C). Changed flatten(1) to view(B, -1) for stability.
        x = x.view(x.size(0), -1)  
        
        # Classification head: (B, C) -> (B, num_classes)
        return self.head(x)

# === Model Instantiation Placeholder ===
# This will be done in __main__ to ensure correct device placement
model_class = MobileNetAdapter


# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    
    try:
        print("loading data...", flush=True)
        # Load data using the fixed pipeline function
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation={"time_shift": True, "noise_injection": True},
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model and move to the device
        backbone = BACKBONE_MODEL_CLASS()
        model = model_class(backbone, num_classes)
        model = model.to(device)
        
        # FIX: Initialize LazyLinear parameters by passing a dummy input through the model
        # This resolves the "uninitialized parameter" error when calculating parameters.
        dummy_input = torch.zeros(1, 1, 1, 1).to(device, dtype=torch.float32)
        with torch.no_grad():
            _ = model(dummy_input)
            
        # Calculate parameters based on the final structure
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Per-class pos_weight from the DatasetProfile
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
                x = x.

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 142: invalid syntax
