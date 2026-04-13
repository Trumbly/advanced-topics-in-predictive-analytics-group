# Task exp_016_task_03_validate_code

- **Experiment:** exp_016
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 06:08:09.023749+00:00
- **Completed:** 2026-04-13 06:08:09.028870+00:00

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

# Registry models must be imported at the top level
from pipelines.models import MobileNetV3Small

# --- Device selection ---
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# --- Hyperparameters ---
# Read epochs from environment variable, default to 1
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Adaptation based on proposal: time_shift=false, noise_injection=false, specaugment=true
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# Helper functions from the pipeline
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight


# ==============================================================================
# CUSTOM MODULE DEFINITIONS (Module Scope)
# ==============================================================================

class TimeAwareAttention(nn.Module):
    """
    A simple Temporal/Channel Attention block applied after the backbone.
    It calculates attention weights across the feature channels (C) 
    and multiplies them back to the feature map.
    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        avg = self.avg_pool(x) # (B, C, 1, 1)
        attention_weights = self.channel_attention(avg) # (B, 1, 1, 1)
        
        # Reshape weights to allow broadcasting multiplication: (B, 1, 1, 1) * (B, C, H, W)
        return x * attention_weights


class AttentionCNN(nn.Module):
    """
    MobileNetV3 Small backbone -> Time-Aware Attention -> Pooling -> Linear Head
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone (Registry Model)
        # The backbone handles the initial feature extraction from (B, 1, 128, 313)
        self.backbone = MobileNetV3Small(pretrained=True)
        
        # Determine the output channels of the backbone for the attention block.
        # Typically, MNetV3 has a final feature channel count (e.g., 128 or 256).
        # For simplicity and robustness, we assume the last convolutional layer 
        # before the final pooling/head in the standard MNetV3 structure 
        # is what we pass through attention.
        # Since we are not modifying MNetV3's internal structure, we must 
        # rely on its output feature map size. We'll use the feature 
        # map output before the final global pooling if possible.
        
        # A safe approach is to wrap the backbone and identify its feature output.
        # For this exercise, we assume the backbone output needs to be reduced 
        # to a fixed channel size suitable for the attention block.
        
        # For MobileNetV3Small, let's assume the feature map has 128 channels 
        # after initial feature extraction layers that we want to modulate.
        # We use the last conv layer's output channels as the input to attention.
        
        # Because we cannot easily hook into the internal feature map before 
        # the final pooling of the registry model, we will simplify: 
        # 1. Pass through backbone.
        # 2. Apply Attention.
        # 3. Pool and Linear.
        
        # We must check the actual output channel count of the backbone. 
        # If we pass the whole backbone output, the output channel is determined 
        # by its final layer. We will use the output feature dimension of the 
        # backbone's feature maps for the attention block input.
        
        # Due to the black-box nature of the registry model's internal flow, 
        # we will apply the attention mechanism immediately after the backbone 
        # and before adaptive pooling. We assume the feature dimension is 128 
        # (a common size for MobileNet variants).
        
        attention_channels = 128 # Assuming MNetV3 outputs 128 features before final pooling
        self.attention = TimeAwareAttention(in_channels=attention_channels)
        
        # 2. Final Classification Head
        # Use AdaptiveAvgPool2d to collapse spatial dims (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use LazyLinear to handle the unknown input feature size robustly
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, 313)
        
        # 1. Backbone Feature Extraction
        # The backbone processes the spectrogram through its initial layers.
        # We need to manually extract the feature map before final global pooling.
        # Since we cannot easily modify the registered backbone's internal 
        # architecture to expose an intermediate feature map, we rely on 
        # the fact that the backbone will process the input and we will 
        # apply attention to its raw output feature map.
        
        # Note: This implementation assumes the backbone's output dimension 
        # is suitable for the attention module's input channels (128).
        features = self.backbone(x) # (B, C_feat, H', W')
        
        # 2. Attention Modulation
        attended_features = self.attention(features) # (B, C_feat, H', W')
        
        # 3. Pooling and Classification
        pooled_features = self.pool(attended_features) # (B, C_feat, 1, 1)
        
        # Flatten (B, C_feat)
        x = pooled_features.flatten(1)
        
        # Final Linear Layer
        logits = self.head(x) # (B, num_classes)
        return logits


# ==============================================================================
# RUNTIME BLOCK
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
        # Load dataset using class-specific augmentation settings
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here.
        model = AttentionCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Use the input shape (1, 128, 313) for the dummy tensor.
        dummy = torch.zeros(1, 1, 128, 313, device=device, dtype=torch.float32)
        with torch.no_grad():
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Per-class pos_weight from the DatasetProfile
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
                # Move each batch to the training device.
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
                    # Note: For BirdCLEF, average_precision_score is used as a proxy
                    # for class-

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 243: expected an indented block after 'if' statement on line 241
