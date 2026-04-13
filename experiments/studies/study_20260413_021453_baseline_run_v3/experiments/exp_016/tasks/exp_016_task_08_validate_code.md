# Task exp_016_task_08_validate_code

- **Experiment:** exp_016
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 06:20:14.413267+00:00
- **Completed:** 2026-04-13 06:20:14.414170+00:00

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
from torchvision.models import mobilenet_v3_small

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# Note: The actual values are read from environment variables by the skeleton block.
# We only define the structure constants here.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# --- Custom Components ---

class ChannelAdapt(nn.Module):
    """Adapts the initial convolution layer of a backbone to accept 1 input channel."""
    def __init__(self, backbone_module):
        super().__init__()
        # MobileNetV3's initial layer expects 3 channels (RGB). We replace it.
        # We assume the first layer is a Conv2d.
        initial_conv = backbone_module[0]
        # Create a new conv layer that accepts 1 channel but outputs the same channels
        self.adapted_conv = nn.Conv2d(1, initial_conv.out_channels, kernel_size=initial_conv.kernel_size, stride=initial_conv.stride, padding=initial_conv.padding)
        # The rest of the backbone layers remain the same.
        self.backbone = nn.Sequential(*[
            self.adapted_conv,
            *[nn.ModuleList([nn.Conv2d(i.in_channels, i.out_channels, i.kernel_size, i.stride, i.padding)) for i in backbone_module[1:]]
        ], params=True) # This is a placeholder structure, direct replacement is safer

    def __init__(self, backbone_module):
        super().__init__()
        # Safely replace the first layer of the backbone
        # We iterate through the module list and replace the first Conv2d
        layers = list(backbone_module)
        if not isinstance(layers[0], nn.Conv2d):
             raise TypeError("Expected backbone to start with nn.Conv2d")

        original_conv = layers[0]
        # Create a new adapted first layer
        adapted_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=True
        )
        
        # Rebuild the sequential model with the adapted layer
        self.backbone = nn.Sequential(adapted_conv, *layers[1:])

    def forward(self, x):
        return self.backbone(x)

class TimeAwareAttention(nn.Module):
    """
    Squeeze-and-Excitation style attention mechanism applied after feature pooling.
    Operates on the feature channel dimension (C').
    Input x shape: (B, C', 1, 1)
    Output shape: (B, C')
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation layer: Global Avg Pool -> Reduction -> Expansion -> Sigmoid
        self.se_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C', H', W')
        # 1. Global Average Pooling over spatial dimensions (H', W')
        y = self.avg_pool(x) # (B, C', 1, 1)
        
        # 2. Squeeze-and-Excitation mechanism
        y = self.se_block(y) # (B, C', 1, 1)
        
        # 3. Reshape to (B, C') while keeping the attention weights
        return y.view(y.size(0), self.feature_dim).unsqueeze(-1) # (B, C', 1) for consistency if needed, but (B, C') is fine for Linear
        
class AttentionCNN(nn.Module):
    """
    MobileNetV3 Small backbone followed by Time-Aware Self-Attention.
    """
    def __init__(self, num_classes):
        super().__init__()
        # 1. Backbone Initialization and Adaptation
        mobilenet = mobilenet_v3_small(pretrained=True)
        # We only use the feature extraction part, excluding the final classification head
        self.backbone = ChannelAdapt(mobilenet.features)

        # Determine the feature dimension (C') after the backbone and pooling
        # The last layer of the backbone determines the channel size.
        # For MobileNetV3-Small, after features, the output channels are typically 128.
        # We rely on the structure: C' = output channels of the last conv layer.
        # Since we can't easily extract this dynamically without running the model,
        # we must assume the final channel count based on the original model structure (128).
        self.feature_dim = 128 
        
        # 2. Attention Block
        self.attention = TimeAwareAttention(self.feature_dim)

        # 3. Adaptive Pooling and Classifier Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Use LazyLinear for the final classification layer (B, C')
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input x: (B, 1, C, T)
        
        # Backbone feature extraction
        x = self.backbone(x) # (B, C', H', W')
        
        # Apply Attention (This step modifies the feature representation)
        # The attention block expects (B, C', H', W') and outputs (B, C') or (B, C', 1, 1)
        x_att = self.attention(x) # (B, C', 1, 1)
        
        # Pool to remove spatial dependence (should already be close)
        x_pooled = self.pool(x_att) # (B, C', 1, 1)
        
        # Flatten: (B, C')
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        
        # Final classification
        logits = self.classifier(x_flat) # (B, num_classes)
        return logits

# ===============================================================================
# MODULE-SCOPE section
# ===============================================================================

# Imports already at top level
# torch, nn, np, json, os, time, roc_auc_score, average_precision_score, f1_score, 
# load_precomputed_dataset, compute_pos_weight

# Model definition is done above the main block

# ===============================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ===============================================================================

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Augmentation specified in JSON proposal: time_shift=False, noise_injection=False, mixup=0.0, specaugment=True
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation={"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True},
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = AttentionCNN(num_classes=num_classes)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Required because the classifier head is lazy.
        with torch.no_grad():
            # Dummy input shape: (1, 1, 128, 313)
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
                    # Calculate probabilities (sigmoid on logits)
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Only calculate if there are positive examples in the validation set for the class
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # Note: average_precision_score is used as a proxy for class-mean AP for simplicity/stability
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 36: closing parenthesis ')' does not match opening parenthesis '['
