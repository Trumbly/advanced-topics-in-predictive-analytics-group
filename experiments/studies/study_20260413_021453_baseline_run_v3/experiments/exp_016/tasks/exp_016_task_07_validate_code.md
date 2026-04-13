# Task exp_016_task_07_validate_code

- **Experiment:** exp_016
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-13 06:17:49.249686+00:00
- **Completed:** 2026-04-13 06:17:49.253497+00:00

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
# EPOCHS is read from an environment variable (default 1).
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class TimeAwareSelfAttention(nn.Module):
    """
    Time-Aware Self-Attention block operating on feature maps (B, C, T).
    Applies attention across the time dimension (T).
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, T, H) -- Note: We treat T as the "Height" dimension for 2D convs
        # The input shape is (B, C, T_frames). We need to reshape it to (B, C, T, 1) 
        # to match the standard (B, C, H, W) expected by Conv2d, treating T as H and 1 as W.
        
        # Reshape from (B, C, T) to (B, C, T, 1)
        B, C, T = x.size(0), x.size(1), x.size(2)
        x_reshaped = x.unsqueeze(-1) # (B, C, T, 1)

        # 1. Compute attention maps using 2D convolutions over the H (Time) dimension
        Q = self.query_conv(x_reshaped) # (B, C/r, T, 1)
        K = self.key_conv(x_reshaped)   # (B, C/r, T, 1)
        V = self.value_conv(x_reshaped) # (B, C, T, 1)

        # 2. Calculate Attention Scores (Q * K_T)
        # We perform element-wise multiplication across the (C/r, T, 1) dimensions.
        # This is a simplified dot-product attention across the T dimension.
        attention_map = torch.matmul(Q, K.transpose(-2, -1)) # (B, C/r, T, T)
        
        # Scale and Softmax
        scale = 1.0 / np.sqrt(Q.size(-2))
        attention_scores = (attention_map * scale).squeeze(-1) # (B, C/r, T, T)
        attention_weights = torch.softmax(attention_scores, dim=-1) # (B, C/r, T, T)

        # 3. Apply weights to V
        attended_output = torch.matmul(attention_weights, V) # (B, C/r, T, C) -> incorrect shape handling
        
        # Reverting to standard attention mechanism logic for simplicity:
        # Calculate attention weights (scores) over the time dimension T.
        
        # Use the standard approach: Attention(Q, K, V) = softmax( (Q*K^T) / sqrt(d_k) ) * V
        Q_t = self.query_conv(x_reshaped).squeeze(-1) # (B, C/r, T)
        K_t = self.key_conv(x_reshaped).squeeze(-1)   # (B, C/r, T)
        V_t = self.value_conv(x_reshaped).squeeze(-1) # (B, C, T)

        # Attention scores: (B, T, C/r) @ (B, C/r, T) -> (B, T, T)
        scores = torch.bmm(Q_t.unsqueeze(1).transpose(1, 2), K_t.unsqueeze(0)) / np.sqrt(Q_t.size(-1))
        weights = torch.softmax(scores, dim=-1) # (B, T, T)

        # Context vector: (B, T, T) @ (B, C, T) -> incorrect
        # We need to weight V along the time dimension.
        output = torch.bmm(weights.unsqueeze(2), V_t.unsqueeze(1)).squeeze(1) # (B, T, C)

        # Final projection
        output = self.output_conv(output.unsqueeze(-1)).squeeze(-1) # (B, C, T)

        return output


class BirdCLEFModel(nn.Module):
    """
    MobileNetV3Small backbone feeding into a Time-Aware Self-Attention block.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        
        # 1. Backbone: MobileNetV3Small
        # We assume MobileNetV3Small is available via pipelines.models
        from pipelines.models import MobileNetV3Small
        self.backbone = MobileNetV3Small(pretrained=False)
        
        # 2. Modify backbone to retain time dimension for attention
        # The backbone typically ends with AdaptiveAvgPool2d((1, 1)).
        # We must strip this and adapt the output.
        
        # We use the full backbone structure but intercept the output before global pooling.
        # The last layer before global pooling is usually the convolutional block.
        
        # We will override the forward pass to stop before the final pooling/flattening.
        self.attention = TimeAwareSelfAttention(in_channels=128)
        
        # 3. Classifier Head
        # Use LazyLinear to handle dynamic feature size from attention output.
        self.classifier_head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, T)
        
        # --- Pass through Backbone ---
        # To stop before global pooling, we temporarily detach the final layers.
        # We will manually run the initial convolutions.
        
        # In a real scenario, one would patch/monkey-patch the backbone.
        # For this controlled exercise, we assume the backbone output before the
        # final pooling layer (which usually reduces T to 1) gives us the
        # feature map (B, C_out, T_out, 1).
        
        # Since we cannot easily patch the internal structure, we will use the backbone
        # and capture the output *after* the main convolutional blocks but *before*
        # the final AdaptiveAvgPool2d((1,1)) that collapses time.
        
        # A common proxy is to run the backbone and then use AdaptiveAvgPool2d((1, T'))
        # to keep the time dimension T'. 
        
        # Best effort: Run the backbone, and then use the actual feature maps from the end.
        # Since the input depth is 128, and the backbone structure is complex, 
        # we will rely on the fact that the backbone output before the final pool is (B, C, H, W).
        
        # We first apply a dummy pass to let the backbone structure initialize its weights.
        # We must adapt the feature map size to feed the attention block.
        
        # For simplicity and stability, we will use a simplified feature extraction path:
        # 1. Initial Conv(1 -> 32)
        # 2. MaxPool
        # 3. Second Conv(32 -> 64)
        # 4. MaxPool
        # 5. Final Conv(64 -> 128)
        
        # Since we must use the MobileNetV3Small backbone, we must use its forward pass.
        # We will run the backbone and then adapt its output shape to (B, C, T).
        
        # --- Simplified Feature Extraction using Backbone ---
        
        # (B, 1, 128, T) -> (B, 128, T)
        x_feat = self.backbone(x)
        
        # After backbone, x_feat is often (B, C, 1, 1). We need (B, C, T).
        # We will use AdaptiveAvgPool2d((1, T)) to collapse the channel dimension
        # while keeping the time dimension T.
        
        # We must use the original time dimension T for attention.
        # We will use AdaptiveAvgPool2d((C_out, 1)) on the feature map (B, C_out, 1, 1)
        # to get (B, C_out, 1, 1), then unsqueeze to (B, C_out, 1, T) which is wrong.
        
        # The most robust way is to adapt the backbone's pooling mechanism:
        # 1. Run backbone (B, 1, 128, T) -> (B, C_out, 1, 1)
        # 2. Reshape: (B, C_out, 1, 1) -> (B, C_out, T, 1) (conceptually)
        # 3. Pass to Attention (B, C_out, T)
        
        # Let's assume the backbone output is (B, C_out, 1, 1) after pooling.
        # We re-introduce the time dimension T by using the original T.
        
        B, _, _, T = x.size()
        
        # Run the backbone: This will yield (B, C_out, 1, 1)
        pooled_features = self.backbone(x)
        
        # Reshape to (B, C_out, 1, 1) -> (B, C_out)
        C_out = pooled_features.size(1)
        
        # For time-aware attention, we need (B, C_out, T). 
        # We will use the original time dimension T for the attention block input.
        # We effectively treat the backbone output as a feature vector and broadcast it across time.
        
        # Reshape (B, C_out, 1, 1) -> (B, C_out, 1, T)
        features_expanded = pooled_features.unsqueeze(3).repeat(1, 1, 1, T).to(x.device) 
        
        # Pass through attention, which expects (B, C, T)
        attention_output = self.attention(features_expanded.permute(0, 1, 3, 2)) # (B, C_out, T)
        
        # Flatten the attention output across the time dimension (T)
        x_flat = attention_output.flatten(1) # (B, C_out * T)
        
        # Final classification
        logits = self.classifier_head(x_flat)
        return logits


if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Use the provided augmentation dictionary from the proposal
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 220: expected 'except' or 'finally' block
