# Task exp_004_task_03_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-12 07:47:22.835862+00:00
- **Completed:** 2026-04-12 07:47:22.836184+00:00

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
from pipelines.models import EfficientNetB0

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters derived from Proposal ===
# Using proposal LR: 0.0008
LR = 0.0008
# Using proposal weight decay
WEIGHT_DECAY = 0.0001
# Using proposal dropout
DROPOUT = 0.25

# Read EPOCHS from env var, default "1"
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))

# Augmentation from proposal
AUGMENTATION = {
    "time_shift": True,
    "noise_injection": True,
    "mixup": 0.5,
    "specaugment": True
}

# === Custom Model Definition: Time-Domain Attention Encoder ===
class TimeAttentionEncoder(nn.Module):
    """
    Applies self-attention across the time dimension of the feature map.
    Input shape: (B, C_feat, H', W')
    Output shape: (B, C_feat, H', W') (with attention weights applied)
    """
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim # C_feat
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.key = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.value = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C_feat, H', W')
        B, C, H, W = x.size()

        Q = self.query(x).view(B, C, 1, H * W).permute(0, 2, 3, 1).contiguous() # (B, H*W, C/H) -> (B, Time, Head_Dim)
        K = self.key(x).view(B, C, 1, H * W).permute(0, 2, 3, 1).contiguous()
        V = self.value(x).view(B, C, 1, H * W).permute(0, 2, 3, 1).contiguous()

        # Q: (B, T, D_h), K: (B, T, D_h), V: (B, T, D_h)
        # Attention: (B, T, D_h) * (B, T, D_h) -> (B, T, T)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply softmax over the time dimension (last dimension)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Context vector: (B, T, D_h)
        context = torch.matmul(attention_weights, V)

        # Reshape back to image format for the output conv layer
        # (B, Time, Head_Dim) -> (B, Head_Dim, Time)
        context = context.permute(0, 2, 1).contiguous()
        
        # This requires us to know the original H' and W' sizes to reshape back correctly.
        # Since we flattened H'*W' into the sequence dimension, we must assume the 
        # output feature size is (B, C, H', W') where H' and W' are derived from 
        # the original spatial dimensions. 
        
        # For simplicity and stability in this constrained environment, we will assume
        # the output of the attention mechanism is meant to be spatial features,
        # and we'll reshape the context vector back to the original (H', W') structure
        # while keeping the channel depth the same.
        
        # The context shape is (B, C/H, H'*W'). We need to reshape it to (B, C, H', W').
        # This is tricky without knowing H' and W' explicitly.
        
        # Fallback: Just use the context as the feature map for the final convolution.
        # The output shape will be (B, C_feat, H', W') if we treat the last dimension
        # as the time dimension and pool the rest.
        
        # Given the constraints, we will treat the output (B, C_feat, H', W') as a feature 
        # map derived from the attention context.
        output = self.output_linear(context)
        return output

class BirdCLEFModel(nn.Module):
    """
    EfficientNetB0 backbone feeding into Time-Domain Self-Attention Encoder.
    """
    def __init__(self, num_classes):
        super().__init__()
        # 1. Backbone: EfficientNetB0
        self.backbone = EfficientNetB0(pretrained=True)
        
        # EfficientNetB0 outputs (B, C_feat, H', W')
        # We need to determine C_feat dynamically, but for simplicity, we assume
        # the last layer feature count is the input to the attention block.
        
        # 2. Time Attention Encoder
        # We assume C_feat is the output channels of the backbone.
        # We rely on the forward pass to determine this.
        self.attention = TimeAttentionEncoder(
            embed_dim=640, # Placeholder: This value must match the backbone output channels
            num_heads=8,
            dropout_rate=DROPOUT
        )
        
        # 3. Final Classifier Head
        # We use LazyLinear to handle arbitrary spatial pooling output sizes.
        self.head = nn.LazyLinear(num_classes)
        
        # The backbone output channels for EfficientNetB0 are usually 1280.
        # We must initialize the attention block with the correct channel count (1280).
        # Re-initializing the model structure to fix placeholder values:
        self.attention.attention = TimeAttentionEncoder(
            embed_dim=1280, # Corrected based on standard EfficientNetB0 output
            num_heads=8,
            dropout_rate=DROPOUT
        )

    def forward(self, x):
        # Input x: (B, 1, C, T)
        
        # 1. Backbone feature extraction
        # EfficientNetB0 expects (B, 3, H, W), but we pass (B, 1, H, W)
        x_feat = self.backbone(x) # Output: (B, C_feat, H', W')

        # 2. Time-Domain Self-Attention
        # The attention block processes the feature map.
        x_attn = self.attention(x_feat) # Output: (B, C_feat, H', W')

        # 3. Global Pooling and Classification
        # Pool spatially to remove H' and W' dependencies, leaving (B, C_feat).
        x_pooled = F.adaptive_avg_pool2d(x_attn, (1, 1)) # (B, C_feat, 1, 1)
        x_flat = x_pooled.flatten(2) # (B, C_feat)
        
        # 4. Final Linear Layer
        return self.head(x_flat)


# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =============================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / persistent_workers / prefetch_factor
        # are read from BIRDCLEF_* env vars (sourced from config.yaml).
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 177: unterminated string literal (detected at line 177)
