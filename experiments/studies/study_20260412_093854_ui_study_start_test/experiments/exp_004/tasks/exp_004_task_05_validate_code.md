# Task exp_004_task_05_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 07:48:42.982003+00:00
- **Completed:** 2026-04-12 07:48:42.987872+00:00

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
EPOCHS = 1

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
        
        # The context shape is (B, C/H, H'*W'). 
        # We treat the output (B, C_feat, H', W') as a feature 
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
        
        # 2. Time Attention Encoder
        # We use 1280 as the correct input dimension for EfficientNetB0 output.
        self.attention = TimeAttentionEncoder(
            embed_dim=1280, 
            num_heads=8,
            dropout_rate=DROPOUT
        )
        
        # 3. Final Classifier Head
        self.head = nn.LazyLinear(num_classes)
        

    def forward(self, x):
        # Input x: (B, 1, H, W) - Note: Input shape needs adjustment for backbone
        
        # 1. Backbone feature extraction
        x_feat = self.backbone(x) # Output: (B, C_feat, H', W')

        # 2. Time-Domain Self-Attention
        x_attn = self.attention(x_feat) # Output: (B, C_feat, H', W')

        # 3. Global Pooling and Classification
        x_pooled = F.adaptive_avg_pool2d(x_attn, (1, 1)) # (B, C_feat, 1, 1)
        x_flat = x_pooled.flatten(2) # (B, C_feat)
        
        # 4. Final Linear Layer
        return self.head(x_flat)


# ==========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ==========================================================================
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
        print(f"data loaded: {num_classes} classes detected.", flush=True)

        # Model initialization
        model = BirdCLEFModel(num_classes=num_classes).to(device)
        model.train()
        
        # Optimizer setup
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # Loss setup
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

        # Training Loop
        print(f"Starting training for {EPOCHS} epoch(s)...", flush=True)
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            print(f"--- Epoch {epoch+1}/{EPOCHS} ---", flush=True)
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                # Move inputs to device and ensure float32
                x = x_batch.to(device, dtype=torch.float32)
                y = y_batch.to(device, dtype=torch.float32)
                
                # Forward pass
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * x.size(0)
            
            epoch_loss = running_loss / train_loader.batch_size
            print(f"Train Loss: {epoch_loss:.4f}", flush=True)

        # Validation Loop
        model.eval()
        print("--- Validation ---", flush=True)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x = x_batch.to(device, dtype=torch.float32)
                y = y_batch.to(device, dtype=torch.float32)
                
                # Forward pass
                out = model(x)
                
                # Calculate probabilities and store
                probs = torch.sigmoid(out).cpu().numpy()
                targets = y.cpu().numpy()
                
                all_preds.append(probs)
                all_targets.append(targets)

        # Concatenate results
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate ROC-AUC
        try:
            auc_score = roc_auc_score(all_targets, all_preds, multi_label=True, average='macro')
            results['roc_auc_macro'] = auc_score
            print(f"Validation ROC-AUC (Macro): {auc_score:.4f}", flush=True)
        except ValueError as e:
            print(f"Could not compute ROC-AUC: {e}", flush=True)
            results['roc_auc_macro'] = np.nan


    except Exception as e:
        print(f"An error occurred during training: {e}", flush=True)
        results['error'] = str(e)

    finally:
        # Save results
        print("Saving results...", flush=True)
        with open("results.json", "w") as f:
            json.dump(results, f)
        print("Done.", flush=True)

```

## Output
- **validation:** passed
- **code_bytes:** 8285
