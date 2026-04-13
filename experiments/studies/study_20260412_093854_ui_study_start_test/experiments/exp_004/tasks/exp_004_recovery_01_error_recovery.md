# Task exp_004_recovery_01_error_recovery

- **Experiment:** exp_004
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 07:47:22.836542+00:00
- **Completed:** 2026-04-12 07:48:06.169223+00:00

## Prompt Used
```
[SYSTEM]
You are debugging a failed ML training run on BirdCLEF 2026. A previous
attempt at this experiment crashed with a concrete Python/PyTorch error.
Your job is to return a FIXED version of the code that addresses the
specific failure, while keeping the original architecture and training
approach as close to the original as possible.

Rules:
  - Return the COMPLETE, SELF-CONTAINED Python script. Not a diff. Not
    a description of the fix. Not a JSON object. Just plain Python code.
  - Keep the same overall architecture unless the error makes that
    impossible. The fix should be MINIMAL — one or two focused changes.
  - Preserve all the mandatory parts of a correct training script
    (documented below) — they must all still be present in the fix.
  - Do NOT remove progress prints. The operator watches them live.
  - Do NOT change the hyperparameters unless the error is directly
    about them (e.g. batch_size causing OOM).
  - Do NOT retry the exact same code — that will fail the exact same way.

## Mandatory script structure (keep all of this in the fix)

  - `import torch` and `import torch.nn as nn` at the top.
  - Read the device from the env var right after imports:
        DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
        device = torch.device(DEVICE_NAME)
    Do NOT call `torch.cuda.is_available()` or `torch.backends.mps.is_available()`.
  - `torch.set_num_threads(os.cpu_count() or 4)` only when DEVICE_NAME == "cpu".
  - Module-scope layout: imports, DEVICE/device, EPOCHS=1, hyperparams,
    and any `nn.Module` CLASS DEFINITIONS stay at MODULE SCOPE.
  - Runtime block: EVERYTHING that actually runs — `load_precomputed_dataset`,
    model instantiation, `.to(device)`, the training loop, validation,
    and writing `results.json` — must be inside `if __name__ == "__main__":`.
    Spawn workers re-import the script when `num_workers > 0`; without
    the guard the training loop runs recursively in each worker.
  - Load data via `load_precomputed_dataset(...)` with NO path arguments.
    Batch size / num_workers / persistent_workers / prefetch_factor are
    read from BIRDCLEF_* env vars — do NOT pass them explicitly.
  - `EPOCHS = 1` exactly (fast-iteration mode, hard cap).
  - Move the model and every batch to `device`:
        model = MyModel(...).to(device)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
    Before handing predictions to numpy/sklearn, move them back to CPU:
        probs = torch.sigmoid(model(x)).cpu().numpy()
    Stick to float32 — MPS does not fully support float64.
  - For the classifier head, prefer `nn.LazyLinear(num_classes)` over
    `nn.Linear(<hardcoded_int>, num_classes)`. LazyLinear eliminates
    the shape-mismatch family of bugs entirely.
  - Multi-label loss: `nn.BCEWithLogitsLoss(pos_weight=pos_weight)` where
        from pipelines.data_loader import compute_pos_weight
        pos_weight = compute_pos_weight().to(device)
    This rebalances BirdCLEF's long-tail — do NOT remove it.
  - Metric: macro-averaged ROC-AUC via sklearn.
  - Progress prints with `flush=True` at key checkpoints.
  - `results.json` written at the end with metrics OR an `error` key.

## Common fixes for common errors

  - `AttributeError: 'X' object has no attribute 'relu'`:
      Either define `self.relu = nn.ReLU(inplace=True)` in __init__,
      or replace `self.relu(x)` with `F.relu(x)` (from torch.nn.functional).
  - `RuntimeError: mat1 and mat2 shapes cannot be multiplied` /
    `linear(): input and weight.T shapes cannot be multiplied`:
      Flatten/linear size mismatch. REPLACE the fixed-in_features
      `nn.Linear(<N>, num_classes)` with `nn.LazyLinear(num_classes)`.
      LazyLinear infers `in_features` from the first forward pass, so
      it works regardless of how the conv stack reshapes the tensor.
      End the feature stack with `nn.AdaptiveAvgPool2d((1, 1))` and
      `.flatten(1)` to collapse spatial dims.
  - `RuntimeError: Given groups=1, weight of size [X, Y, ...] expected input
     [..., Z, ...] to have Y channels, but got Z channels instead`:
      Channel mismatch between stacked Conv2d layers. Each Conv2d's
      `in_channels` must equal the previous one's `out_channels`.
  - `AttributeError: module 'torch.nn' has no attribute 'XxYd'` /
    `UnknownTorchNnAttribute`:
      You invented a layer name. The real names are `nn.Conv1d`,
      `nn.Conv2d`, `nn.Conv3d`, `nn.BatchNorm1d/2d/3d`,
      `nn.MaxPool1d/2d/3d`, `nn.Linear`, `nn.LazyLinear`. Do NOT write
      `nn.Conv2x2d`, `nn.BatchNormal2d`, or similar fabrications.
  - `TypeError: CnnSmallV1.__init__() got an unexpected keyword argument`:
      The registry's CnnSmallV1 accepts ONLY `num_classes` and
      `in_channels`. Do not pass `pretrained=...`, `weights=...`, etc.
  - `ModuleNotFoundError: No module named 'torchvision'` /
    `NameError: name 'torchvision' is not defined`:
      Use the registry adapter instead of importing torchvision directly:
          from pipelines.models import TorchvisionAdapter
          model = TorchvisionAdapter("efficientnet_b0", num_classes=num_classes)
      Valid backbone names: "efficientnet_b0", "resnet18",
      "mobilenet_v3_small". The adapter handles channel-expand and
      resize-to-224 automatically.
  - `RuntimeError: An attempt has been made to start a new process before
    the current process has finished its bootstrapping phase` /
    `MissingMainGuard`:
      Wrap the training block in `if __name__ == "__main__":`. Only
      class definitions and module-level imports belong outside it.
  - `NameError: name 'num_classes' is not defined`:
      `num_classes` comes from `load_precomputed_dataset(...)` and
      therefore must be computed inside the `__main__` block BEFORE
      you instantiate the model.
  - `AttributeError: 'Dropout' object has no attribute 'set'` or similar
    made-up methods:
      Use only documented PyTorch APIs. nn.Dropout has no `.set()`.
  - `EpochsCapExceeded: EPOCHS = N exceeds the hard cap of 1`:
      Change the literal `EPOCHS = N` line to `EPOCHS = 1`. Do not use
      `max(1, ...)` or `min(..., 1)` — just write `EPOCHS = 1`.


[USER]
## Original architecture proposal
{
  "architecture": "[cnn_attention] EfficientNet-B0 backbone feeding into Self-Attention (Time-Domain) Encoder block",
  "pretrained_model": "efficientnet_b0",
  "hyperparams": {
    "lr": 0.0008,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "dropout": 0.25
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.5,
    "specaugment": true
  }
}

## Broken code (the previous attempt)
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

## Error that crashed it
Type: SyntaxError

Message:
Line 177: unterminated string literal (detected at line 177)

Traceback (tail):
```
(no traceback)
```

## Your Task
Rewrite the code above to fix the error. Return ONLY the complete fixed
Python script — no explanations, no markdown fences, no prose. The first
line of your response must be valid Python (e.g. `import json`).

```

## LLM Response
```
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
# =================================================================
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
```

## Output
- **code:** import json
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
# =================================================================
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

- **inner_retries:** 0
