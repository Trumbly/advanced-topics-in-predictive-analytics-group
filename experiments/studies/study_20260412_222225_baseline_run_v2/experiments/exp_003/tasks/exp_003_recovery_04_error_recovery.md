# Task exp_003_recovery_04_error_recovery

- **Experiment:** exp_003
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 22:51:43.040440+00:00
- **Completed:** 2026-04-12 22:53:10.681345+00:00

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
  "architecture": "[cnn_attention] 3-stage approach: 1. Small 3-conv CNN front-end (extract local features) -> 2. Self-Attention Pooling Block (aggregate global context) -> 3. Linear head for multi-label output",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.2
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.3,
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class AttentionBlock(nn.Module):
    """
    Implements a simplified self-attention pooling block.
    Takes features (B, C, H, W), pools globally, and then applies
    a small attention mechanism (e.g., a simple multiplicative gate)
    before the final linear layer.
    """
    def __init__(self, channel_dim):
        super().__init__()
        self.channel_dim = channel_dim
        # Global pooling collapses (H, W) -> (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple attention mechanism: 1x1 convolution on the pooled feature
        # to generate weights, followed by a sigmoid activation.
        # We use a single output channel for the attention weight vector.
        self.attention_conv = nn.Conv2d(channel_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Pooling: (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Attention Weight Generation: (B, 1, 1, 1)
        attn_weights_raw = self.attention_conv(pooled)
        attn_weights = self.sigmoid(attn_weights_raw)
        
        # 3. Apply weights: (B, C, 1, 1) * (B, 1, 1, 1) -> (B, C, 1, 1)
        # We broadcast the weight across all C channels.
        weighted_features = pooled * attn_weights
        
        # The output is now a feature vector (B, C, 1, 1) ready for the linear head
        return weighted_features


class CnnAttentionModel(nn.Module):
    """
    3-stage approach: 
    1. Small 3-conv CNN front-end (extract local features) 
    2. Self-Attention Pooling Block (aggregate global context) 
    3. Linear head for multi-label output
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. Small 3-conv CNN front-end ---
        # Input channels = 1 (single spectrogram channel)
        # Padding=1 and kernel=3 preserves spatial dimensions H and W.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # out: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # out: 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # out: 128
        
        # --- 2. Self-Attention Pooling Block ---
        # The feature dimension exiting the CNN stack is 128 channels.
        self.attention_block = AttentionBlock(channel_dim=128)
        
        # --- 3. Linear Head ---
        # Use LazyLinear to handle input feature size inferred from the 
        # feature map after the attention block and pooling.
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        
        # Stage 1: CNN Feature Extraction
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) # Output shape: (B, 128, H_out, W_out)

        # Stage 2: Attention Pooling
        x = self.attention_block(x) # Output shape: (B, 128, 1, 1)

        # Squeeze the spatial dimensions (1, 1) and flatten to (B, 128)
        x = x.view(x.size(0), -1) # Flatten (B, C, 1, 1) -> (B, C)

        # Stage 3: Linear Head
        logits = self.head(x) # (B, num_classes)
        return logits


# ==============================================================================
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
        # load_precomputed_dataset populates num_classes correctly.
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnAttentionModel(num_classes=num_classes)
        
        # FIX: Initialize LazyLinear parameters before calling sum(p.numel())
        # by running a dummy forward pass.
        dummy_input = torch.randn(1, 1, 224, 224).to(device, dtype=torch.float32)
        with torch.no_grad():
            _ = model(dummy_input)
            
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log progress every 10% or on the last batch
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
                    # Compute probabilities and move results back to CPU/Numpy
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
                    # Using average_precision_score as a proxy for AP@k mean
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )

        final_auc = curves["roc_auc_macro"][-1]
        final_cmap5 = curves["cmap_at_5"][-1]
        final_f1 = curves["f1_macro"][-1]
        final_loss = curves["loss"][-1]

        results = {
            "metrics": {
                "roc_auc_macro": final_auc,
                "cmap_at_5": final_cmap5,
                "f1_macro": final_f1,
                "loss": final_loss,
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_auc:.4f} "
            f"cmap@5={final_cmap5:.4f} "
            f"f1={final_f1:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Error that crashed it
Type: ScriptReportedError

Message:
RuntimeError: Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same

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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from an env var.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class AttentionBlock(nn.Module):
    """
    Implements a simplified self-attention pooling block.
    Takes features (B, C, H, W), pools globally, and then applies
    a small attention mechanism (e.g., a simple multiplicative gate)
    before the final linear layer.
    """
    def __init__(self, channel_dim):
        super().__init__()
        self.channel_dim = channel_dim
        # Global pooling collapses (H, W) -> (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple attention mechanism: 1x1 convolution on the pooled feature
        # to generate weights, followed by a sigmoid activation.
        # We use a single output channel for the attention weight vector.
        self.attention_conv = nn.Conv2d(channel_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Pooling: (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Attention Weight Generation: (B, 1, 1, 1)
        attn_weights_raw = self.attention_conv(pooled)
        attn_weights = self.sigmoid(attn_weights_raw)
        
        # 3. Apply weights: (B, C, 1, 1) * (B, 1, 1, 1) -> (B, C, 1, 1)
        # We broadcast the weight across all C channels.
        weighted_features = pooled * attn_weights
        
        # The output is now a feature vector (B, C, 1, 1) ready for the linear head
        return weighted_features


class CnnAttentionModel(nn.Module):
    """
    3-stage approach: 
    1. Small 3-conv CNN front-end (extract local features) 
    2. Self-Attention Pooling Block (aggregate global context) 
    3. Linear head for multi-label output
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. Small 3-conv CNN front-end ---
        # Input channels = 1 (single spectrogram channel)
        # Padding=1 and kernel=3 preserves spatial dimensions H and W.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # out: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # out: 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # out: 128
        
        # --- 2. Self-Attention Pooling Block ---
        # The feature dimension exiting the CNN stack is 128 channels.
        self.attention_block = AttentionBlock(channel_dim=128)
        
        # --- 3. Linear Head ---
        # Use LazyLinear to handle input feature size inferred from the 
        # feature map after the attention block and pooling.
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        
        # Stage 1: CNN Feature Extraction
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) # Output shape: (B, 128, H_out, W_out)

        # Stage 2: Attention Pooling
        x = self.attention_block(x) # Output shape: (B, 128, 1, 1)

        # Squeeze the spatial dimensions (1, 1) and flatten to (B, 128)
        x = x.view(x.size(0), -1) # Flatten (B, C, 1, 1) -> (B, C)

        # Stage 3: Linear Head
        logits = self.head(x) # (B, num_classes)
        return logits


# ==============================================================================
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
        # load_precomputed_dataset populates num_classes correctly.
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnAttentionModel(num_classes=num_classes)
        
        # FIX: Initialize LazyLinear parameters before calling sum(p.numel())
        # by running a dummy forward pass.
        dummy_input = torch.randn(1, 1, 224, 224).to(device, dtype=torch.float32)
        with torch.no_grad():
            _ = model(dummy_input)
            
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log progress every 10% or on the last batch
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
                    # Compute probabilities and move results back to CPU/Numpy
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
                    # Using average_precision_score as a proxy for AP@k mean
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )

        final_auc = curves["roc_auc_macro"][-1]
        final_cmap5 = curves["cmap_at_5"][-1]
        final_f1 = curves["f1_macro"][-1]
        final_loss = curves["loss"][-1]

        results = {
            "metrics": {
                "roc_auc_macro": final_auc,
                "cmap_at_5": final_cmap5,
                "f1_macro": final_f1,
                "loss": final_loss,
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_auc:.4f} "
            f"cmap@5={final_cmap5:.4f} "
            f"f1={final_f1:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)
```

## Output
- **code:** import json
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
# EPOCHS is read from an env var.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===

class AttentionBlock(nn.Module):
    """
    Implements a simplified self-attention pooling block.
    Takes features (B, C, H, W), pools globally, and then applies
    a small attention mechanism (e.g., a simple multiplicative gate)
    before the final linear layer.
    """
    def __init__(self, channel_dim):
        super().__init__()
        self.channel_dim = channel_dim
        # Global pooling collapses (H, W) -> (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple attention mechanism: 1x1 convolution on the pooled feature
        # to generate weights, followed by a sigmoid activation.
        # We use a single output channel for the attention weight vector.
        self.attention_conv = nn.Conv2d(channel_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Pooling: (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Attention Weight Generation: (B, 1, 1, 1)
        attn_weights_raw = self.attention_conv(pooled)
        attn_weights = self.sigmoid(attn_weights_raw)
        
        # 3. Apply weights: (B, C, 1, 1) * (B, 1, 1, 1) -> (B, C, 1, 1)
        # We broadcast the weight across all C channels.
        weighted_features = pooled * attn_weights
        
        # The output is now a feature vector (B, C, 1, 1) ready for the linear head
        return weighted_features


class CnnAttentionModel(nn.Module):
    """
    3-stage approach: 
    1. Small 3-conv CNN front-end (extract local features) 
    2. Self-Attention Pooling Block (aggregate global context) 
    3. Linear head for multi-label output
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. Small 3-conv CNN front-end ---
        # Input channels = 1 (single spectrogram channel)
        # Padding=1 and kernel=3 preserves spatial dimensions H and W.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # out: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # out: 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # out: 128
        
        # --- 2. Self-Attention Pooling Block ---
        # The feature dimension exiting the CNN stack is 128 channels.
        self.attention_block = AttentionBlock(channel_dim=128)
        
        # --- 3. Linear Head ---
        # Use LazyLinear to handle input feature size inferred from the 
        # feature map after the attention block and pooling.
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, H, W)
        
        # Stage 1: CNN Feature Extraction
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x) # Output shape: (B, 128, H_out, W_out)

        # Stage 2: Attention Pooling
        x = self.attention_block(x) # Output shape: (B, 128, 1, 1)

        # Squeeze the spatial dimensions (1, 1) and flatten to (B, 128)
        x = x.view(x.size(0), -1) # Flatten (B, C, 1, 1) -> (B, C)

        # Stage 3: Linear Head
        logits = self.head(x) # (B, num_classes)
        return logits


# ==============================================================================
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
        # load_precomputed_dataset populates num_classes correctly.
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnAttentionModel(num_classes=num_classes)
        
        # FIX: Initialize LazyLinear parameters before calling sum(p.numel())
        # by running a dummy forward pass.
        dummy_input = torch.randn(1, 1, 224, 224).to(device, dtype=torch.float32)
        with torch.no_grad():
            _ = model(dummy_input)
            
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        # Log progress every 10% or on the last batch
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
                    # Compute probabilities and move results back to CPU/Numpy
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
                    # Using average_precision_score as a proxy for AP@k mean
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 at threshold 0.5 ---
            preds_binary = (probs >= 0.5).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_binary, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            curves["f1_macro"].append(val_f1)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc={val_auc:.4f} "
                f"cmap@5={val_cmap5:.4f} f1={val_f1:.4f}",
                flush=True,
            )

        final_auc = curves["roc_auc_macro"][-1]
        final_cmap5 = curves["cmap_at_5"][-1]
        final_f1 = curves["f1_macro"][-1]
        final_loss = curves["loss"][-1]

        results = {
            "metrics": {
                "roc_auc_macro": final_auc,
                "cmap_at_5": final_cmap5,
                "f1_macro": final_f1,
                "loss": final_loss,
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_auc:.4f} "
            f"cmap@5={final_cmap5:.4f} "
            f"f1={final_f1:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

- **inner_retries:** 0
