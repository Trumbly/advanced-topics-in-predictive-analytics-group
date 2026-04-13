# Task exp_005_recovery_01_error_recovery

- **Experiment:** exp_005
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 08:46:39.996448+00:00
- **Completed:** 2026-04-12 08:47:16.220120+00:00

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
  "architecture": "[efficientnet_b0] EfficientNet-B0 backbone followed by a multi-head self-attention pooling layer before the final classification head",
  "pretrained_model": "efficientnet_b0",
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
    "mixup": 0.0,
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
from pipelines.models import EfficientNetB0 # Assuming this import snippet exists for the registry model

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Custom Components ---

class SelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling layer to aggregate spatial information
    and reduce the feature map to channel-wise statistics (B, C, 1, 1) -> (B, C).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple channel-wise attention mechanism (Squeeze-Excitation style concept)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Average Pooling to get (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Apply attention mechanism
        attention_weights = self.fc(pooled) # (B, C, 1, 1)
        
        # 3. Re-scale the original features by the attention weights
        # We multiply element-wise, so we need to unsqueeze the weight across batch/channel if necessary.
        # Since both are (B, C, 1, 1), direct multiplication works.
        output = x * attention_weights
        
        # Final pooling to get the representative feature (B, C, 1, 1)
        final_representation = self.avg_pool(output)
        return final_representation.flatten(2) # Returns (B, C)

# --- Final Model Definition ---

class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: EfficientNet-B0
        # NOTE: Input channels must be 1. We assume the registry model wrapper 
        # handles the necessary adaptation or that the provided snippet 
        # implicitly handles single-channel input for this task.
        self.backbone = EfficientNetB0(pretrained=False)
        
        # 2. Feature Adaptation Layer (To handle the 1-channel input mismatch 
        # if the backbone expects 3 channels, we project 1 -> 3 for the first pass 
        # ONLY IF the backbone's first layer is convolutional and expects 3 channels.
        # Since we cannot reliably inspect the first layer, we rely on the 
        # assumption that the pipeline model wrapper is robust.
        # If the backbone fails on 1 channel, this block must be adjusted 
        # based on the actual structure of EfficientNetB0.
        self.feature_adaptor = nn.Identity() # Placeholder, trust the backbone initialization for now.

        # 3. Global Pooling and Attention
        # The output channels (C) of the backbone must be determined.
        # We rely on the SelfAttentionPooling to handle the feature map size.
        # We need to pass a dummy tensor to determine the output channels C. 
        # Since we cannot do that easily, we must assume the backbone's last 
        # convolutional layer output dimension is the correct 'in_channels' for the pooling.
        # For EfficientNetB0, the final feature map channel count is typically 1280.
        BACKBONE_OUTPUT_CHANNELS = 1280 # Standard output channels for EfficientNetB0
        self.attention_pool = SelfAttentionPooling(BACKBONE_OUTPUT_CHANNELS)

        # 4. Classification Head: Uses LazyLinear to handle variable feature sizes
        self.classifier_head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, H, W)
        
        # 1. Pass through backbone
        # The backbone processes the input tensor x.
        x_backbone = self.backbone(x) # Output shape: (B, C, H_out, W_out)
        
        # 2. Self-Attention Pooling
        # Output shape: (B, C)
        x_pooled = self.attention_pool(x_backbone)
        
        # 3. Classification Head
        # x_pooled is already (B, C), suitable for LazyLinear
        logits = self.classifier_head(x_pooled) # Output shape: (B, num_classes)
        
        return logits

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ============================================================================
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
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = BirdCLEFModel(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        # for the heavy long-tail class imbalance. Capped at 50x.
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
                # Move each batch to the training device. Float32 only —
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
                    # `.cpu()` before `.numpy()` — sklearn and numpy only
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Error that crashed it
Type: ImportError

Message:
ImportError: cannot import name 'EfficientNetB0' from 'pipelines.models' (/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/pipelines/models.py)

Traceback (tail):
```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_103619_ui_study_start_test/exp_005/code.py", line 10, in <module>
    from pipelines.models import EfficientNetB0 # Assuming this import snippet exists for the registry model
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ImportError: cannot import name 'EfficientNetB0' from 'pipelines.models' (/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/pipelines/models.py)
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
from pipelines.models import TorchvisionAdapter # Use the adapter instead of direct module import

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Custom Components ---

class SelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling layer to aggregate spatial information
    and reduce the feature map to channel-wise statistics (B, C, H, W) -> (B, C).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple channel-wise attention mechanism (Squeeze-Excitation style concept)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Average Pooling to get (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Apply attention mechanism
        attention_weights = self.fc(pooled) # (B, C, 1, 1)
        
        # 3. Re-scale the original features by the attention weights
        # Since both are (B, C, 1, 1), direct multiplication works.
        output = x * attention_weights
        
        # Final pooling to get the representative feature (B, C, 1, 1)
        final_representation = self.avg_pool(output)
        return final_representation.flatten(2) # Returns (B, C)

# --- Final Model Definition ---

class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: EfficientNet-B0 via Adapter
        # Use TorchvisionAdapter which handles loading and adaptation.
        self.backbone = TorchvisionAdapter("efficientnet_b0", pretrained=False)
        
        # 2. Feature Adaptation Layer (Placeholder - relies on adapter structure)
        self.feature_adaptor = nn.Identity() 

        # 3. Global Pooling and Attention
        # For EfficientNetB0, the final feature map channel count is typically 1280.
        BACKBONE_OUTPUT_CHANNELS = 1280 
        self.attention_pool = SelfAttentionPooling(BACKBONE_OUTPUT_CHANNELS)

        # 4. Classification Head: Uses LazyLinear to handle variable feature sizes
        self.classifier_head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, H, W)
        
        # 1. Pass through backbone
        # The backbone processes the input tensor x.
        x_backbone = self.backbone(x) # Output shape: (B, C, H_out, W_out)
        
        # 2. Self-Attention Pooling
        # Output shape: (B, C)
        x_pooled = self.attention_pool(x_backbone)
        
        # 3. Classification Head
        # x_pooled is already (B, C), suitable for LazyLinear
        logits = self.classifier_head(x_pooled) # Output shape: (B, num_classes)
        
        return logits

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ============================================================================
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
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = BirdCLEFModel(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        # for the heavy long-tail class imbalance. Capped at 50x.
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
                # Move each batch to the training device. Float32 only —
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
                    # `.cpu()` before `.numpy()` — sklearn and numpy only
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}",
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
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
from pipelines.models import TorchvisionAdapter # Use the adapter instead of direct module import

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Custom Components ---

class SelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling layer to aggregate spatial information
    and reduce the feature map to channel-wise statistics (B, C, H, W) -> (B, C).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Simple channel-wise attention mechanism (Squeeze-Excitation style concept)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Global Average Pooling to get (B, C, 1, 1)
        pooled = self.avg_pool(x)
        
        # 2. Apply attention mechanism
        attention_weights = self.fc(pooled) # (B, C, 1, 1)
        
        # 3. Re-scale the original features by the attention weights
        # Since both are (B, C, 1, 1), direct multiplication works.
        output = x * attention_weights
        
        # Final pooling to get the representative feature (B, C, 1, 1)
        final_representation = self.avg_pool(output)
        return final_representation.flatten(2) # Returns (B, C)

# --- Final Model Definition ---

class BirdCLEFModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: EfficientNet-B0 via Adapter
        # Use TorchvisionAdapter which handles loading and adaptation.
        self.backbone = TorchvisionAdapter("efficientnet_b0", pretrained=False)
        
        # 2. Feature Adaptation Layer (Placeholder - relies on adapter structure)
        self.feature_adaptor = nn.Identity() 

        # 3. Global Pooling and Attention
        # For EfficientNetB0, the final feature map channel count is typically 1280.
        BACKBONE_OUTPUT_CHANNELS = 1280 
        self.attention_pool = SelfAttentionPooling(BACKBONE_OUTPUT_CHANNELS)

        # 4. Classification Head: Uses LazyLinear to handle variable feature sizes
        self.classifier_head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # Input x shape: (B, 1, H, W)
        
        # 1. Pass through backbone
        # The backbone processes the input tensor x.
        x_backbone = self.backbone(x) # Output shape: (B, C, H_out, W_out)
        
        # 2. Self-Attention Pooling
        # Output shape: (B, C)
        x_pooled = self.attention_pool(x_backbone)
        
        # 3. Classification Head
        # x_pooled is already (B, C), suitable for LazyLinear
        logits = self.classifier_head(x_pooled) # Output shape: (B, num_classes)
        
        return logits

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ============================================================================
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
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = BirdCLEFModel(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        # for the heavy long-tail class imbalance. Capped at 50x.
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
                # Move each batch to the training device. Float32 only —
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
                    # `.cpu()` before `.numpy()` — sklearn and numpy only
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

- **inner_retries:** 0
