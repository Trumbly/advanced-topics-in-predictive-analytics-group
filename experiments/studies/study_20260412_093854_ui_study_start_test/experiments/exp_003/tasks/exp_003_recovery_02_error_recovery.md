# Task exp_003_recovery_02_error_recovery

- **Experiment:** exp_003
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 07:45:05.737006+00:00
- **Completed:** 2026-04-12 07:46:04.096111+00:00

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
  "architecture": "[mobilenet_v3_small] MobileNetV3Small via TorchvisionAdapter, standard classification head",
  "pretrained_model": "mobilenet_v3_small",
  "hyperparams": {
    "lr": 0.0005,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0005,
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
            # We fake a sequence of convs that output a feature map.
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
        
        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)  
        
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
                    # Get probabilities and move back to CPU/Numpy for sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                # Skip columns with no positives
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
Type: ScriptReportedError

Message:
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 32, 1, 1])

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
- **code:** import json
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

- **inner_retries:** 0
