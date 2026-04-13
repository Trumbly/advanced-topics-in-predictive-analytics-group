# Task exp_003_recovery_02_error_recovery

- **Experiment:** exp_003
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 08:43:29.092875+00:00
- **Completed:** 2026-04-12 08:44:26.035926+00:00

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
  "architecture": "[mobilenet_v3_small] MobileNetV3 backbone with Global Average Pooling (GAP) before the final classification head",
  "pretrained_model": "mobilenet_v3_small",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": false,
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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# EPOCHS reads from BIRDCLEF_EPOCHS env var, defaults to "1".
EPOCHS = 1
LR = 1e-3
# The proposal specifies: time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
# The error was NameError: name 'BACKBONE_CLASS' is not defined.
# This suggests that the placeholder class definitions were somehow removed or bypassed
# in the execution environment, or the definition scope was wrong.
# We must ensure all class definitions are self-contained and correctly structured.

try:
    # Attempt to import the specific registry model name
    from pipelines.models import MobileNetV3Small
    BACKBONE_CLASS = MobileNetV3Small
except ImportError:
    # Fallback/Placeholder if the exact registry import fails in the execution environment
    # This ensures the script structure is complete even if the exact import path changes.
    print("Warning: Could not import MobileNetV3Small from pipelines.models. Using placeholder structure.", flush=True)
    class MobileNetV3Small(nn.Module):
        """Placeholder for the actual MobileNetV3 backbone."""
        def __init__(self, num_classes):
            super().__init__()
            # Simulate a feature extractor block (e.g., a stack of convolutions)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
            # The actual backbone usually outputs a feature map C x H x W
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            # The final head must use LazyLinear
            self.classifier_head = nn.LazyLinear(num_classes)

        def forward(self, x):
            # In a real scenario, the backbone would be called here.
            # We simulate the sequence: Conv -> GAP -> Linear
            x = self.features(x)
            x = self.gap(x)
            x = x.flatten(2) # Flatten features (B, C, 1, 1) -> (B, C)
            return self.classifier_head(x)

class MobileNetV3Wrapper(nn.Module):
    """Wraps the backbone to enforce Global Average Pooling (GAP) before the classifier head."""
    def __init__(self, num_classes):
        super().__init__()
        # Initialize the backbone using the actual registry class
        self.backbone = BACKBONE_CLASS(num_classes=num_classes)
        
        # NOTE: Since the registry model structure is unknown, we rely on the
        # wrapper to enforce the GAP logic if the backbone doesn't already.
        # For safety, we explicitly ensure the final layer is LazyLinear.
        # We assume the backbone handles the initial feature extraction.
        
    def forward(self, x):
        # 1. Pass through the backbone (which might already include pooling/flattening)
        x = self.backbone(x)
        
        # 2. Enforce GAP if the backbone output isn't already suitable for the head
        # If the backbone already ends with GAP/Flattening, this might be redundant,
        # but it strictly adheres to the proposal step.
        if x.shape[-2:] != (1, 1):
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 3. Pass through the final classification head (which is LazyLinear)
        return x


# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads required configs (batch_size, num_workers, etc.)
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
        model = MobileNetV3Wrapper(num_classes=num_classes)
        # IMPORTANT: Move to device immediately.
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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

            # Validation
            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities and move results back to CPU
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC calculation
            aucs = []
            for c in range(targs.shape[1]):
                # Check for positive labels to avoid sklearn error
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        # Should not happen if sum > 0, but safe guard
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    finally:
        # Final logging and results writing (runs even if training fails)
        if 'results' not in locals() or "error" not in results:
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

        with open("results.json", "w") as fh:
            json.dump(results, fh)

```

## Error that crashed it
Type: ScriptReportedError

Message:
NameError: name 'BACKBONE_CLASS' is not defined

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
# EPOCHS reads from BIRDCLEF_EPOCHS env var, defaults to "1".
EPOCHS = 1
LR = 1e-3
# The proposal specifies: time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
# Initialize BACKBONE_CLASS to ensure it exists before class definitions read it.
BACKBONE_CLASS = None

try:
    # Attempt to import the specific registry model name
    from pipelines.models import MobileNetV3Small
    BACKBONE_CLASS = MobileNetV3Small
except ImportError:
    # Fallback/Placeholder if the exact registry import fails in the execution environment
    print("Warning: Could not import MobileNetV3Small from pipelines.models. Using placeholder structure.", flush=True)
    
    # Define the placeholder class structure locally
    class PlaceholderMobileNetV3Small(nn.Module):
        """Placeholder for the actual MobileNetV3 backbone."""
        def __init__(self, num_classes):
            super().__init__()
            # Simulate a feature extractor block (e.g., a stack of convolutions)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
            # The actual backbone usually outputs a feature map C x H x W
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            # The final head must use LazyLinear
            self.classifier_head = nn.LazyLinear(num_classes)

        def forward(self, x):
            # Simulate the sequence: Conv -> GAP -> Linear
            x = self.features(x)
            x = self.gap(x)
            x = x.flatten(2) # Flatten features (B, C, 1, 1) -> (B, C)
            return self.classifier_head(x)
            
    # Assign the placeholder class to the globally scoped variable
    BACKBONE_CLASS = PlaceholderMobileNetV3Small


class MobileNetV3Wrapper(nn.Module):
    """Wraps the backbone to enforce Global Average Pooling (GAP) before the classifier head."""
    def __init__(self, num_classes):
        super().__init__()
        # Initialize the backbone using the actual registry class
        self.backbone = BACKBONE_CLASS(num_classes=num_classes)
        
    def forward(self, x):
        # 1. Pass through the backbone (which might already include pooling/flattening)
        x = self.backbone(x)
        
        # 2. Enforce GAP if the backbone output isn't already suitable for the head
        if x.shape[-2:] != (1, 1):
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 3. Pass through the final classification head (which is LazyLinear)
        return x


# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads required configs (batch_size, num_workers, etc.)
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
        model = MobileNetV3Wrapper(num_classes=num_classes)
        # IMPORTANT: Move to device immediately.
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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

            # Validation
            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities and move results back to CPU
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC calculation
            aucs = []
            for c in range(targs.shape[1]):
                # Check for positive labels to avoid sklearn error
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        # Should not happen if sum > 0, but safe guard
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    finally:
        # Final logging and results writing (runs even if training fails)
        if 'results' not in locals() or "error" not in results:
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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# EPOCHS reads from BIRDCLEF_EPOCHS env var, defaults to "1".
EPOCHS = 1
LR = 1e-3
# The proposal specifies: time_shift: false, noise_injection: false, mixup: 0.0, specaugment: true
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
# Initialize BACKBONE_CLASS to ensure it exists before class definitions read it.
BACKBONE_CLASS = None

try:
    # Attempt to import the specific registry model name
    from pipelines.models import MobileNetV3Small
    BACKBONE_CLASS = MobileNetV3Small
except ImportError:
    # Fallback/Placeholder if the exact registry import fails in the execution environment
    print("Warning: Could not import MobileNetV3Small from pipelines.models. Using placeholder structure.", flush=True)
    
    # Define the placeholder class structure locally
    class PlaceholderMobileNetV3Small(nn.Module):
        """Placeholder for the actual MobileNetV3 backbone."""
        def __init__(self, num_classes):
            super().__init__()
            # Simulate a feature extractor block (e.g., a stack of convolutions)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
            # The actual backbone usually outputs a feature map C x H x W
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            # The final head must use LazyLinear
            self.classifier_head = nn.LazyLinear(num_classes)

        def forward(self, x):
            # Simulate the sequence: Conv -> GAP -> Linear
            x = self.features(x)
            x = self.gap(x)
            x = x.flatten(2) # Flatten features (B, C, 1, 1) -> (B, C)
            return self.classifier_head(x)
            
    # Assign the placeholder class to the globally scoped variable
    BACKBONE_CLASS = PlaceholderMobileNetV3Small


class MobileNetV3Wrapper(nn.Module):
    """Wraps the backbone to enforce Global Average Pooling (GAP) before the classifier head."""
    def __init__(self, num_classes):
        super().__init__()
        # Initialize the backbone using the actual registry class
        self.backbone = BACKBONE_CLASS(num_classes=num_classes)
        
    def forward(self, x):
        # 1. Pass through the backbone (which might already include pooling/flattening)
        x = self.backbone(x)
        
        # 2. Enforce GAP if the backbone output isn't already suitable for the head
        if x.shape[-2:] != (1, 1):
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 3. Pass through the final classification head (which is LazyLinear)
        return x


# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # The data loader reads required configs (batch_size, num_workers, etc.)
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
        model = MobileNetV3Wrapper(num_classes=num_classes)
        # IMPORTANT: Move to device immediately.
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

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

            # Validation
            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities and move results back to CPU
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC calculation
            aucs = []
            for c in range(targs.shape[1]):
                # Check for positive labels to avoid sklearn error
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        # Should not happen if sum > 0, but safe guard
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0
            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["roc_auc_macro"].append(val_auc)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    finally:
        # Final logging and results writing (runs even if training fails)
        if 'results' not in locals() or "error" not in results:
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

        with open("results.json", "w") as fh:
            json.dump(results, fh)

- **inner_retries:** 0
