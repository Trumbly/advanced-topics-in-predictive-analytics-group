# Task exp_005_recovery_02_error_recovery

- **Experiment:** exp_005
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 07:50:49.211253+00:00
- **Completed:** 2026-04-12 07:51:26.906657+00:00

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
  "architecture": "[cnn_gru_hybrid] 3-conv CNN front-end (2-3 blocks) extracting spectral features, followed by a 1-layer GRU(256) processing the temporal sequence, and a final linear projection.",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0005,
    "dropout": 0.3
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

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# EPOCHS is read from an env var. Default of 1 keeps fast iteration working.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Model Definition ---
class CnnGruHybrid(nn.Module):
    """
    Architecture: 3-conv CNN front-end (spectral features) -> GRU (temporal) -> Linear projection.
    Input shape: (B, 1, N_mels=128, T=313)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- CNN Front-end (Spectral Feature Extraction) ---
        # Input: (B, 1, 128, 313)
        
        # Block 1: 1 -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Block 3: 128 -> 256 channels (C3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # --- Temporal Preparation ---
        # Pool across the Mel dimension (dim=2, index 2) to collapse 128 to 1.
        # Shape: (B, 256, 128, T) -> (B, 256, 1, T)
        self.pool_mel = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- GRU Layer (Temporal Processing) ---
        # Input size must match the channel count from the last CNN block (256).
        # Hidden size = 256.
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=False)
        
        # --- Final Projection ---
        # After GRU, output is (B, T, 256). Pool across time (dim=1) to get (B, 256).
        self.pool_time = nn.AdaptiveAvgPool1d(1)
        # Final linear layer uses LazyLinear to handle variable feature sizes.
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, T)
        
        # 1. CNN Feature Extraction
        x = self.relu1(self.conv1(x))    # (B, 64, 128, T)
        x = self.relu2(self.conv2(x))    # (B, 128, 128, T)
        x = self.relu3(self.conv3(x))    # (B, 256, 128, T)
        
        # 2. Temporal Preparation (Pool Mel dim)
        x = self.pool_mel(x)            # (B, 256, 1, T)
        
        # Permute from (B, C, 1, T) -> (B, 1, C, T) -> (B, T, C) for GRU
        # We want the feature dimension (256) to be the input_size for GRU.
        # PyTorch default: (B, C, H, W). We want (B, T, C).
        # After pool_mel: (B, 256, 1, T)
        x = x.squeeze(2) # (B, 256, T)
        x = x.permute(0, 2, 1) # (B, T, 256) - Correct input for batch_first=True/False GRU setup
        
        # 3. GRU Processing
        # GRU expects (seq_len, batch, input_size) if batch_first=False (default)
        # Our current shape is (B, T, 256). We need (T, B, 256).
        gru_input = x.permute(1, 0, 2) # (T, B, 256)
        gru_output = self.gru(gru_input) # (T, B, 256)
        
        # 4. Aggregation and Classification
        # (T, B, 256) -> (B, 256)
        # We use AdaptiveAvgPool1d over the sequence length dimension (T).
        # Needs to be (B, C, L) for AvgPool1d.
        gru_output = gru_output.permute(1, 2, 0) # (B, 256, T)
        pooled = self.pool_time(gru_output)     # (B, 256, 1)
        
        # Flatten (B, 256, 1) -> (B, 256)
        x = pooled.flatten(1)
        
        # Final linear projection
        logits = self.classifier(x) # (B, num_classes)
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
        # load_precomputed_dataset reads batch_size/num_workers etc. from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
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
                    # Calculate probabilities (sigmoid) and move to CPU/numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for i in range(num_classes):
                if np.any(targs[:, i] == 1):
                    try:
                        auc = roc_auc_score(targs[:, i], probs[:, i])
                        aucs.append(auc)
                    except ValueError:
                        # Handle case where only one class is present in the validation set
                        pass
            
            macro_auc = np.mean(aucs) if aucs else 0.0
            curves["roc_auc_macro"].append(macro_auc)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}", flush=True)
        results["error"] = str(e)
    finally:
        # Ensure results are saved even if an error occurred
        results["final_auc"] = np.mean(curves["roc_auc_macro"]) if curves["roc_auc_macro"] else 0.0
        results["epochs_completed"] = EPOCHS
        
        print("\nSaving results...", flush=True)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Done.", flush=True)

```

## Error that crashed it
Type: UnknownError

Message:
NameError: name 'curves' is not defined

Traceback (tail):
```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_093854_ui_study_start_test/exp_005/code.py", line 198, in <module>
    results["final_auc"] = np.mean(curves["roc_auc_macro"]) if curves["roc_auc_macro"] else 0.0
                                                               ^^^^^^
NameError: name 'curves' is not defined
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
# EPOCHS is read from an env var. Default of 1 keeps fast iteration working.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Model Definition ---
class CnnGruHybrid(nn.Module):
    """
    Architecture: 3-conv CNN front-end (spectral features) -> GRU (temporal) -> Linear projection.
    Input shape: (B, 1, N_mels=128, T=313)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- CNN Front-end (Spectral Feature Extraction) ---
        # Input: (B, 1, 128, 313)
        
        # Block 1: 1 -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Block 3: 128 -> 256 channels (C3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # --- Temporal Preparation ---
        # Pool across the Mel dimension (dim=2, index 2) to collapse 128 to 1.
        # Shape: (B, 256, 128, T) -> (B, 256, 1, T)
        self.pool_mel = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- GRU Layer (Temporal Processing) ---
        # Input size must match the channel count from the last CNN block (256).
        # Hidden size = 256.
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=False)
        
        # --- Final Projection ---
        # After GRU, output is (B, T, 256). Pool across time (dim=1) to get (B, 256).
        self.pool_time = nn.AdaptiveAvgPool1d(1)
        # Final linear layer uses LazyLinear to handle variable feature sizes.
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, T)
        
        # 1. CNN Feature Extraction
        x = self.relu1(self.conv1(x))    # (B, 64, 128, T)
        x = self.relu2(self.conv2(x))    # (B, 128, 128, T)
        x = self.relu3(self.conv3(x))    # (B, 256, 128, T)
        
        # 2. Temporal Preparation (Pool Mel dim)
        x = self.pool_mel(x)            # (B, 256, 1, T)
        
        # Permute from (B, C, 1, T) -> (B, 256, T)
        x = x.squeeze(2) # (B, 256, T)
        
        # 3. GRU Processing
        # GRU expects (seq_len, batch, input_size) if batch_first=False (default)
        # Our current shape is (B, 256, T). We need (T, B, 256).
        # Permute to (B, T, 256) -> (T, B, 256)
        gru_input = x.permute(2, 0, 1) # (T, B, 256)
        gru_output = self.gru(gru_input) # (T, B, 256)
        
        # 4. Aggregation and Classification
        # (T, B, 256) -> (B, 256)
        # Needs to be (B, C, L) for AvgPool1d.
        gru_output = gru_output.permute(1, 2, 0) # (B, 256, T)
        pooled = self.pool_time(gru_output)     # (B, 256, 1)
        
        # Flatten (B, 256, 1) -> (B, 256)
        x = pooled.flatten(1)
        
        # Final linear projection
        logits = self.classifier(x) # (B, num_classes)
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
    curves = {"loss": [], "roc_auc_macro": []} # FIX: Initialize 'curves' dictionary here
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size/num_workers etc. from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

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
                    # Calculate probabilities (sigmoid) and move to CPU/numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for i in range(num_classes):
                if np.any(targs[:, i] == 1):
                    try:
                        auc = roc_auc_score(targs[:, i], probs[:, i])
                        aucs.append(auc)
                    except ValueError:
                        # Handle case where only one class is present in the validation set
                        pass
            
            macro_auc = np.mean(aucs) if aucs else 0.0
            curves["roc_auc_macro"].append(macro_auc)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}", flush=True)
        results["error"] = str(e)
    finally:
        # Ensure results are saved even if an error occurred
        results["final_auc"] = np.mean(curves["roc_auc_macro"]) if curves["roc_auc_macro"] else 0.0
        results["epochs_completed"] = EPOCHS
        
        print("\nSaving results...", flush=True)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Done.", flush=True)
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
# EPOCHS is read from an env var. Default of 1 keeps fast iteration working.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# --- Model Definition ---
class CnnGruHybrid(nn.Module):
    """
    Architecture: 3-conv CNN front-end (spectral features) -> GRU (temporal) -> Linear projection.
    Input shape: (B, 1, N_mels=128, T=313)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- CNN Front-end (Spectral Feature Extraction) ---
        # Input: (B, 1, 128, 313)
        
        # Block 1: 1 -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Block 2: 64 -> 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Block 3: 128 -> 256 channels (C3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # --- Temporal Preparation ---
        # Pool across the Mel dimension (dim=2, index 2) to collapse 128 to 1.
        # Shape: (B, 256, 128, T) -> (B, 256, 1, T)
        self.pool_mel = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- GRU Layer (Temporal Processing) ---
        # Input size must match the channel count from the last CNN block (256).
        # Hidden size = 256.
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=False)
        
        # --- Final Projection ---
        # After GRU, output is (B, T, 256). Pool across time (dim=1) to get (B, 256).
        self.pool_time = nn.AdaptiveAvgPool1d(1)
        # Final linear layer uses LazyLinear to handle variable feature sizes.
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, 1, 128, T)
        
        # 1. CNN Feature Extraction
        x = self.relu1(self.conv1(x))    # (B, 64, 128, T)
        x = self.relu2(self.conv2(x))    # (B, 128, 128, T)
        x = self.relu3(self.conv3(x))    # (B, 256, 128, T)
        
        # 2. Temporal Preparation (Pool Mel dim)
        x = self.pool_mel(x)            # (B, 256, 1, T)
        
        # Permute from (B, C, 1, T) -> (B, 256, T)
        x = x.squeeze(2) # (B, 256, T)
        
        # 3. GRU Processing
        # GRU expects (seq_len, batch, input_size) if batch_first=False (default)
        # Our current shape is (B, 256, T). We need (T, B, 256).
        # Permute to (B, T, 256) -> (T, B, 256)
        gru_input = x.permute(2, 0, 1) # (T, B, 256)
        gru_output = self.gru(gru_input) # (T, B, 256)
        
        # 4. Aggregation and Classification
        # (T, B, 256) -> (B, 256)
        # Needs to be (B, C, L) for AvgPool1d.
        gru_output = gru_output.permute(1, 2, 0) # (B, 256, T)
        pooled = self.pool_time(gru_output)     # (B, 256, 1)
        
        # Flatten (B, 256, 1) -> (B, 256)
        x = pooled.flatten(1)
        
        # Final linear projection
        logits = self.classifier(x) # (B, num_classes)
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
    curves = {"loss": [], "roc_auc_macro": []} # FIX: Initialize 'curves' dictionary here
    try:
        print("loading data...", flush=True)
        # load_precomputed_dataset reads batch_size/num_workers etc. from env vars
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybrid(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

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
                    # Calculate probabilities (sigmoid) and move to CPU/numpy
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for i in range(num_classes):
                if np.any(targs[:, i] == 1):
                    try:
                        auc = roc_auc_score(targs[:, i], probs[:, i])
                        aucs.append(auc)
                    except ValueError:
                        # Handle case where only one class is present in the validation set
                        pass
            
            macro_auc = np.mean(aucs) if aucs else 0.0
            curves["roc_auc_macro"].append(macro_auc)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}", flush=True)
        results["error"] = str(e)
    finally:
        # Ensure results are saved even if an error occurred
        results["final_auc"] = np.mean(curves["roc_auc_macro"]) if curves["roc_auc_macro"] else 0.0
        results["epochs_completed"] = EPOCHS
        
        print("\nSaving results...", flush=True)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("Done.", flush=True)

- **inner_retries:** 0
