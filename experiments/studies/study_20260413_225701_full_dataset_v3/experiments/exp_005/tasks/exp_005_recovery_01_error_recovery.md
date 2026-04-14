# Task exp_005_recovery_01_error_recovery

- **Experiment:** exp_005
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-14 04:40:02.425566+00:00
- **Completed:** 2026-04-14 04:41:52.371761+00:00

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
  - Module-scope layout: imports, DEVICE/device, EPOCHS, hyperparams,
    and any `nn.Module` CLASS DEFINITIONS stay at MODULE SCOPE.
  - Runtime block: EVERYTHING that actually runs — `load_precomputed_dataset`,
    model instantiation, `.to(device)`, the training loop, validation,
    and writing `results.json` — must be inside `if __name__ == "__main__":`.
    Spawn workers re-import the script when `num_workers > 0`; without
    the guard the training loop runs recursively in each worker.
  - Load data via `load_precomputed_dataset(...)` with NO path arguments.
    Batch size / num_workers / persistent_workers / prefetch_factor are
    read from BIRDCLEF_* env vars — do NOT pass them explicitly.
  - `EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))` — read from env var.
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

  - `TypeError: 'str' object is not callable` on `device.type()`:
      `device.type` is a PROPERTY (a plain string), not a method.
      Write `device.type == 'cuda'`, NOT `device.type() == 'cuda'`.
      Remove the parentheses.
  - `ValueError: Attempted to use an uninitialized parameter` /
    `UninitializedParameter` / `LazyModule`:
      You are using `nn.LazyLinear` but calling `p.numel()` (e.g. to
      count parameters) BEFORE the first forward pass. LazyLinear's
      weight is uninitialized until data flows through it. Fix:
          model = MyModel(num_classes=num_classes).to(device)
          with torch.no_grad():
              dummy = torch.zeros(1, 1, 128, 313, device=device)
              model(dummy)
          n_params = sum(p.numel() for p in model.parameters())
      The dummy forward pass materializes all lazy parameters.
  - `RuntimeError: Input type (MPSFloatType) and weight type (torch.FloatTensor)`:
      The model (or parts of it) is on CPU while the input is on MPS.
      Make sure the ENTIRE model is on the device:
          model = MyModel(num_classes=num_classes).to(device)
      If using a pretrained backbone, also move it:
          model.backbone = model.backbone.to(device)
      Check that no submodule is created AFTER the `.to(device)` call.
  - `RuntimeError: MPS backend out of memory`:
      The model is too large for the available GPU memory. Reduce
      model size: fewer channels, fewer layers, smaller hidden dims.
      Target under 5M parameters for MPS with batch_size 512.
  - `AttributeError: 'tuple' object has no attribute 'permute'` /
    `'transpose'` / `'squeeze'` / `'contiguous'`:
      `nn.GRU` and `nn.LSTM` return `(output, hidden_state)` — a TUPLE.
      You must unpack: `output, _ = self.gru(x)` then use `output`.
      Do NOT call `.permute()` / `.transpose()` on the tuple directly.
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
  - `EpochsCapExceeded: EPOCHS = N exceeds the hard cap`:
      Replace the literal `EPOCHS = N` with the env-var pattern:
          EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
      Do NOT hardcode a literal like `EPOCHS = 4` — use the env var.
  - `AttributeError: 'Conv2dNormActivation' object has no attribute 'kernel_size'`:
      MobileNetV3's first layer is `Conv2dNormActivation`, not `nn.Conv2d`.
      Do NOT access `.kernel_size` on it. Use the TorchvisionAdapter from
      the registry instead of manually modifying pretrained model layers.
  - `ValueError: Target size (torch.Size([B, C])) must be the same as
    input size (torch.Size([B, X, C]))`:
      Your model outputs 3D tensors `(batch, something, num_classes)` but
      the loss expects 2D `(batch, num_classes)`. You are missing a
      pooling + flatten step. End the feature extractor with
      `nn.AdaptiveAvgPool2d((1, 1))` then `.flatten(1)` before the
      classifier Linear/LazyLinear head.
  - `RuntimeError: expected input to have 3 channels, but got 1` /
    `expected input[...] to have 3 channels, but got 1 channels`:
      Pretrained torchvision models expect 3 channels. Spectrograms
      have 1. Do NOT import torchvision directly. Use:
          from pipelines.models import TorchvisionAdapter
          model = TorchvisionAdapter("mobilenet_v3_small", num_classes=num_classes)
      The adapter handles 1→3 channel expansion automatically.
  - `ValueError: Input dimension should be at least 3`:
      You are squeezing or reshaping a tensor incorrectly before
      passing it to a layer that expects at least 3D input. Check
      your `.squeeze()`, `.view()`, or `.reshape()` calls. The input
      to Conv2d must be 4D: (batch, channels, height, width).
  - `MOCK` / `MockLoader` / `MockDataset` in output:
      NEVER create mock/fake data loaders. The real
      `load_precomputed_dataset` from `pipelines.data_loader` is
      always available. Remove ALL mock code and use the real import.


[USER]
## Original architecture proposal
{
  "architecture": "[cnn_gru_hybrid] 3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128) for temporal sequence modeling",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 7,
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters from the proposal ===
# Note: LR is set in the skeleton block below, but we track the proposal's value: 0.001
LR = 1e-3
# The skeleton uses Adam by default for optimization, which matches the proposal's optimizer.
# We will use the skeleton's Adam initialization.

# Overriding default augmentation to match the proposal
AUGMENTATION = {
    "time_shift": False,
    "noise_injection": False,
    "mixup": 0.0,
    "specaugment": True
}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128)
    for temporal sequence modeling.
    Input: (B, 1, Mel, Time) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Block CNN Feature Extractor ---
        # Input: (B, 1, 128, 313)
        self.cnn_blocks = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Output: (B, 32, 128, 313)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 -> 64 channels (keeping feature depth consistent for GRU)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsample the Mel dimension (128) to 1, keeping Time (313) intact.
        self.mel_pool = nn.AdaptiveAvgPool2d((1, 1)) # Output: (B, 64, 1, 313)
        
        # --- GRU Layer ---
        # Input to GRU: (B, T, F) -> T=313, F=64
        # We need to permute the output of pooling to (B, T, F) before GRU.
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True # Crucial for sequence processing
        )
        
        # --- Final Classification Head ---
        # After GRU, we pool the sequence output (B, T, 128) down to a fixed vector (B, 128)
        # Then map to num_classes (206).
        self.fc_head = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.cnn_blocks(x) # (B, 64, 128, 313)
        
        # 2. Pool Mel dimension
        x = self.mel_pool(x) # (B, 64, 1, 313)
        
        # 3. Permute for GRU: (B, C, T) -> (B, T, C)
        # Here C=64, T=313
        x = x.squeeze(2).permute(0, 2, 1) # (B, 313, 64)
        
        # 4. GRU Sequence Modeling
        # Output: (B, T, H) -> (B, 313, 128)
        output, _ = self.gru(x)
        
        # 5. Sequence Pooling: Average over the time dimension (dim=1)
        # Result: (B, 128)
        pooled_output = torch.mean(output, dim=1)
        
        # 6. Final Linear Classification Head
        logits = self.fc_head(pooled_output) # (B, 206)
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
    
    # Variables derived from the dataset profile (hardcoded for this task)
    NUM_CLASSES = 206 

    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / persistent_workers / prefetch_factor
        # are read from BIRDCLEF_* env vars (sourced from config.yaml).
        train_loader, val_loader, num_classes_check = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        
        # Safety check (should match 206)
        if num_classes_check != NUM_CLASSES:
             raise ValueError(f"Expected {NUM_CLASSES} classes, but data loader provided {num_classes_check}")
        
        print(
            f"data loaded: {num_classes_check} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybridModel(num_classes=NUM_CLASSES)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # The final layer is nn.Linear, which is not lazy, but we run this
        # check to ensure correct parameter counting/initialization.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mel=128, Time=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        PATIENCE = 3
        best_auc = 0.0
        epochs_no_improve = 0
        
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
                    # Sigmoid for probabilities, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError: # Handle case where all labels are the same in a column
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score computes AP, which is the integral of PR curve.
                    # For BirdCLEF, this serves as the proxy for the required metric.
                    try:
                        aps.append(average_precision_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        pass
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
            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

        # Final results compilation
        final_metrics = {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "cmap_at_5": curves["cmap_at_5"][-1],
            "f1_macro": curves["f1_macro"][-1],
            "loss": curves["loss"][-1],
        }
        
        results = {
            "metrics": final_metrics,
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_metrics['roc_auc_macro']:.4f} "
            f"cmap@5={final_metrics['cmap_at_5']:.4f} "
            f"f1={final_metrics['f1_macro']:.4f}",
            flush=True,
        )
        
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    # Write results.json
    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Error that crashed it
Type: ScriptReportedError

Message:
NameError: name 'EPOCHS' is not defined

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

# === Hyperparameters from the proposal ===
# Note: LR is set in the skeleton block below, but we track the proposal's value: 0.001
LR = 1e-3

# Read EPOCHS from environment variable as per mandatory structure
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))

# Overriding default augmentation to match the proposal
AUGMENTATION = {
    "time_shift": False,
    "noise_injection": False,
    "mixup": 0.0,
    "specaugment": True
}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128)
    for temporal sequence modeling.
    Input: (B, 1, Mel, Time) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Block CNN Feature Extractor ---
        # Input: (B, 1, 128, 313)
        self.cnn_blocks = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Output: (B, 32, 128, 313)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 -> 64 channels (keeping feature depth consistent for GRU)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsample the Mel dimension (128) to 1, keeping Time (313) intact.
        self.mel_pool = nn.AdaptiveAvgPool2d((1, 1)) # Output: (B, 64, 1, 313)
        
        # --- GRU Layer ---
        # Input to GRU: (B, T, F) -> T=313, F=64
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True # Crucial for sequence processing
        )
        
        # --- Final Classification Head ---
        # After GRU, we pool the sequence output (B, T, 128) down to a fixed vector (B, 128)
        # Then map to num_classes (206).
        self.fc_head = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.cnn_blocks(x) # (B, 64, 128, 313)
        
        # 2. Pool Mel dimension
        x = self.mel_pool(x) # (B, 64, 1, 313)
        
        # 3. Permute for GRU: (B, C, T) -> (B, T, C)
        # Here C=64, T=313
        x = x.squeeze(2).permute(0, 2, 1) # (B, 313, 64)
        
        # 4. GRU Sequence Modeling
        # Output: (B, T, H) -> (B, 313, 128)
        output, _ = self.gru(x)
        
        # 5. Sequence Pooling: Average over the time dimension (dim=1)
        # Result: (B, 128)
        pooled_output = torch.mean(output, dim=1)
        
        # 6. Final Linear Classification Head
        logits = self.fc_head(pooled_output) # (B, 206)
        return logits


# ======================================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    
    # Variables derived from the dataset profile (hardcoded for this task)
    NUM_CLASSES = 206 

    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / persistent_workers / prefetch_factor
        # are read from BIRDCLEF_* env vars (sourced from config.yaml).
        train_loader, val_loader, num_classes_check = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        
        # Safety check (should match 206)
        if num_classes_check != NUM_CLASSES:
             raise ValueError(f"Expected {NUM_CLASSES} classes, but data loader provided {num_classes_check}")
        
        print(
            f"data loaded: {num_classes_check} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybridModel(num_classes=NUM_CLASSES)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # The final layer is nn.Linear, which is not lazy, but we run this
        # check to ensure correct parameter counting/initialization.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mel=128, Time=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        PATIENCE = 3
        best_auc = 0.0
        epochs_no_improve = 0
        
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
                    # Sigmoid for probabilities, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError: # Handle case where all labels are the same in a column
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score computes AP, which is the integral of PR curve.
                    # For BirdCLEF, this serves as the proxy for the required metric.
                    try:
                        aps.append(average_precision_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        pass
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
            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

        # Final results compilation
        final_metrics = {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "cmap_at_5": curves["cmap_at_5"][-1],
            "f1_macro": curves["f1_macro"][-1],
            "loss": curves["loss"][-1],
        }
        
        results = {
            "metrics": final_metrics,
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_metrics['roc_auc_macro']:.4f} "
            f"cmap@5={final_metrics['cmap_at_5']:.4f} "
            f"f1={final_metrics['f1_macro']:.4f}",
            flush=True,
        )
        
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    # Write results.json
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

# === Hyperparameters from the proposal ===
# Note: LR is set in the skeleton block below, but we track the proposal's value: 0.001
LR = 1e-3

# Read EPOCHS from environment variable as per mandatory structure
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))

# Overriding default augmentation to match the proposal
AUGMENTATION = {
    "time_shift": False,
    "noise_injection": False,
    "mixup": 0.0,
    "specaugment": True
}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    3-block CNN feature extractor (32->64 channels) followed by nn.GRU(128)
    for temporal sequence modeling.
    Input: (B, 1, Mel, Time) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 3-Block CNN Feature Extractor ---
        # Input: (B, 1, 128, 313)
        self.cnn_blocks = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Output: (B, 32, 128, 313)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 -> 64 channels (keeping feature depth consistent for GRU)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Output: (B, 64, 128, 313)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsample the Mel dimension (128) to 1, keeping Time (313) intact.
        self.mel_pool = nn.AdaptiveAvgPool2d((1, 1)) # Output: (B, 64, 1, 313)
        
        # --- GRU Layer ---
        # Input to GRU: (B, T, F) -> T=313, F=64
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True # Crucial for sequence processing
        )
        
        # --- Final Classification Head ---
        # After GRU, we pool the sequence output (B, T, 128) down to a fixed vector (B, 128)
        # Then map to num_classes (206).
        self.fc_head = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.cnn_blocks(x) # (B, 64, 128, 313)
        
        # 2. Pool Mel dimension
        x = self.mel_pool(x) # (B, 64, 1, 313)
        
        # 3. Permute for GRU: (B, C, T) -> (B, T, C)
        # Here C=64, T=313
        x = x.squeeze(2).permute(0, 2, 1) # (B, 313, 64)
        
        # 4. GRU Sequence Modeling
        # Output: (B, T, H) -> (B, 313, 128)
        output, _ = self.gru(x)
        
        # 5. Sequence Pooling: Average over the time dimension (dim=1)
        # Result: (B, 128)
        pooled_output = torch.mean(output, dim=1)
        
        # 6. Final Linear Classification Head
        logits = self.fc_head(pooled_output) # (B, 206)
        return logits


# ======================================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# =================================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    
    # Variables derived from the dataset profile (hardcoded for this task)
    NUM_CLASSES = 206 

    try:
        print("loading data...", flush=True)
        # batch_size / num_workers / persistent_workers / prefetch_factor
        # are read from BIRDCLEF_* env vars (sourced from config.yaml).
        train_loader, val_loader, num_classes_check = load_precomputed_dataset(
            augmentation=AUGMENTATION,
        )
        
        # Safety check (should match 206)
        if num_classes_check != NUM_CLASSES:
             raise ValueError(f"Expected {NUM_CLASSES} classes, but data loader provided {num_classes_check}")
        
        print(
            f"data loaded: {num_classes_check} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        # Instantiate the model here. IMMEDIATELY move to the selected device.
        model = CnnGruHybridModel(num_classes=NUM_CLASSES)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # The final layer is nn.Linear, which is not lazy, but we run this
        # check to ensure correct parameter counting/initialization.
        with torch.no_grad():
            # Dummy input shape: (Batch=1, Channels=1, Mel=128, Time=313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
        PATIENCE = 3
        best_auc = 0.0
        epochs_no_improve = 0
        
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
                    # Sigmoid for probabilities, move to CPU for numpy/sklearn
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    try:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                    except ValueError: # Handle case where all labels are the same in a column
                        pass
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    # average_precision_score computes AP, which is the integral of PR curve.
                    # For BirdCLEF, this serves as the proxy for the required metric.
                    try:
                        aps.append(average_precision_score(targs[:, c], probs[:, c]))
                    except ValueError:
                        pass
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
            scheduler.step()

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

        # Final results compilation
        final_metrics = {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "cmap_at_5": curves["cmap_at_5"][-1],
            "f1_macro": curves["f1_macro"][-1],
            "loss": curves["loss"][-1],
        }
        
        results = {
            "metrics": final_metrics,
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={final_metrics['roc_auc_macro']:.4f} "
            f"cmap@5={final_metrics['cmap_at_5']:.4f} "
            f"f1={final_metrics['f1_macro']:.4f}",
            flush=True,
        )
        
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    # Write results.json
    with open("results.json", "w") as fh:
        json.dump(results, fh)

- **inner_retries:** 0
