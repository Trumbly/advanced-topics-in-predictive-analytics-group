# Task exp_003_recovery_03_error_recovery

- **Experiment:** exp_003
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-13 02:54:03.937426+00:00
- **Completed:** 2026-04-13 02:56:08.082154+00:00

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
  - `EpochsCapExceeded: EPOCHS = N exceeds the hard cap of 1`:
      Change the literal `EPOCHS = N` line to `EPOCHS = 1`. Do not use
      `max(1, ...)` or `min(..., 1)` — just write `EPOCHS = 1`.
  - `AttributeError: 'Conv2dNormActivation' object has no attribute 'kernel_size'`:
      MobileNetV3's first layer is `Conv2dNormActivation`, not `nn.Conv2d`.
      Do NOT access `.kernel_size` on it. Use the TorchvisionAdapter from
      the registry instead of manually modifying pretrained model layers.


[USER]
## Original architecture proposal
{
  "architecture": "[cnn_gru_hybrid] 3-conv CNN front-end (32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head",
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

# Set number of threads if on CPU
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# === Hyperparameters ===
# EPOCHS must be hardcoded to 1 as per instructions
EPOCHS = 1
LR = 1e-3
MODEL_DROPOUT = 0.2
BATCH_SIZE = int(os.environ.get("BIRDCLEF_BATCH_SIZE", 128))
NUM_WORKERS = int(os.environ.get("BIRDCLEF_NUM_WORKERS", 0))
# FIX: Reading environment variables for workers that might contain 'true' instead of integers.
# Defaulting to 0 if the environment variable is non-numeric or missing.
PERSISTENT_WORKERS_ENV = os.environ.get("BIRDCLEF_PERSISTENT_WORKERS", "0")
try:
    PERSISTENT_WORKERS = int(PERSISTENT_WORKERS_ENV)
except ValueError:
    PERSISTENT_WORKERS = 0

PREFETCH_FACTOR = int(os.environ.get("BIRDCLEF_PREFETCH_FACTOR", 1))


# =============================================================================
# MODEL DEFINITION: CNN-GRU Hybrid
# Architecture: 3-conv CNN front-end (1->32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head
# Input: (B, 1, 128, 313)
# =============================================================================

class CnnGruHybridModel(nn.Module):
    """
    CNN-GRU Hybrid model for spectrogram processing.
    Uses 3 consecutive Conv blocks to extract features, pools spatially,
    and passes the resulting feature sequence through a GRU.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. CNN Front-end (1 -> 32 -> 64) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Block 3: 64 -> 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- 2. Pooling and Feature Flattening ---
        self.pool_h = nn.AdaptiveAvgPool2d((1, None)) # Pool H=128 to 1, keep W=313
        
        # --- 3. GRU Head ---
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Final linear layer to map the GRU's hidden state dimension (128) 
        # to the required number of classes.
        self.fc = nn.LazyLinear(num_classes)
        
        self.dropout = nn.Dropout(MODEL_DROPOUT)


    def forward(self, x):
        # x shape: (B, 1, 128, 313)
        
        # CNN Stack
        x = self.relu(self.conv1(x)) # (B, 32, 128, 313)
        x = self.relu(self.conv2(x)) # (B, 64, 128, 313)
        x = self.relu(self.conv3(x)) # (B, 64, 128, 313)
        
        # Pool Height dimension: (B, 64, 128, 313) -> (B, 64, 1, 313)
        x = self.pool_h(x) 

        # Squeeze the height dimension: (B, 64, 1, 313) -> (B, 64, 313)
        x = x.squeeze(2) 

        # Pass through GRU: (B, T, C) -> (B, T, H) where T=313, C=64, H=128
        # Currently: (B, 64, 313). We need (B, 313, 64) for batch_first=True.
        x = x.permute(0, 2, 1) # (B, 313, 64)
        
        # GRU Output: (B, 313, 128)
        gru_output, _ = self.gru(x)
        
        # Average the sequence output over time: (B, 313, 128) -> (B, 128)
        pooled_output = torch.mean(gru_output, dim=1)

        # Apply dropout
        x = self.dropout(pooled_output)

        # Final linear layer: (B, 128) -> (B, num_classes)
        return self.fc(x)


def train_one_epoch(model, data_loader, pos_weight, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Progress print for training start
    print("Starting training epoch...")
    print(f"Device: {device}, Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    if DEVICE_NAME == "cuda":
        torch.cuda.empty_cache()

    for batch_idx, (x, y) in enumerate(data_loader):
        # Move batch to device
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x) # (B, num_classes)
        
        # Calculate loss
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress print for batch completion
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f} | Avg Loss: {total_loss / (batch_idx + 1):.4f}", flush=True)

    avg_loss = total_loss / len(data_loader)
    print(f"Finished training epoch. Average Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(model, data_loader, pos_weight, device):
    model.eval()
    print("\nStarting evaluation...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            
            # Forward pass
            output = model(x)
            
            # Calculate probabilities (Sigmoid applied before numpy conversion)
            probs = torch.sigmoid(output).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    # Concatenate results
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    
    print("Evaluation complete. Calculating metrics...")
    
    # Calculate metrics (Macro-averaged ROC-AUC)
    try:
        # ROC AUC requires probabilities (one per class) and binary labels.
        roc_auc = roc_auc_score(labels_np, probs_np, average='macro', multi_label=True)
        
        avg_precision = average_precision_score(labels_np, probs_np, average='macro', multi_label=True)
        
        # F1 score is calculated using argmax for simplicity in this context
        f1 = f1_score(labels_np, np.argmax(probs_np, axis=1), average='macro')

        metrics = {
            "roc_auc_macro": roc_auc,
            "avg_precision_macro": avg_precision,
            "f1_macro": f1
        }
        return metrics
    except Exception as e:
        print(f"Error during metric calculation: {e}. Returning dummy success metrics.")
        return {"roc_auc_macro": 0.0, "avg_precision_macro": 0.0, "f1_macro": 0.0}


if __name__ == "__main__":
    # 1. Load Data and Determine Class Count
    print("Loading precomputed dataset...")
    # This function reads from environment variables for batch_size, workers, etc.
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    print(f"Data loaded successfully. Number of classes: {num_classes}")

    # 2. Initialize Model, Optimizer, and Pos Weight
    model = CnnGruHybridModel(num_classes=num_classes).to(device)
    
    # FIX: Dummy forward pass required for LazyLinear initialization
    print("Initializing model parameters...")
    # Dummy input shape: (Batch, Channels, Height, Width) -> (1, 1, 128, 313)
    dummy_input = torch.zeros(1, 1, 128, 313, device=device)
    with torch.no_grad():
        # A dummy run forces LazyLinear weights to materialize
        model(dummy_input)
    print("Model parameters materialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    pos_weight = compute_pos_weight().to(device)

    # 3. Training Loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    best_metrics = {"roc_auc_macro": -1.0}
    
    for epoch in range(EPOCHS):
        # Training Phase
        train_loss = train_one_epoch(model, train_loader, pos_weight, optimizer, device)
        
        # Validation Phase
        metrics = evaluate_model(model, val_loader, pos_weight, device)
        
        print("\n" + "="*20 + " EPOCH END " + "="*20)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        print("="*50)
        
        # Simple checkpointing logic (only track best ROC-AUC)
        if metrics['roc_auc_macro'] > best_metrics["roc_auc_macro"]:
            best_metrics = metrics
            print("New best model checkpoint saved.")

    # 4. Final Results and Cleanup
    print("\nTraining finished.")
    final_results = {
        "best_metrics": best_metrics,
        "average_epoch_loss": train_loss,
        "epochs_run": EPOCHS
    }
    
    # Write results.json
    try:
        with open("results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        print("\nSUCCESS: Results written to results.json")
    except Exception as e:
        print(f"\nERROR: Could not write results.json: {e}")
        # Write error key if file writing fails
        error_results = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(error_results, f, indent=4)
        print("FAILURE: Wrote error key to results.json instead.")
    pass

```

## Error that crashed it
Type: MetricsParseError

Message:
/Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260413_021453_baseline_run_v3/exp_003/results.json: missing required metrics ['roc_auc_macro', 'loss']. Got: []

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

# Set number of threads if on CPU
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# === Hyperparameters ===
# EPOCHS must be hardcoded to 1 as per instructions
EPOCHS = 1
LR = 1e-3
MODEL_DROPOUT = 0.2
BATCH_SIZE = int(os.environ.get("BIRDCLEF_BATCH_SIZE", 128))
NUM_WORKERS = int(os.environ.get("BIRDCLEF_NUM_WORKERS", 0))
# FIX: Reading environment variables for workers that might contain 'true' instead of integers.
# Defaulting to 0 if the environment variable is non-numeric or missing.
PERSISTENT_WORKERS_ENV = os.environ.get("BIRDCLEF_PERSISTENT_WORKERS", "0")
try:
    PERSISTENT_WORKERS = int(PERSISTENT_WORKERS_ENV)
except ValueError:
    PERSISTENT_WORKERS = 0

PREFETCH_FACTOR = int(os.environ.get("BIRDCLEF_PREFETCH_FACTOR", 1))


# =============================================================================
# MODEL DEFINITION: CNN-GRU Hybrid
# Architecture: 3-conv CNN front-end (1->32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head
# Input: (B, 1, 128, 313)
# =============================================================================

class CnnGruHybridModel(nn.Module):
    """
    CNN-GRU Hybrid model for spectrogram processing.
    Uses 3 consecutive Conv blocks to extract features, pools spatially,
    and passes the resulting feature sequence through a GRU.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. CNN Front-end (1 -> 32 -> 64) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Block 3: 64 -> 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- 2. Pooling and Feature Flattening ---
        self.pool_h = nn.AdaptiveAvgPool2d((1, None)) # Pool H=128 to 1, keep W=313
        
        # --- 3. GRU Head ---
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Final linear layer to map the GRU's hidden state dimension (128) 
        # to the required number of classes.
        self.fc = nn.LazyLinear(num_classes)
        
        self.dropout = nn.Dropout(MODEL_DROPOUT)


    def forward(self, x):
        # x shape: (B, 1, 128, 313)
        
        # CNN Stack
        x = self.relu(self.conv1(x)) # (B, 32, 128, 313)
        x = self.relu(self.conv2(x)) # (B, 64, 128, 313)
        x = self.relu(self.conv3(x)) # (B, 64, 128, 313)
        
        # Pool Height dimension: (B, 64, 128, 313) -> (B, 64, 1, 313)
        x = self.pool_h(x) 

        # Squeeze the height dimension: (B, 64, 1, 313) -> (B, 64, 313)
        x = x.squeeze(2) 

        # Pass through GRU: (B, T, C) -> (B, T, H) where T=313, C=64, H=128
        # Currently: (B, 64, 313). We need (B, 313, 64) for batch_first=True.
        x = x.permute(0, 2, 1) # (B, 313, 64)
        
        # GRU Output: (B, 313, 128)
        gru_output, _ = self.gru(x)
        
        # Average the sequence output over time: (B, 313, 128) -> (B, 128)
        pooled_output = torch.mean(gru_output, dim=1)

        # Apply dropout
        x = self.dropout(pooled_output)

        # Final linear layer: (B, 128) -> (B, num_classes)
        return self.fc(x)


def train_one_epoch(model, data_loader, pos_weight, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Progress print for training start
    print("Starting training epoch...")
    print(f"Device: {device}, Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    if DEVICE_NAME == "cuda":
        torch.cuda.empty_cache()

    for batch_idx, (x, y) in enumerate(data_loader):
        # Move batch to device
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x) # (B, num_classes)
        
        # Calculate loss
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress print for batch completion
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f} | Avg Loss: {total_loss / (batch_idx + 1):.4f}", flush=True)

    avg_loss = total_loss / len(data_loader)
    print(f"Finished training epoch. Average Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(model, data_loader, pos_weight, device):
    model.eval()
    print("\nStarting evaluation...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            
            # Forward pass
            output = model(x)
            
            # Calculate probabilities (Sigmoid applied before numpy conversion)
            probs = torch.sigmoid(output).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    # Concatenate results
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    
    print("Evaluation complete. Calculating metrics...")
    
    # Calculate metrics (Macro-averaged ROC-AUC)
    try:
        # ROC AUC requires probabilities (one per class) and binary labels.
        roc_auc = roc_auc_score(labels_np, probs_np, average='macro', multi_label=True)
        
        avg_precision = average_precision_score(labels_np, probs_np, average='macro', multi_label=True)
        
        # F1 score is calculated using argmax for simplicity in this context
        f1 = f1_score(labels_np, np.argmax(probs_np, axis=1), average='macro')

        metrics = {
            "roc_auc_macro": roc_auc,
            "avg_precision_macro": avg_precision,
            "f1_macro": f1
        }
        return metrics
    except Exception as e:
        print(f"Error during metric calculation: {e}. Returning dummy success metrics.")
        return {"roc_auc_macro": 0.0, "avg_precision_macro": 0.0, "f1_macro": 0.0}


if __name__ == "__main__":
    # 1. Load Data and Determine Class Count
    print("Loading precomputed dataset...")
    # This function reads from environment variables for batch_size, workers, etc.
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    print(f"Data loaded successfully. Number of classes: {num_classes}")

    # 2. Initialize Model, Optimizer, and Pos Weight
    model = CnnGruHybridModel(num_classes=num_classes).to(device)
    
    # FIX: Dummy forward pass required for LazyLinear initialization
    print("Initializing model parameters...")
    # Dummy input shape: (Batch, Channels, Height, Width) -> (1, 1, 128, 313)
    dummy_input = torch.zeros(1, 1, 128, 313, device=device)
    with torch.no_grad():
        # A dummy run forces LazyLinear weights to materialize
        model(dummy_input)
    print("Model parameters materialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    pos_weight = compute_pos_weight().to(device)

    # 3. Training Loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    best_metrics = {"roc_auc_macro": -1.0}
    
    for epoch in range(EPOCHS):
        # Training Phase
        train_loss = train_one_epoch(model, train_loader, pos_weight, optimizer, device)
        
        # Validation Phase
        metrics = evaluate_model(model, val_loader, pos_weight, device)
        
        print("\n" + "="*20 + " EPOCH END " + "="*20)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        print("="*50)
        
        # Simple checkpointing logic (only track best ROC-AUC)
        if metrics['roc_auc_macro'] > best_metrics["roc_auc_macro"]:
            best_metrics = metrics
            print("New best model checkpoint saved.")

    # 4. Final Results and Cleanup
    print("\nTraining finished.")
    
    # FIX: Updated structure to explicitly include 'roc_auc_macro' and 'loss'
    final_results = {
        "roc_auc_macro": best_metrics["roc_auc_macro"],
        "loss": train_loss, # Using the average loss of the final epoch as the required 'loss' metric
        "best_metrics": best_metrics,
        "average_epoch_loss": train_loss,
        "epochs_run": EPOCHS
    }
    
    # Write results.json
    try:
        with open("results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        print("\nSUCCESS: Results written to results.json")
    except Exception as e:
        print(f"\nERROR: Could not write results.json: {e}")
        # Write error key if file writing fails
        error_results = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(error_results, f, indent=4)
        print("FAILURE: Wrote error key to results.json instead.")
    pass
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

# Set number of threads if on CPU
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# === Hyperparameters ===
# EPOCHS must be hardcoded to 1 as per instructions
EPOCHS = 1
LR = 1e-3
MODEL_DROPOUT = 0.2
BATCH_SIZE = int(os.environ.get("BIRDCLEF_BATCH_SIZE", 128))
NUM_WORKERS = int(os.environ.get("BIRDCLEF_NUM_WORKERS", 0))
# FIX: Reading environment variables for workers that might contain 'true' instead of integers.
# Defaulting to 0 if the environment variable is non-numeric or missing.
PERSISTENT_WORKERS_ENV = os.environ.get("BIRDCLEF_PERSISTENT_WORKERS", "0")
try:
    PERSISTENT_WORKERS = int(PERSISTENT_WORKERS_ENV)
except ValueError:
    PERSISTENT_WORKERS = 0

PREFETCH_FACTOR = int(os.environ.get("BIRDCLEF_PREFETCH_FACTOR", 1))


# =============================================================================
# MODEL DEFINITION: CNN-GRU Hybrid
# Architecture: 3-conv CNN front-end (1->32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head
# Input: (B, 1, 128, 313)
# =============================================================================

class CnnGruHybridModel(nn.Module):
    """
    CNN-GRU Hybrid model for spectrogram processing.
    Uses 3 consecutive Conv blocks to extract features, pools spatially,
    and passes the resulting feature sequence through a GRU.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # --- 1. CNN Front-end (1 -> 32 -> 64) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Block 3: 64 -> 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- 2. Pooling and Feature Flattening ---
        self.pool_h = nn.AdaptiveAvgPool2d((1, None)) # Pool H=128 to 1, keep W=313
        
        # --- 3. GRU Head ---
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Final linear layer to map the GRU's hidden state dimension (128) 
        # to the required number of classes.
        self.fc = nn.LazyLinear(num_classes)
        
        self.dropout = nn.Dropout(MODEL_DROPOUT)


    def forward(self, x):
        # x shape: (B, 1, 128, 313)
        
        # CNN Stack
        x = self.relu(self.conv1(x)) # (B, 32, 128, 313)
        x = self.relu(self.conv2(x)) # (B, 64, 128, 313)
        x = self.relu(self.conv3(x)) # (B, 64, 128, 313)
        
        # Pool Height dimension: (B, 64, 128, 313) -> (B, 64, 1, 313)
        x = self.pool_h(x) 

        # Squeeze the height dimension: (B, 64, 1, 313) -> (B, 64, 313)
        x = x.squeeze(2) 

        # Pass through GRU: (B, T, C) -> (B, T, H) where T=313, C=64, H=128
        # Currently: (B, 64, 313). We need (B, 313, 64) for batch_first=True.
        x = x.permute(0, 2, 1) # (B, 313, 64)
        
        # GRU Output: (B, 313, 128)
        gru_output, _ = self.gru(x)
        
        # Average the sequence output over time: (B, 313, 128) -> (B, 128)
        pooled_output = torch.mean(gru_output, dim=1)

        # Apply dropout
        x = self.dropout(pooled_output)

        # Final linear layer: (B, 128) -> (B, num_classes)
        return self.fc(x)


def train_one_epoch(model, data_loader, pos_weight, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Progress print for training start
    print("Starting training epoch...")
    print(f"Device: {device}, Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    if DEVICE_NAME == "cuda":
        torch.cuda.empty_cache()

    for batch_idx, (x, y) in enumerate(data_loader):
        # Move batch to device
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x) # (B, num_classes)
        
        # Calculate loss
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress print for batch completion
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f} | Avg Loss: {total_loss / (batch_idx + 1):.4f}", flush=True)

    avg_loss = total_loss / len(data_loader)
    print(f"Finished training epoch. Average Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(model, data_loader, pos_weight, device):
    model.eval()
    print("\nStarting evaluation...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            
            # Forward pass
            output = model(x)
            
            # Calculate probabilities (Sigmoid applied before numpy conversion)
            probs = torch.sigmoid(output).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

    # Concatenate results
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    
    print("Evaluation complete. Calculating metrics...")
    
    # Calculate metrics (Macro-averaged ROC-AUC)
    try:
        # ROC AUC requires probabilities (one per class) and binary labels.
        roc_auc = roc_auc_score(labels_np, probs_np, average='macro', multi_label=True)
        
        avg_precision = average_precision_score(labels_np, probs_np, average='macro', multi_label=True)
        
        # F1 score is calculated using argmax for simplicity in this context
        f1 = f1_score(labels_np, np.argmax(probs_np, axis=1), average='macro')

        metrics = {
            "roc_auc_macro": roc_auc,
            "avg_precision_macro": avg_precision,
            "f1_macro": f1
        }
        return metrics
    except Exception as e:
        print(f"Error during metric calculation: {e}. Returning dummy success metrics.")
        return {"roc_auc_macro": 0.0, "avg_precision_macro": 0.0, "f1_macro": 0.0}


if __name__ == "__main__":
    # 1. Load Data and Determine Class Count
    print("Loading precomputed dataset...")
    # This function reads from environment variables for batch_size, workers, etc.
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    print(f"Data loaded successfully. Number of classes: {num_classes}")

    # 2. Initialize Model, Optimizer, and Pos Weight
    model = CnnGruHybridModel(num_classes=num_classes).to(device)
    
    # FIX: Dummy forward pass required for LazyLinear initialization
    print("Initializing model parameters...")
    # Dummy input shape: (Batch, Channels, Height, Width) -> (1, 1, 128, 313)
    dummy_input = torch.zeros(1, 1, 128, 313, device=device)
    with torch.no_grad():
        # A dummy run forces LazyLinear weights to materialize
        model(dummy_input)
    print("Model parameters materialized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    pos_weight = compute_pos_weight().to(device)

    # 3. Training Loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    best_metrics = {"roc_auc_macro": -1.0}
    
    for epoch in range(EPOCHS):
        # Training Phase
        train_loss = train_one_epoch(model, train_loader, pos_weight, optimizer, device)
        
        # Validation Phase
        metrics = evaluate_model(model, val_loader, pos_weight, device)
        
        print("\n" + "="*20 + " EPOCH END " + "="*20)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        print("="*50)
        
        # Simple checkpointing logic (only track best ROC-AUC)
        if metrics['roc_auc_macro'] > best_metrics["roc_auc_macro"]:
            best_metrics = metrics
            print("New best model checkpoint saved.")

    # 4. Final Results and Cleanup
    print("\nTraining finished.")
    
    # FIX: Updated structure to explicitly include 'roc_auc_macro' and 'loss'
    final_results = {
        "roc_auc_macro": best_metrics["roc_auc_macro"],
        "loss": train_loss, # Using the average loss of the final epoch as the required 'loss' metric
        "best_metrics": best_metrics,
        "average_epoch_loss": train_loss,
        "epochs_run": EPOCHS
    }
    
    # Write results.json
    try:
        with open("results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        print("\nSUCCESS: Results written to results.json")
    except Exception as e:
        print(f"\nERROR: Could not write results.json: {e}")
        # Write error key if file writing fails
        error_results = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(error_results, f, indent=4)
        print("FAILURE: Wrote error key to results.json instead.")
    pass

- **inner_retries:** 0
