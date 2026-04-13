# Task exp_010_recovery_02_error_recovery

- **Experiment:** exp_010
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-13 00:21:25.661762+00:00
- **Completed:** 2026-04-13 00:22:28.495505+00:00

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
  "architecture": "[cnn_attention] 3-block Conv stack (64 channels) -> Self-Attention over time dimension -> Global Average Pooling -> Linear Head",
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

# === Hyperparameters (Using env vars as source of truth, but respecting hard cap) ===
# Mandatory structure requires EPOCHS = 1 literal if the environment variable approach is too complex, 
# but sticking to the provided env var logic while ensuring the *value* used is fixed if the error was elsewhere.
# Since the error is structural (missing file write), I'll keep the env var reading but ensure the loop logic is sound.
EPOCHS = 1 
LR = 1e-3
# Augmentation settings from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnAttentionModel(nn.Module):
    """
    Implements the proposed 3-block CNN stack -> Attention (simulated via advanced pooling) -> Linear Head.
    The architecture processes the input (B, 1, Mel, Time) through 3 Conv layers
    with intermediate feature expansion, ending with 64 channels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 3-block Conv Stack (Targeting 64 output channels) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Block 2: 64 -> 128 (Intermediate expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Block 3: 128 -> 64 (Final projection to 64 channels)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # --- Attention / Feature Aggregation ---
        # Pool over H and T dimensions to get a fixed feature size (B, 64, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Linear Head ---
        # Use LazyLinear to handle the inferred input size (64 channels)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Shape: (B, 64, H, T)

        # 2. Global Pooling (Collapses H and T to 1x1)
        x = self.global_pool(x)
        # Shape: (B, 64, 1, 1)

        # 3. Flatten and Linear Projection
        # Flatten the last two dimensions (C, 1, 1) -> (C)
        x = x.view(x.size(0), -1)
        
        # Project to class logits
        x = self.head(x)
        return x

if __name__ == "__main__":
    print(f"--- Starting BirdCLEF Training Run ---")
    print(f"Device: {device}")
    
    # 1. Load Dataset and get metadata
    print("Loading precomputed dataset...")
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    
    print(f"Dataset loaded. Number of classes: {num_classes}")

    # 2. Setup Loss and Pos_Weight
    pos_weight = compute_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Model Initialization
    model = CnnAttentionModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    
    print("Model initialized.")

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move batch data to device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Loss calculation
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_loss = running_loss / train_loader.batch_size
        print(f"Training Loss: {epoch_loss:.4f}")
        torch.cuda.empty_cache()
        # Flush output to ensure progress print is visible
        print("Training step complete.", flush=True)

    # 5. Validation/Testing Loop (Evaluation)
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n--- Starting Validation ---")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Get probabilities (sigmoid applied)
            probs = torch.sigmoid(logits)
            
            # Move predictions back to CPU and append
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            print(f"Validation batch {batch_idx+1}/{len(val_loader)} complete.", flush=True)

    # 6. Metric Calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate macro-averaged ROC-AUC
    roc_auc = roc_auc_score(all_targets, all_preds, average='macro')
    # Calculate macro-averaged Average Precision Score
    ap_score = average_precision_score(all_targets, all_preds, average='macro')
    # Calculate macro-averaged F1 Score (Requires thresholding, often done post-hoc, but included for completeness)
    # For AUC/AP, we use the raw probabilities.
    
    metrics = {
        "roc_auc_macro": roc_auc,
        "ap_score_macro": ap_score
    }
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Metrics calculated: {metrics}")

    # 7. Write results.json (FIX for NoResultsFile error)
    output_filename = "results.json"
    results_data = {
        "model_architecture": "cnn_attention_model",
        "hyperparameters": {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": os.environ.get("BIRDCLEF_BATCH_SIZE", "128"),
            "device": DEVICE_NAME
        },
        "metrics": metrics
    }
    
    with open(output_filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\nSuccessfully wrote results to {output_filename}", flush=True)<unused56>

```

## Error that crashed it
Type: SyntaxError

Message:
Line 179: invalid syntax

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
from sklearn.metrics import roc_auc_score, average_precision_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# Set number of threads if running on CPU
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# === Hyperparameters (Using env vars as source of truth, but respecting hard cap) ===
EPOCHS = 1 
LR = 1e-3
# Augmentation settings from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnAttentionModel(nn.Module):
    """
    Implements the proposed 3-block CNN stack -> Attention (simulated via advanced pooling) -> Linear Head.
    The architecture processes the input (B, 1, Mel, Time) through 3 Conv layers
    with intermediate feature expansion, ending with 64 channels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 3-block Conv Stack (Targeting 64 output channels) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Block 2: 64 -> 128 (Intermediate expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Block 3: 128 -> 64 (Final projection to 64 channels)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # --- Attention / Feature Aggregation ---
        # Pool over H and T dimensions to get a fixed feature size (B, 64, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Linear Head ---
        # Use LazyLinear to handle the inferred input size (64 channels)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Shape: (B, 64, H, T)

        # 2. Global Pooling (Collapses H and T to 1x1)
        x = self.global_pool(x)
        # Shape: (B, 64, 1, 1)

        # 3. Flatten and Linear Projection
        # Flatten the last two dimensions (C, 1, 1) -> (C)
        x = x.view(x.size(0), -1)
        
        # Project to class logits
        x = self.head(x)
        return x

if __name__ == "__main__":
    print(f"--- Starting BirdCLEF Training Run ---")
    print(f"Device: {device}")
    
    # 1. Load Dataset and get metadata
    print("Loading precomputed dataset...")
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    
    print(f"Dataset loaded. Number of classes: {num_classes}")

    # 2. Setup Loss and Pos_Weight
    pos_weight = compute_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Model Initialization
    model = CnnAttentionModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    
    print("Model initialized.")

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move batch data to device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Loss calculation
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_loss = running_loss / train_loader.batch_size
        print(f"Training Loss: {epoch_loss:.4f}")
        if device.type() == 'cuda':
             torch.cuda.empty_cache()
        # Flush output to ensure progress print is visible
        print("Training step complete.", flush=True)

    # 5. Validation/Testing Loop (Evaluation)
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n--- Starting Validation ---")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Get probabilities (sigmoid applied)
            probs = torch.sigmoid(logits)
            
            # Move predictions back to CPU and append
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            print(f"Validation batch {batch_idx+1}/{len(val_loader)} complete.", flush=True)

    # 6. Metric Calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate macro-averaged ROC-AUC
    roc_auc = roc_auc_score(all_targets, all_preds, average='macro')
    # Calculate macro-averaged Average Precision Score
    ap_score = average_precision_score(all_targets, all_preds, average='macro')
    
    metrics = {
        "roc_auc_macro": roc_auc,
        "ap_score_macro": ap_score
    }
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Metrics calculated: {metrics}")

    # 7. Write results.json
    output_filename = "results.json"
    results_data = {
        "model_architecture": "cnn_attention_model",
        "hyperparameters": {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": os.environ.get("BIRDCLEF_BATCH_SIZE", "128"),
            "device": DEVICE_NAME
        },
        "metrics": metrics
    }
    
    with open(output_filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\nSuccessfully wrote results to {output_filename}", flush=True)
```

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# Set number of threads if running on CPU
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

# === Hyperparameters (Using env vars as source of truth, but respecting hard cap) ===
EPOCHS = 1 
LR = 1e-3
# Augmentation settings from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnAttentionModel(nn.Module):
    """
    Implements the proposed 3-block CNN stack -> Attention (simulated via advanced pooling) -> Linear Head.
    The architecture processes the input (B, 1, Mel, Time) through 3 Conv layers
    with intermediate feature expansion, ending with 64 channels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 3-block Conv Stack (Targeting 64 output channels) ---
        # Input: (B, 1, 128, 313)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # Block 2: 64 -> 128 (Intermediate expansion)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Block 3: 128 -> 64 (Final projection to 64 channels)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # --- Attention / Feature Aggregation ---
        # Pool over H and T dimensions to get a fixed feature size (B, 64, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Linear Head ---
        # Use LazyLinear to handle the inferred input size (64 channels)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Feature Extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Shape: (B, 64, H, T)

        # 2. Global Pooling (Collapses H and T to 1x1)
        x = self.global_pool(x)
        # Shape: (B, 64, 1, 1)

        # 3. Flatten and Linear Projection
        # Flatten the last two dimensions (C, 1, 1) -> (C)
        x = x.view(x.size(0), -1)
        
        # Project to class logits
        x = self.head(x)
        return x

if __name__ == "__main__":
    print(f"--- Starting BirdCLEF Training Run ---")
    print(f"Device: {device}")
    
    # 1. Load Dataset and get metadata
    print("Loading precomputed dataset...")
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    
    print(f"Dataset loaded. Number of classes: {num_classes}")

    # 2. Setup Loss and Pos_Weight
    pos_weight = compute_pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 3. Model Initialization
    model = CnnAttentionModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    
    print("Model initialized.")

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Move batch data to device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Loss calculation
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_loss = running_loss / train_loader.batch_size
        print(f"Training Loss: {epoch_loss:.4f}")
        if device.type() == 'cuda':
             torch.cuda.empty_cache()
        # Flush output to ensure progress print is visible
        print("Training step complete.", flush=True)

    # 5. Validation/Testing Loop (Evaluation)
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n--- Starting Validation ---")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Forward pass
            logits = model(x)
            
            # Get probabilities (sigmoid applied)
            probs = torch.sigmoid(logits)
            
            # Move predictions back to CPU and append
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            print(f"Validation batch {batch_idx+1}/{len(val_loader)} complete.", flush=True)

    # 6. Metric Calculation
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate macro-averaged ROC-AUC
    roc_auc = roc_auc_score(all_targets, all_preds, average='macro')
    # Calculate macro-averaged Average Precision Score
    ap_score = average_precision_score(all_targets, all_preds, average='macro')
    
    metrics = {
        "roc_auc_macro": roc_auc,
        "ap_score_macro": ap_score
    }
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Metrics calculated: {metrics}")

    # 7. Write results.json
    output_filename = "results.json"
    results_data = {
        "model_architecture": "cnn_attention_model",
        "hyperparameters": {
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": os.environ.get("BIRDCLEF_BATCH_SIZE", "128"),
            "device": DEVICE_NAME
        },
        "metrics": metrics
    }
    
    with open(output_filename, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"\nSuccessfully wrote results to {output_filename}", flush=True)

- **inner_retries:** 0
