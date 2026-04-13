# Task exp_014_recovery_05_error_recovery

- **Experiment:** exp_014
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-13 05:48:20.580339+00:00
- **Completed:** 2026-04-13 05:49:46.461502+00:00

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
  "architecture": "[cnn_gru_hybrid] cnn_small_v1 backbone output features sequence fed into 2-layer GRU(128) head",
  "pretrained_model": "cnn_small_v1",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight
from pipelines.models import CnnSmallV1

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hyperparameters ===
# Per mandate, hardcap EPOCHS to 1.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        self.backbone = CnnSmallV1()

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Temporal reshaping: (B, 128, 1, 1

```

## Error that crashed it
Type: NoResultsFile

Message:
Script exited with status 0 but did not produce a results.json file in the working directory. Make sure your code writes `results.json` at the end of training.

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

# === Hyperparameters ===
# Per mandate, hardcap EPOCHS to 1.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
# Assuming CnnSmallV1 is defined elsewhere and accepts no args or only num_classes
# We must handle the module scope dependency on num_classes correctly.
# Since we cannot change the module structure significantly, we will adapt the model to accept num_classes.

class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        # NOTE: Original code had CnnSmallV1(). We must pass num_classes if the backbone needs it, 
        # but based on the original structure, we assume it's self-contained or initialized correctly.
        # We replicate the broken code's initialization style for minimal change.
        self.backbone = CnnSmallV1() 

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. GRU Head: Process the sequence (here, sequence length 1, features = C)
        # The input sequence dimension size is derived from the backbone features (C).
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True)

        # 4. Final Linear Layer: Map GRU output features (128) to num_classes
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Backbone pass
        x = self.backbone(x)
        
        # 2. Global Pooling: (B, C, 1, 1)
        x = self.pool(x)
        
        # 3. Reshape for GRU: (B, C) -> (B, 1, C) 
        # We treat the pooled feature map C as the feature dimension, and sequence length as 1.
        c_features = x.view(x.size(0), -1) # (B, C)
        x_gru = c_features.unsqueeze(1) # (B, 1, C)
        
        # 4. GRU pass: output shape (B, 1, 128)
        # We use batch_first=True, so output is (B, SeqLen, HiddenSize)
        gru_out, _ = self.gru(x_gru)
        
        # 5. Final Linear Layer: (B, 128)
        logits = self.fc(gru_out.squeeze(1))
        
        return logits

if __name__ == "__main__":
    print("--- Starting BirdCLEF Training Script ---")
    
    # Load Dataset and calculate necessary constants
    try:
        # Dummy call to load data to get num_classes
        dummy_dataset, dummy_loader = load_precomputed_dataset()
        # Assuming num_classes is derived from the loaded dataset structure
        # We need to fetch num_classes from the loaded structure or env vars if the function doesn't return it explicitly.
        # Given the context, we assume the data loader provides the necessary size.
        # Forcing a placeholder value if the data loader doesn't expose it easily, but must be derived from the data.
        num_classes = dummy_dataset.num_classes # Assuming this attribute exists
        print(f"Successfully loaded dataset. Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}. Cannot proceed.")
        exit(1)

    # Instantiate Model and move to device
    model = CnnGruHybrid(num_classes=num_classes).to(device)
    model.train()
    
    print(f"Model initialized and moved to {device}.")
    
    # Check for uninitialized parameters in LazyLinear (Mandatory fix)
    print("Warming up model parameters...")
    with torch.no_grad():
        # Dummy input matching expected shape (Batch, Channel, Height, Width)
        dummy_input = torch.randn(1, 1, 128, 313).to(device, dtype=torch.float32)
        model(dummy_input)
    print("Model warmed up successfully.")

    # --- Training Loop ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight().to(device))
    
    print("Starting training loop...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- Training Step ---
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Using dummy_loader as the actual data loader
        for batch_idx, (data, labels) in enumerate(dummy_loader):
            # Move data to device
            x = data.to(device, dtype=torch.float32)
            y = labels.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  [Train] Average Loss: {avg_loss:.4f}")

        # --- Validation Step ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dummy_loader:
                x = data.to(device, dtype=torch.float32)
                y = labels.to(device, dtype=torch.float32)
                
                # Forward pass
                outputs = model(x)
                
                # Calculate probabilities and store
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(y.cpu().numpy())
        
        # Concatenate results
        preds = np.concatenate(all_preds, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        # Calculate metrics (Macro-averaged ROC-AUC)
        try:
            auc_macro = roc_auc_score(labels_np, preds, average='macro')
            print(f"  [Validation] Macro ROC-AUC: {auc_macro:.4f}")
        except ValueError as e:
            print(f"  [Validation] Could not calculate AUC (likely single class or zero variance): {e}")
            auc_macro = 0.0

    # --- Final Results Writing (FIX FOR MISSING FILE) ---
    print("\n--- Training Complete ---")
    results = {
        "epoch": EPOCHS,
        "average_train_loss": avg_loss,
        "macro_roc_auc": auc_macro,
        "status": "success"
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Successfully wrote results to results.json")
    
    # Mandatory flush
    torch.cuda.empty_cache()
    print("Script finished successfully.")

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

# === Hyperparameters ===
# Per mandate, hardcap EPOCHS to 1.
EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition at MODULE scope ===
# Assuming CnnSmallV1 is defined elsewhere and accepts no args or only num_classes
# We must handle the module scope dependency on num_classes correctly.
# Since we cannot change the module structure significantly, we will adapt the model to accept num_classes.

class CnnGruHybrid(nn.Module):
    """
    CNN_Small_V1 backbone output features sequence fed into 2-layer GRU(128) head.
    Input: (B, 1, 128, 313)
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. Backbone: cnn_small_v1
        # NOTE: Original code had CnnSmallV1(). We must pass num_classes if the backbone needs it, 
        # but based on the original structure, we assume it's self-contained or initialized correctly.
        # We replicate the broken code's initialization style for minimal change.
        self.backbone = CnnSmallV1() 

        # 2. Global Pooling: Collapse spatial dims (H, W) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. GRU Head: Process the sequence (here, sequence length 1, features = C)
        # The input sequence dimension size is derived from the backbone features (C).
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True)

        # 4. Final Linear Layer: Map GRU output features (128) to num_classes
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Backbone pass
        x = self.backbone(x)
        
        # 2. Global Pooling: (B, C, 1, 1)
        x = self.pool(x)
        
        # 3. Reshape for GRU: (B, C) -> (B, 1, C) 
        # We treat the pooled feature map C as the feature dimension, and sequence length as 1.
        c_features = x.view(x.size(0), -1) # (B, C)
        x_gru = c_features.unsqueeze(1) # (B, 1, C)
        
        # 4. GRU pass: output shape (B, 1, 128)
        # We use batch_first=True, so output is (B, SeqLen, HiddenSize)
        gru_out, _ = self.gru(x_gru)
        
        # 5. Final Linear Layer: (B, 128)
        logits = self.fc(gru_out.squeeze(1))
        
        return logits

if __name__ == "__main__":
    print("--- Starting BirdCLEF Training Script ---")
    
    # Load Dataset and calculate necessary constants
    try:
        # Dummy call to load data to get num_classes
        dummy_dataset, dummy_loader = load_precomputed_dataset()
        # Assuming num_classes is derived from the loaded dataset structure
        # We need to fetch num_classes from the loaded structure or env vars if the function doesn't return it explicitly.
        # Given the context, we assume the data loader provides the necessary size.
        # Forcing a placeholder value if the data loader doesn't expose it easily, but must be derived from the data.
        num_classes = dummy_dataset.num_classes # Assuming this attribute exists
        print(f"Successfully loaded dataset. Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}. Cannot proceed.")
        exit(1)

    # Instantiate Model and move to device
    model = CnnGruHybrid(num_classes=num_classes).to(device)
    model.train()
    
    print(f"Model initialized and moved to {device}.")
    
    # Check for uninitialized parameters in LazyLinear (Mandatory fix)
    print("Warming up model parameters...")
    with torch.no_grad():
        # Dummy input matching expected shape (Batch, Channel, Height, Width)
        dummy_input = torch.randn(1, 1, 128, 313).to(device, dtype=torch.float32)
        model(dummy_input)
    print("Model warmed up successfully.")

    # --- Training Loop ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight().to(device))
    
    print("Starting training loop...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- Training Step ---
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Using dummy_loader as the actual data loader
        for batch_idx, (data, labels) in enumerate(dummy_loader):
            # Move data to device
            x = data.to(device, dtype=torch.float32)
            y = labels.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)
            
            # Calculate loss
            loss = criterion(outputs, y)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"  [Train] Average Loss: {avg_loss:.4f}")

        # --- Validation Step ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dummy_loader:
                x = data.to(device, dtype=torch.float32)
                y = labels.to(device, dtype=torch.float32)
                
                # Forward pass
                outputs = model(x)
                
                # Calculate probabilities and store
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(y.cpu().numpy())
        
        # Concatenate results
        preds = np.concatenate(all_preds, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        # Calculate metrics (Macro-averaged ROC-AUC)
        try:
            auc_macro = roc_auc_score(labels_np, preds, average='macro')
            print(f"  [Validation] Macro ROC-AUC: {auc_macro:.4f}")
        except ValueError as e:
            print(f"  [Validation] Could not calculate AUC (likely single class or zero variance): {e}")
            auc_macro = 0.0

    # --- Final Results Writing (FIX FOR MISSING FILE) ---
    print("\n--- Training Complete ---")
    results = {
        "epoch": EPOCHS,
        "average_train_loss": avg_loss,
        "macro_roc_auc": auc_macro,
        "status": "success"
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Successfully wrote results to results.json")
    
    # Mandatory flush
    torch.cuda.empty_cache()
    print("Script finished successfully.")

- **inner_retries:** 0
