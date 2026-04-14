# Task exp_004_recovery_04_error_recovery

- **Experiment:** exp_004
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-14 04:32:00.714532+00:00
- **Completed:** 2026-04-14 04:34:05.300270+00:00

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
  "architecture": "[deep_cnn] 4-block residual CNN backbone with BatchNorm and final attention pooling",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.0008,
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

# === Hardcoded hyperparameters from the proposal ===
# Note: LR and AUGMENTATION are updated based on the JSON proposal.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "7"))
LR = 0.0008  # Overriding skeleton default 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A standard residual block for 2D spectrogram inputs, using BatchNorm and ReLU.
    Handles channel dimension mismatch via projection if needed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. First convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # 2. Second convolution path
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # 3. Dropout and residual connection
        out = self.dropout(out)
        
        # Residual connection: must ensure shapes match for addition
        if out.shape != x.shape:
            # If shapes mismatch, we must project the input 'x' to match 'out' dimensions
            # Since we control the channel progression, the residual connection should ideally
            # happen after appropriate pooling/projection if dimensions change drastically.
            # For this specific fix, we assume the input x needs to be projected if channels mismatch.
            if out.shape[1] != x.shape[1]:
                # Use a simple projection layer for the residual connection if channels change
                projection = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=1, padding=0).to(x.device)
                x = projection(x)
                
        return out + x

class AttentionPooling(nn.Module):
    """
    Global Attention Pooling module.
    Takes (B, C, H, W) -> (B, C) by attending to spatial dimensions.
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # Global Average Pooling is the baseline feature map
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature map size after pooling: (B, C, 1, 1)
        # We concatenate this with the average feature map (B, 1, 1, 1)
        # to create a combined feature vector for attention scoring.
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. Global Average Pooling (Feature map -> (B, C, 1, 1))
        avg = self.avg_pool(x)
        
        # 2. Self-Attention mechanism (using the feature map itself)
        # We average the feature map across spatial dimensions to get the channel mean (B, C, 1, 1)
        attn_map = avg
        
        # Concatenate the two (effectively, we use avg twice, but the conv layer handles the dimension increase)
        # The output of self.attention_conv is (B, 1, 1, 1)
        attn_out = self.attention_conv(torch.cat([attn_map, attn_map], dim=1))
        
        # 3. Apply softmax-like scoring by squeezing the feature map to (B, C)
        # We use the attention output as a weight map and apply it across the channel dimension.
        # Since we are after linear pooling, we simply average the attention output.
        attention_weights = torch.sigmoid(attn_out.view(x.size(0), 1, 1, 1)).squeeze(1) # (B, 1, 1, 1) -> (B, 1)
        
        # Final weighted feature vector: (B, C, 1, 1) * (B, 1, 1, 1)
        # This scales the entire feature map by the computed attention weight.
        output = avg * attention_weights.view(1, -1, 1, 1)
        return output

class ResidualCNN(nn.Module):
    """
    Deep CNN architecture with 4 residual blocks, BN, and Attention Pooling.
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # Initial convolution layer (Input: 1 channel)
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)
        self.relu_init = nn.ReLU(inplace=True)
        
        # 4 Residual Blocks: 1->32 -> 32->64 -> 64->128 -> 128->256
        # Note: The residual connection forces the feature map size to be preserved.
        self.res1 = ResidualBlock(1, 32, dropout_rate=dropout_rate) # Input: 1, Output: 32
        self.res2 = ResidualBlock(32, 64, dropout_rate=dropout_rate) # Input: 32, Output: 64
        self.res3 = ResidualBlock(64, 128, dropout_rate=dropout_rate) # Input: 64, Output: 128
        self.res4 = ResidualBlock(128, 256, dropout_rate=dropout_rate) # Input: 128, Output: 256
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(dropout_rate=dropout_rate)
        
        # Classifier Head: Uses LazyLinear to handle dynamic feature size
        # The input size will be 256 (the output of res4)
        self.classifier_head = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # Pass through residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Attention Pooling
        x = self.attention_pool(x)
        
        # Final projection and flattening: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1) 
        
        # Classifier head
        logits = self.classifier_head(x)
        return logits


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
        # Load data using the specified augmentation config
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
        model = ResidualCNN(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Input shape: (B, 1, 128, 313)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        # Calculate and print parameter count
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup: Using specified LR and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        
        # Per-class pos_weight from the DatasetProfile
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
                    # Get logits, apply sigmoid, move to CPU, convert to numpy
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

            # --- Metric 2: cmap@5 (average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
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

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
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
RuntimeError: Given groups=1, weight of size [32, 1, 3, 3], expected input[1, 32, 128, 313] to have 1 channels, but got 32 channels instead

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

# === Hardcoded hyperparameters from the proposal ===
# Note: LR and AUGMENTATION are updated based on the JSON proposal.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "7"))
LR = 0.0008  # Overriding skeleton default 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A standard residual block for 2D spectrogram inputs, using BatchNorm and ReLU.
    Handles channel dimension mismatch via projection if needed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. First convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # 2. Second convolution path
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # 3. Dropout and residual connection
        out = self.dropout(out)
        
        # Residual connection: must ensure shapes match for addition
        # If shapes mismatch, we must project the input 'x' to match 'out' dimensions
        if out.shape != x.shape:
            # Check if channel dimensions match for the residual addition
            if out.shape[1] != x.shape[1] or out.shape[2:] != x.shape[2:]:
                # Use a simple projection layer for the residual connection if channels change
                # We use the original kernel size (3) and padding to maintain spatial dimensions
                projection = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=3, padding=1).to(x.device)
                x = projection(x)
                
        return out + x

class AttentionPooling(nn.Module):
    """
    Global Attention Pooling module.
    Takes (B, C, H, W) -> (B, C) by attending to spatial dimensions.
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # Global Average Pooling is the baseline feature map
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature map size after pooling: (B, C, 1, 1)
        # We concatenate this with the average feature map (B, 1, 1, 1)
        # to create a combined feature vector for attention scoring.
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. Global Average Pooling (Feature map -> (B, C, 1, 1))
        avg = self.avg_pool(x)
        
        # 2. Self-Attention mechanism (using the feature map itself)
        # Concatenate the two (effective: [avg, avg])
        attn_in = torch.cat([avg, avg], dim=1)
        attn_out = self.attention_conv(attn_in) # Output: (B, 1, 1, 1)
        
        # 3. Apply softmax-like scoring
        # Squeezing (B, 1, 1, 1) -> (B, 1)
        attention_weights = torch.sigmoid(attn_out).squeeze(-1).squeeze(-1) # (B, 1, 1, 1) -> (B, 1)
        
        # Final weighted feature vector: (B, C, 1, 1) * (B, 1, 1, 1)
        # Reshape weights to broadcast correctly: (B, 1) -> (B, 1, 1, 1)
        attention_weights_view = attention_weights.unsqueeze(-1).unsqueeze(-1)
        output = avg * attention_weights_view
        return output

class ResidualCNN(nn.Module):
    """
    Deep CNN architecture with 4 residual blocks, BN, and Attention Pooling.
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # Initial convolution layer (Input: 1 channel)
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)
        self.relu_init = nn.ReLU(inplace=True)
        
        # 4 Residual Blocks: 1->32 -> 32->64 -> 64->128 -> 128->256
        self.res1 = ResidualBlock(1, 32, dropout_rate=dropout_rate) # Input: 1, Output: 32
        self.res2 = ResidualBlock(32, 64, dropout_rate=dropout_rate) # Input: 32, Output: 64
        self.res3 = ResidualBlock(64, 128, dropout_rate=dropout_rate) # Input: 64, Output: 128
        self.res4 = ResidualBlock(128, 256, dropout_rate=dropout_rate) # Input: 128, Output: 256
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(dropout_rate=dropout_rate)
        
        # Classifier Head: Uses LazyLinear to handle dynamic feature size
        # The input size will be 256 (the output of res4)
        self.classifier_head = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # Pass through residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Attention Pooling
        x = self.attention_pool(x)
        
        # Final projection and flattening: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1) 
        
        # Classifier head
        logits = self.classifier_head(x)
        return logits


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
        # Load data using the specified augmentation config
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
        model = ResidualCNN(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Input shape: (B, 1, 128, 313) - Keeping these dimensions as they were in the original code.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        # Calculate and print parameter count
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup: Using specified LR and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        
        # Per-class pos_weight from the DatasetProfile
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
                    # Get logits, apply sigmoid, move to CPU, convert to numpy
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

            # --- Metric 2: cmap@5 (average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
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

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
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

# === Hardcoded hyperparameters from the proposal ===
# Note: LR and AUGMENTATION are updated based on the JSON proposal.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "7"))
LR = 0.0008  # Overriding skeleton default 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===

class ResidualBlock(nn.Module):
    """
    A standard residual block for 2D spectrogram inputs, using BatchNorm and ReLU.
    Handles channel dimension mismatch via projection if needed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. First convolution path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # 2. Second convolution path
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # 3. Dropout and residual connection
        out = self.dropout(out)
        
        # Residual connection: must ensure shapes match for addition
        # If shapes mismatch, we must project the input 'x' to match 'out' dimensions
        if out.shape != x.shape:
            # Check if channel dimensions match for the residual addition
            if out.shape[1] != x.shape[1] or out.shape[2:] != x.shape[2:]:
                # Use a simple projection layer for the residual connection if channels change
                # We use the original kernel size (3) and padding to maintain spatial dimensions
                projection = nn.Conv2d(x.shape[1], out.shape[1], kernel_size=3, padding=1).to(x.device)
                x = projection(x)
                
        return out + x

class AttentionPooling(nn.Module):
    """
    Global Attention Pooling module.
    Takes (B, C, H, W) -> (B, C) by attending to spatial dimensions.
    """
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        # Global Average Pooling is the baseline feature map
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature map size after pooling: (B, C, 1, 1)
        # We concatenate this with the average feature map (B, 1, 1, 1)
        # to create a combined feature vector for attention scoring.
        self.attention_conv = nn.Conv2d(2, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 1. Global Average Pooling (Feature map -> (B, C, 1, 1))
        avg = self.avg_pool(x)
        
        # 2. Self-Attention mechanism (using the feature map itself)
        # Concatenate the two (effective: [avg, avg])
        attn_in = torch.cat([avg, avg], dim=1)
        attn_out = self.attention_conv(attn_in) # Output: (B, 1, 1, 1)
        
        # 3. Apply softmax-like scoring
        # Squeezing (B, 1, 1, 1) -> (B, 1)
        attention_weights = torch.sigmoid(attn_out).squeeze(-1).squeeze(-1) # (B, 1, 1, 1) -> (B, 1)
        
        # Final weighted feature vector: (B, C, 1, 1) * (B, 1, 1, 1)
        # Reshape weights to broadcast correctly: (B, 1) -> (B, 1, 1, 1)
        attention_weights_view = attention_weights.unsqueeze(-1).unsqueeze(-1)
        output = avg * attention_weights_view
        return output

class ResidualCNN(nn.Module):
    """
    Deep CNN architecture with 4 residual blocks, BN, and Attention Pooling.
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # Initial convolution layer (Input: 1 channel)
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)
        self.relu_init = nn.ReLU(inplace=True)
        
        # 4 Residual Blocks: 1->32 -> 32->64 -> 64->128 -> 128->256
        self.res1 = ResidualBlock(1, 32, dropout_rate=dropout_rate) # Input: 1, Output: 32
        self.res2 = ResidualBlock(32, 64, dropout_rate=dropout_rate) # Input: 32, Output: 64
        self.res3 = ResidualBlock(64, 128, dropout_rate=dropout_rate) # Input: 64, Output: 128
        self.res4 = ResidualBlock(128, 256, dropout_rate=dropout_rate) # Input: 128, Output: 256
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(dropout_rate=dropout_rate)
        
        # Classifier Head: Uses LazyLinear to handle dynamic feature size
        # The input size will be 256 (the output of res4)
        self.classifier_head = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        x = self.bn_init(x)
        x = self.relu_init(x)
        
        # Pass through residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        # Attention Pooling
        x = self.attention_pool(x)
        
        # Final projection and flattening: (B, C, 1, 1) -> (B, C)
        x = x.view(x.size(0), -1) 
        
        # Classifier head
        logits = self.classifier_head(x)
        return logits


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
        # Load data using the specified augmentation config
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
        model = ResidualCNN(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        # Input shape: (B, 1, 128, 313) - Keeping these dimensions as they were in the original code.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        
        # Calculate and print parameter count
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        # Optimizer setup: Using specified LR and weight decay
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        
        # Per-class pos_weight from the DatasetProfile
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
                    # Get logits, apply sigmoid, move to CPU, convert to numpy
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

            # --- Metric 2: cmap@5 (average precision) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
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

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "f1_macro": curves["f1_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f} "
            f"f1={curves['f1_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

- **inner_retries:** 0
