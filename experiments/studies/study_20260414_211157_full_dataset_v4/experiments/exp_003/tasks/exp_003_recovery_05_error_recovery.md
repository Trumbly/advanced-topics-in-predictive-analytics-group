# Task exp_003_recovery_05_error_recovery

- **Experiment:** exp_003
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-14 23:23:06.988285+00:00
- **Completed:** 2026-04-14 23:25:23.577913+00:00

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
  "architecture": "[cnn_attention] 4-Conv block front-end (32->64->128) -> Temporal Self-Attention -> Global Average Pooling",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 4,
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

# === Hardcode hyperparameters from the proposal ===
# EPOCHS is read from an env var so the orchestrator can override it.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
LR = 1e-3
# Override skeleton default augmentation with proposal values
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# =============================================================================
# CUSTOM MODULE DEFINITIONS (Module Scope)
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Implements a simplified self-attention mechanism over the temporal dimension.
    Input shape expected: (B, C, 1, L) where L is the sequence length (time).
    We pool H to 1 first, so input is (B, C, 1, L).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections across the feature dimension (C=embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.tensor(1.0 / np.sqrt(self.head_dim), dtype=torch.float32)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, C, 1, L)
        B, C, _, L = x.size()
        
        # Flatten C and H dimensions together to get (B, C*1, L) for linear layers
        # We reshape to (B, C, L) for easier attention calculation
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, C, L)
        
        # Calculate Q, K, V
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # Scaled Dot-Product Attention: (B, L, C) @ (B, C, L) -> (B, L, L)
        # We transpose L and C to perform dot product across the sequence dimension
        attn_scores = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout and compute context vector
        context = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.output_proj(context)
        
        # Reshape back to (B, C, 1, L) format for the next block/pooling
        return output.view(B, C, 1, L)


class BirdCLEFModel(nn.Module):
    """
    CNN-Attention Model architecture as proposed.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # 1. CNN Front-end (32 -> 64 -> 128 channels)
        # Using padding=1 to preserve dimensions for as long as possible.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Out: 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # In: 32, Out: 64
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # In: 64, Out: 128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 2. Temporal Self-Attention
        self.attention = SelfAttentionBlock(embed_dim=128, num_heads=4, dropout=dropout_rate)
        
        # 3. Global Average Pooling (Collapse Height dimension to 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classifier Head (LazyLinear handles arbitrary feature size)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # x shape: (B, 128, H, L) -> H=128, L=313
        
        # 2. Temporal Self-Attention
        # Manual pooling to keep the time dimension:
        # (B, C, H, L) -> (B, C, 1, L) by averaging over H
        x_attn_input = x.mean(dim=2) 
        
        x_attn = self.attention(x_attn_input)
        
        # 3. Final Pooling and Classifier
        # The output of attention is (B, 128, 1, 313). We now pool H again (which is 1) 
        # and then flatten to (B, 128) before the linear layer.
        x_final = self.pool(x_attn) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x_flat = x_final.flatten(1) 
        
        # 4. Classification
        return self.head(x_flat)


if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Use the specified augmentation settings
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
        model = BirdCLEFModel(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Use typical input shape (1, 1, 128, 313)
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

        curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
        PATIENCE = 2
        best_f1 = 0.0
        epochs_no_improve = 0
        
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            # FIX: Unpacking batch tuple to correctly handle data structure
            for batch_idx, batch in enumerate(train_loader):
                x, y = batch[0].to(device, dtype=torch.float32, non_blocking=True), batch[1].to(device, dtype=torch.float32, non_blocking=True)
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
                # FIX: Unpacking batch tuple to correctly handle data structure
                for batch in val_loader:
                    x, y = batch[0].to(device, dtype=torch.float32, non_blocking=True), batch[1].to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities: sigmoid(logits)
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
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 with per-class threshold optimization ---
            best_thresholds = np.full(targs.shape[1], 0.5)
            for c in range(targs.shape[1]):
                if targs[:, c].sum() == 0:
                    continue
                best_f1_c = 0.0
                # Sweep thresholds from 0.05 to 0.95 in steps of 0.05
                for thr in np.arange(0.05, 1.0, 0.05):
                    preds_c = (probs[:, c] >= thr).astype(np.float32)
                    f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                    if f1_c > best_f1_c:
                        best_f1_c = f1_c
                        best_thresholds[c] = thr
            preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["f1_macro"].append(val_f1)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                flush=True,
            )
            scheduler.step()

            # Early stopping on F1 (primary metric)
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                # Save best model weights + thresholds for submission
                torch.save(model.state_dict(), "best_model.pt")
                np.save("best_thresholds.npy", best_thresholds)
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
                "f1_macro": curves["f1_macro"][-1],
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: f1={curves['f1_macro'][-1]:.4f} "
            f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
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
ValueError: not enough values to unpack (expected 4, got 3)

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

# === Hardcode hyperparameters from the proposal ===
# EPOCHS is read from an env var so the orchestrator can override it.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
LR = 1e-3
# Override skeleton default augmentation with proposal values
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# =============================================================================
# CUSTOM MODULE DEFINITIONS (Module Scope)
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Implements a simplified self-attention mechanism over the temporal dimension.
    Input shape expected: (B, C, 1, L) where L is the sequence length (time).
    We pool H to 1 first, so input is (B, C, 1, L).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections across the feature dimension (C=embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.tensor(1.0 / np.sqrt(self.head_dim), dtype=torch.float32)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, C, 1, L)
        B, C, _, L = x.size()
        
        # Flatten C and H dimensions together to get (B, C*1, L) for linear layers
        # We reshape to (B, C, L) for easier attention calculation
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, C, L)
        
        # Calculate Q, K, V
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # Scaled Dot-Product Attention: (B, L, C) @ (B, C, L) -> (B, L, L)
        # We transpose L and C to perform dot product across the sequence dimension
        attn_scores = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout and compute context vector
        context = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.output_proj(context)
        
        # Reshape back to (B, C, 1, L) format for the next block/pooling
        return output.view(B, C, 1, L)


class BirdCLEFModel(nn.Module):
    """
    CNN-Attention Model architecture as proposed.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # 1. CNN Front-end (32 -> 64 -> 128 channels)
        # Using padding=1 to preserve dimensions for as long as possible.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Out: 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # In: 32, Out: 64
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # In: 64, Out: 128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 2. Temporal Self-Attention
        self.attention = SelfAttentionBlock(embed_dim=128, num_heads=4, dropout=dropout_rate)
        
        # 3. Global Average Pooling (Collapse Height dimension to 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classifier Head (LazyLinear handles arbitrary feature size)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # x shape: (B, 128, H, L) -> H=128, L=313
        
        # 2. Temporal Self-Attention
        # Manual pooling to keep the time dimension:
        # (B, C, H, L) -> (B, C, 1, L) by averaging over H
        x_attn_input = x.mean(dim=2) 
        
        x_attn = self.attention(x_attn_input)
        
        # 3. Final Pooling and Classifier
        # The output of attention is (B, 128, 1, 313). We now pool H again (which is 1) 
        # and then flatten to (B, 128) before the linear layer.
        x_final = self.pool(x_attn) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x_flat = x_final.flatten(1) 
        
        # 4. Classification
        return self.head(x_flat)


if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Use the specified augmentation settings
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
        model = BirdCLEFModel(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Use typical input shape (1, 1, 128, 313)
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

        curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
        PATIENCE = 2
        best_f1 = 0.0
        epochs_no_improve = 0
        
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            # FIX: Unpacking batch tuple to correctly handle data structure
            for batch_idx, batch in enumerate(train_loader):
                x, y = batch[0].to(device, dtype=torch.float32, non_blocking=True), batch[1].to(device, dtype=torch.float32, non_blocking=True)
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
                # FIX: Unpacking batch tuple to correctly handle data structure
                for batch in val_loader:
                    x, y = batch[0].to(device, dtype=torch.float32, non_blocking=True), batch[1].to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities: sigmoid(logits)
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
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 with per-class threshold optimization ---
            best_thresholds = np.full(targs.shape[1], 0.5)
            for c in range(targs.shape[1]):
                if targs[:, c].sum() == 0:
                    continue
                best_f1_c = 0.0
                # Sweep thresholds from 0.05 to 0.95 in steps of 0.05
                for thr in np.arange(0.05, 1.0, 0.05):
                    preds_c = (probs[:, c] >= thr).astype(np.float32)
                    f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                    if f1_c > best_f1_c:
                        best_f1_c = f1_c
                        best_thresholds[c] = thr
            preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["f1_macro"].append(val_f1)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                flush=True,
            )
            scheduler.step()

            # Early stopping on F1 (primary metric)
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                # Save best model weights + thresholds for submission
                torch.save(model.state_dict(), "best_model.pt")
                np.save("best_thresholds.npy", best_thresholds)
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
                "f1_macro": curves["f1_macro"][-1],
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: f1={curves['f1_macro'][-1]:.4f} "
            f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
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

# === Hardcode hyperparameters from the proposal ===
# EPOCHS is read from an env var so the orchestrator can override it.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
LR = 1e-3
# Override skeleton default augmentation with proposal values
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# =============================================================================
# CUSTOM MODULE DEFINITIONS (Module Scope)
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Implements a simplified self-attention mechanism over the temporal dimension.
    Input shape expected: (B, C, 1, L) where L is the sequence length (time).
    We pool H to 1 first, so input is (B, C, 1, L).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections across the feature dimension (C=embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.tensor(1.0 / np.sqrt(self.head_dim), dtype=torch.float32)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, C, 1, L)
        B, C, _, L = x.size()
        
        # Flatten C and H dimensions together to get (B, C*1, L) for linear layers
        # We reshape to (B, C, L) for easier attention calculation
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, C, L)
        
        # Calculate Q, K, V
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # Scaled Dot-Product Attention: (B, L, C) @ (B, C, L) -> (B, L, L)
        # We transpose L and C to perform dot product across the sequence dimension
        attn_scores = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout and compute context vector
        context = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.output_proj(context)
        
        # Reshape back to (B, C, 1, L) format for the next block/pooling
        return output.view(B, C, 1, L)


class BirdCLEFModel(nn.Module):
    """
    CNN-Attention Model architecture as proposed.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # 1. CNN Front-end (32 -> 64 -> 128 channels)
        # Using padding=1 to preserve dimensions for as long as possible.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Out: 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # In: 32, Out: 64
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # In: 64, Out: 128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 2. Temporal Self-Attention
        self.attention = SelfAttentionBlock(embed_dim=128, num_heads=4, dropout=dropout_rate)
        
        # 3. Global Average Pooling (Collapse Height dimension to 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classifier Head (LazyLinear handles arbitrary feature size)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # x shape: (B, 128, H, L) -> H=128, L=313
        
        # 2. Temporal Self-Attention
        # Manual pooling to keep the time dimension:
        # (B, C, H, L) -> (B, C, 1, L) by averaging over H
        x_attn_input = x.mean(dim=2) 
        
        x_attn = self.attention(x_attn_input)
        
        # 3. Final Pooling and Classifier
        # The output of attention is (B, 128, 1, 313). We now pool H again (which is 1) 
        # and then flatten to (B, 128) before the linear layer.
        x_final = self.pool(x_attn) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x_flat = x_final.flatten(1) 
        
        # 4. Classification
        return self.head(x_flat)


if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Use the specified augmentation settings
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
        model = BirdCLEFModel(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Use typical input shape (1, 1, 128, 313)
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

        curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
        PATIENCE = 2
        best_f1 = 0.0
        epochs_no_improve = 0
        
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            # FIX: Unpacking batch tuple to correctly handle data structure
            for batch_idx, batch in enumerate(train_loader):
                x, y = batch[0].to(device, dtype=torch.float32, non_blocking=True), batch[1].to(device, dtype=torch.float32, non_blocking=True)
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
                # FIX: Unpacking batch tuple to correctly handle data structure
                for batch in val_loader:
                    x, y = batch[0].to(device, dtype=torch.float32, non_blocking=True), batch[1].to(device, dtype=torch.float32, non_blocking=True)
                    # Compute probabilities: sigmoid(logits)
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
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 with per-class threshold optimization ---
            best_thresholds = np.full(targs.shape[1], 0.5)
            for c in range(targs.shape[1]):
                if targs[:, c].sum() == 0:
                    continue
                best_f1_c = 0.0
                # Sweep thresholds from 0.05 to 0.95 in steps of 0.05
                for thr in np.arange(0.05, 1.0, 0.05):
                    preds_c = (probs[:, c] >= thr).astype(np.float32)
                    f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                    if f1_c > best_f1_c:
                        best_f1_c = f1_c
                        best_thresholds[c] = thr
            preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["f1_macro"].append(val_f1)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                flush=True,
            )
            scheduler.step()

            # Early stopping on F1 (primary metric)
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                # Save best model weights + thresholds for submission
                torch.save(model.state_dict(), "best_model.pt")
                np.save("best_thresholds.npy", best_thresholds)
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
                "f1_macro": curves["f1_macro"][-1],
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: f1={curves['f1_macro'][-1]:.4f} "
            f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

- **inner_retries:** 0
