# Task exp_001_recovery_02_error_recovery

- **Experiment:** exp_001
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-11 08:42:42.786354+00:00
- **Completed:** 2026-04-11 08:42:55.473102+00:00

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

  - `import torch` and `import torch.nn as nn` at the top
  - `torch.set_num_threads(os.cpu_count() or 4)` for CPU saturation
  - Load data via `load_precomputed_dataset` (no path arguments)
  - `EPOCHS = 1` exactly (fast-iteration mode, hard cap)
  - BCEWithLogitsLoss + macro-averaged ROC-AUC via sklearn
  - Progress prints with `flush=True` at key checkpoints
  - `results.json` written at the end with metrics OR an `error` key

## Common fixes for common errors

  - `AttributeError: 'X' object has no attribute 'relu'`:
      Either define `self.relu = nn.ReLU(inplace=True)` in __init__,
      or replace `self.relu(x)` with `F.relu(x)` (from torch.nn.functional).
  - `RuntimeError: mat1 and mat2 shapes cannot be multiplied`:
      Flatten/linear size mismatch. Fix by ending the feature stack with
      `nn.AdaptiveAvgPool2d((1, 1))` → `.flatten(1)` → `nn.Linear(C, num_classes)`
      where C is the last Conv2d's out_channels.
  - `RuntimeError: Given groups=1, weight of size [X, Y, ...] expected input
     [..., Z, ...] to have Y channels, but got Z channels instead`:
      Channel mismatch between stacked Conv2d layers. Each Conv2d's
      `in_channels` must equal the previous one's `out_channels`.
  - `NameError: name 'num_classes' is not defined`:
      num_classes must be defined at module scope via
      `train_loader, val_loader, num_classes = load_precomputed_dataset(...)`
      BEFORE the model is instantiated.
  - `AttributeError: 'Dropout' object has no attribute 'set'` or similar
    made-up methods:
      Use only documented PyTorch APIs. nn.Dropout has no `.set()`.
  - `EpochsCapExceeded: EPOCHS = N exceeds the hard cap of 1`:
      Change the literal `EPOCHS = N` line to `EPOCHS = 1`. Do not use
      `max(1, ...)` or `min(..., 1)` — just write `EPOCHS = 1`.


[USER]
## Original architecture proposal
{
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.0,
    "specaugment": false
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
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CnnSmallV1

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition ===
# Using registry model as specified in the proposal
model = CnnSmallV1(num_classes=206, in_channels=1)

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

    # Build the model here — num_classes is now defined at MODULE scope
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

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
                all_probs.append(torch.sigmoid(model(x)).numpy())
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
Type: ScriptReportedError

Message:
AttributeError: Can't pickle local object 'load_precomputed_dataset.<locals>._TorchAdapter'

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
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CnnSmallV1

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition ===
# Using registry model as specified in the proposal
model = CnnSmallV1(num_classes=206, in_channels=1)

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

    # Build the model here — num_classes is now defined at MODULE scope
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

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
                all_probs.append(torch.sigmoid(model(x)).numpy())
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
```

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CnnSmallV1

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition ===
# Using registry model as specified in the proposal
model = CnnSmallV1(num_classes=206, in_channels=1)

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

    # Build the model here — num_classes is now defined at MODULE scope
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

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
                all_probs.append(torch.sigmoid(model(x)).numpy())
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

