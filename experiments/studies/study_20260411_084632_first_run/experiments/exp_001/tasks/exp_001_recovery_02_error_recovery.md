# Task exp_001_recovery_02_error_recovery

- **Experiment:** exp_001
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-11 08:47:24.394645+00:00
- **Completed:** 2026-04-11 08:48:05.346876+00:00

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

torch.set_num_threads(os.cpu_count() or 4)

class CnnSmallV1(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(nn.AdaptiveAvgPool2d((8, 8))(self.conv2(self.relu(self.conv1(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

print("loading data...", flush=True)
train_loader, val_loader, num_classes = load_precomputed_dataset(
    augmentation=AUGMENTATION,
)
print(
    f"data loaded: {num_classes} classes, "
    f"{len(train_loader.dataset)} train samples, "
    f"{len(val_loader.dataset)} val samples",
    flush=True,
)

model = CnnSmallV1(num_classes=num_classes, in_channels=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train_batches = len(train_loader)
log_every = max(1, n_train_batches // 10)
curves = {"loss": [], "roc_auc_macro": []}
start = time.time()

for epoch in range(EPOCHS):
    print(f"epoch {epoch + 1}/{EPOCHS} starting ({n_train_batches} batches)...", flush=True)
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

    aucs = []
    for c in range(targs.shape[1]):
        if targs[:, c].sum() > 0:
            aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
    val_auc = float(np.mean(aucs)) if aucs else 0.0
    epoch_loss = float(np.mean(epoch_losses))

    curves["loss"].append(epoch_loss)
    curves["roc_auc_macro"].append(val_auc)
    print(f"epoch {epoch + 1}/{EPOCHS} done: loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}", flush=True)

results = {
    "metrics": {
        "roc_auc_macro": curves["roc_auc_macro"][-1],
        "loss": curves["loss"][-1],
    },
    "training_curves": curves,
    "duration_seconds": time.time() - start,
}
print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Error that crashed it
Type: UnknownError

Message:
AttributeError: Can't pickle local object 'load_precomputed_dataset.<locals>._TorchAdapter'

Traceback (tail):
```
up/sandbox/study_20260411_084632_first_run/exp_001/code.py", line 60, in <module>
    for batch_idx, (x, y) in enumerate(train_loader):
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 495, in __iter__
    self._iterator = self._get_iterator()
                     ^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/context.py", line 288, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't pickle local object 'load_precomputed_dataset.<locals>._TorchAdapter'
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
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

torch.set_num_threads(os.cpu_count() or 4)

class CnnSmallV1(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(nn.AdaptiveAvgPool2d((8, 8))(self.conv2(self.relu(self.conv1(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

print("loading data...", flush=True)
train_loader, val_loader, num_classes = load_precomputed_dataset(
    augmentation=AUGMENTATION,
)
print(
    f"data loaded: {num_classes} classes, "
    f"{len(train_loader.dataset)} train samples, "
    f"{len(val_loader.dataset)} val samples",
    flush=True,
)

model = CnnSmallV1(num_classes=num_classes, in_channels=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train_batches = len(train_loader)
log_every = max(1, n_train_batches // 10)
curves = {"loss": [], "roc_auc_macro": []}
start = time.time()

for epoch in range(EPOCHS):
    print(f"epoch {epoch + 1}/{EPOCHS} starting ({n_train_batches} batches)...", flush=True)
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

    aucs = []
    for c in range(targs.shape[1]):
        if targs[:, c].sum() > 0:
            aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
    val_auc = float(np.mean(aucs)) if aucs else 0.0
    epoch_loss = float(np.mean(epoch_losses))

    curves["loss"].append(epoch_loss)
    curves["roc_auc_macro"].append(val_auc)
    print(f"epoch {epoch + 1}/{EPOCHS} done: loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}", flush=True)

results = {
    "metrics": {
        "roc_auc_macro": curves["roc_auc_macro"][-1],
        "loss": curves["loss"][-1],
    },
    "training_curves": curves,
    "duration_seconds": time.time() - start,
}
print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)

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
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

torch.set_num_threads(os.cpu_count() or 4)

class CnnSmallV1(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(nn.AdaptiveAvgPool2d((8, 8))(self.conv2(self.relu(self.conv1(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

print("loading data...", flush=True)
train_loader, val_loader, num_classes = load_precomputed_dataset(
    augmentation=AUGMENTATION,
)
print(
    f"data loaded: {num_classes} classes, "
    f"{len(train_loader.dataset)} train samples, "
    f"{len(val_loader.dataset)} val samples",
    flush=True,
)

model = CnnSmallV1(num_classes=num_classes, in_channels=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train_batches = len(train_loader)
log_every = max(1, n_train_batches // 10)
curves = {"loss": [], "roc_auc_macro": []}
start = time.time()

for epoch in range(EPOCHS):
    print(f"epoch {epoch + 1}/{EPOCHS} starting ({n_train_batches} batches)...", flush=True)
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

    aucs = []
    for c in range(targs.shape[1]):
        if targs[:, c].sum() > 0:
            aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
    val_auc = float(np.mean(aucs)) if aucs else 0.0
    epoch_loss = float(np.mean(epoch_losses))

    curves["loss"].append(epoch_loss)
    curves["roc_auc_macro"].append(val_auc)
    print(f"epoch {epoch + 1}/{EPOCHS} done: loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}", flush=True)

results = {
    "metrics": {
        "roc_auc_macro": curves["roc_auc_macro"][-1],
        "loss": curves["loss"][-1],
    },
    "training_curves": curves,
    "duration_seconds": time.time() - start,
}
print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)

with open("results.json", "w") as fh:
    json.dump(results, fh)

