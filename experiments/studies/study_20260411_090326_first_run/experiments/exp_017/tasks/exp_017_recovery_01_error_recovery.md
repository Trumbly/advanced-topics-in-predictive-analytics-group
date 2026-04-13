# Task exp_017_recovery_01_error_recovery

- **Experiment:** exp_017
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-11 10:45:34.328695+00:00
- **Completed:** 2026-04-11 10:46:08.599555+00:00

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
  "architecture": "3-conv CNN with fused residual blocks and SE attention",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.001,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": false,
    "mixup": 0,
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

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Linear(64, 64)
        self.se_act = nn.Sigmoid()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x2 = self.relu2(self.conv2(x))
        x = self.pool2(x2)
        x3 = self.relu3(self.conv3(x))
        x = self.pool3(x3)
        x = x3 + x2
        x = self.se_act(self.se(x)).unsqueeze(-1).unsqueeze(-1)
        x = x * x3
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
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

        model = MyModel(num_classes=num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.A

```

## Error that crashed it
Type: SyntaxError

Message:
Line 69: expected 'except' or 'finally' block

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
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Linear(64, 64)
        self.se_act = nn.Sigmoid()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x2 = self.relu2(self.conv2(x))
        x = self.pool2(x2)
        x3 = self.relu3(self.conv3(x))
        x = self.pool3(x3)
        x = x3 + x2
        x = self.se_act(self.se(x)).unsqueeze(-1).unsqueeze(-1)
        x = x * x3
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
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

        model = MyModel(num_classes=num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        for epoch in range(EPOCHS):
            model.train()
            for step, (imgs, y) in enumerate(train_loader):
                imgs = imgs.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = loss_fn(logits, y)
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}", flush=True)

            val_logits = [model(x).cpu().numpy() for x in val_loader]
            val_labels = [y.numpy() for _, y in val_loader]
            val_pred = (val_logits > 0.5).astype(int).flatten()
            val_true = val_labels.flatten()
            val_auc = roc_auc_score(val_true, val_logits.ravel())
            results['val_auc'] = val_auc
            print(f"val_auc: {val_auc:.4f}", flush=True)

        # Save results
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        results['error'] = str(e)
        print("error: failed, details:", results['error'], flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"training completed in {time.time() - start:.2f}s", flush=True)
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

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Linear(64, 64)
        self.se_act = nn.Sigmoid()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x2 = self.relu2(self.conv2(x))
        x = self.pool2(x2)
        x3 = self.relu3(self.conv3(x))
        x = self.pool3(x3)
        x = x3 + x2
        x = self.se_act(self.se(x)).unsqueeze(-1).unsqueeze(-1)
        x = x * x3
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
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

        model = MyModel(num_classes=num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        for epoch in range(EPOCHS):
            model.train()
            for step, (imgs, y) in enumerate(train_loader):
                imgs = imgs.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = loss_fn(logits, y)
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}", flush=True)

            val_logits = [model(x).cpu().numpy() for x in val_loader]
            val_labels = [y.numpy() for _, y in val_loader]
            val_pred = (val_logits > 0.5).astype(int).flatten()
            val_true = val_labels.flatten()
            val_auc = roc_auc_score(val_true, val_logits.ravel())
            results['val_auc'] = val_auc
            print(f"val_auc: {val_auc:.4f}", flush=True)

        # Save results
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        results['error'] = str(e)
        print("error: failed, details:", results['error'], flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"training completed in {time.time() - start:.2f}s", flush=True)

