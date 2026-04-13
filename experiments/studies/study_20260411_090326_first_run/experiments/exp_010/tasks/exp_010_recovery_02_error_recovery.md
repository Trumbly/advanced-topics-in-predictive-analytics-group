# Task exp_010_recovery_02_error_recovery

- **Experiment:** exp_010
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-11 10:17:22.150145+00:00
- **Completed:** 2026-04-11 10:17:44.638850+00:00

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
  "architecture": "custom 3-conv CNN with self-attention and dropout",
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
    "mixup": 0.2,
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
import torch.nn

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
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

torch.set_num_threads(os.cpu_count() or 4)

train_loader, val_loader, num_classes = load_precomputed_dataset()

EPOCHS = 1

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.1)
        self.self_attn = torch.nn.Linear(128, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.self_attn(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    train_loss /= len(train_loader)
    print(f'Train loss {train_loss:.4f}', flush=True)

    model.eval()
    with torch.no_grad():
        val_outputs = model(val_loader[0])
        val_probs = F.sigmoid(val_outputs).cpu().numpy()
        val_labels = val_loader[1].numpy()
        auc = roc_auc_score(val_labels, val_probs)
    print(f'Val ROC-AUC {auc:.4f}', flush=True)

    results = {
        "train_loss": train_loss,
        "val_auc": auc,
        "epoch": epoch + 1,
        "error": None
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

print('Training finished, results.json written.', flush=True)
```

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

torch.set_num_threads(os.cpu_count() or 4)

train_loader, val_loader, num_classes = load_precomputed_dataset()

EPOCHS = 1

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.1)
        self.self_attn = torch.nn.Linear(128, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.self_attn(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    train_loss /= len(train_loader)
    print(f'Train loss {train_loss:.4f}', flush=True)

    model.eval()
    with torch.no_grad():
        val_outputs = model(val_loader[0])
        val_probs = F.sigmoid(val_outputs).cpu().numpy()
        val_labels = val_loader[1].numpy()
        auc = roc_auc_score(val_labels, val_probs)
    print(f'Val ROC-AUC {auc:.4f}', flush=True)

    results = {
        "train_loss": train_loss,
        "val_auc": auc,
        "epoch": epoch + 1,
        "error": None
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

print('Training finished, results.json written.', flush=True)

