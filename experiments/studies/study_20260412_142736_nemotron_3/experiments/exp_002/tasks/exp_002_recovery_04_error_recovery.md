# Task exp_002_recovery_04_error_recovery

- **Experiment:** exp_002
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-12 12:33:40.321801+00:00
- **Completed:** 2026-04-12 12:34:03.183130+00:00

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
  "architecture": "[mobilenet_v3_small] lightweight Conv2d backbone (1M params) with LazyLinear head",
  "pretrained_model": "mobilenet_v3_small",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
lr = 0.001
batch_size = 128
weight_decay = 0.0
dropout = 0.1

class TorchvisionAdapter(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        backbone = getattr(models, backbone_name)
        self.backbone = backbone(pretrained=False).features
        self.classifier = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x
        if x.shape[2] == 28:
            x = T.ToTensor()(x)
            x = T.Resize((224, 224)).transform(x)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def load_precomputed_dataset():
    dataset = datasets.TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
    data_loader = datasets.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=0, prefetch_factor=None)
    pos_weight = compute_pos_weight()
    return data_loader, pos_weight

if __name__ == "__main__":
    data_loader, pos_weight = load_precomputed_dataset()
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = TorchvisionAdapter("mobilenet_v3_small", num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc = 0.0
    all_probs = []
    all_labels = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
        epoch_loss /= len(data_loader)
        print(f'Epoch {epoch+1} loss: {epoch_loss:.4f}')
        if epoch == 0:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f}')
        else:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f} (best: {best_auc:.4f})')
    results = {"macro_roc_auc": best_auc}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

```

## Error that crashed it
Type: UnknownError

Message:
NameError: name 'datasets' is not defined. Did you mean: 'dataset'?

Traceback (tail):
```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002/code.py", line 49, in <module>
    data_loader, pos_weight = load_precomputed_dataset()
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002/code.py", line 43, in load_precomputed_dataset
    dataset = datasets.TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
              ^^^^^^^^
NameError: name 'datasets' is not defined. Did you mean: 'dataset'?
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset
from pipelines.data_loader import compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
lr = 0.001
batch_size = 128
weight_decay = 0.0
dropout = 0.1

class TorchvisionAdapter(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        backbone = getattr(models, backbone_name)
        self.backbone = backbone(pretrained=False).features
        self.classifier = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x
        if x.shape[2] == 28:
            x = T.ToTensor()(x)
            x = T.Resize((224, 224)).transform(x)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def load_precomputed_dataset():
    dataset = TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=0, prefetch_factor=None)
    pos_weight = compute_pos_weight()
    return data_loader, pos_weight

if __name__ == "__main__":
    data_loader, pos_weight = load_precomputed_dataset()
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = TorchvisionAdapter("mobilenet_v3_small", num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc = 0.0
    all_probs = []
    all_labels = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
        epoch_loss /= len(data_loader)
        print(f'Epoch {epoch+1} loss: {epoch_loss:.4f}')
        if epoch == 0:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f}')
        else:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f} (best: {best_auc:.4f})')
    results = {"macro_roc_auc": best_auc}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
```

## Output
- **code:** import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset
from pipelines.data_loader import compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
lr = 0.001
batch_size = 128
weight_decay = 0.0
dropout = 0.1

class TorchvisionAdapter(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        backbone = getattr(models, backbone_name)
        self.backbone = backbone(pretrained=False).features
        self.classifier = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x
        if x.shape[2] == 28:
            x = T.ToTensor()(x)
            x = T.Resize((224, 224)).transform(x)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def load_precomputed_dataset():
    dataset = TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=0, prefetch_factor=None)
    pos_weight = compute_pos_weight()
    return data_loader, pos_weight

if __name__ == "__main__":
    data_loader, pos_weight = load_precomputed_dataset()
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = TorchvisionAdapter("mobilenet_v3_small", num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc = 0.0
    all_probs = []
    all_labels = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
        epoch_loss /= len(data_loader)
        print(f'Epoch {epoch+1} loss: {epoch_loss:.4f}')
        if epoch == 0:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f}')
        else:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f} (best: {best_auc:.4f})')
    results = {"macro_roc_auc": best_auc}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

- **inner_retries:** 0
