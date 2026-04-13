# Task exp_003_recovery_02_error_recovery

- **Experiment:** exp_003
- **Type:** llm
- **Name:** error_recovery
- **Status:** completed
- **Started:** 2026-04-11 08:45:20.831369+00:00
- **Completed:** 2026-04-11 08:45:44.885025+00:00

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
  "architecture": "Lightweight 3-Block CNN with Self-Attention and Dynamic Filtering",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0001,
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
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True, "specaugment": True}

# === Model definition ===
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=1)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return self.gamma * out + x

class DynamicFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = SelfAttention(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.attention(x)
        return x

class Lightweight3BlockCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(64)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(256)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(1024)
        )
        
        self.dynamic_filter = DynamicFilter(1024, 512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dynamic_filter(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

# Load data before model creation to avoid pickling issues
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

# Build the model here — num_classes is now defined at MODULE scope
model = Lightweight3BlockCNN(num_classes=num_classes, dropout=0.2)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

start = time.time()
results = {}
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
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
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True, "specaugment": True}

# === Model definition ===
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=1)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return self.gamma * out + x

class DynamicFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = SelfAttention(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.attention(x)
        return x

class Lightweight3BlockCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(64)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(256)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(1024)
        )
        
        self.dynamic_filter = DynamicFilter(1024, 512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dynamic_filter(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

# Load data before model creation to avoid pickling issues (moved to top level)
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

# Build the model here — num_classes is now defined at MODULE scope
model = Lightweight3BlockCNN(num_classes=num_classes, dropout=0.2)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

start = time.time()
results = {}
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
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

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True, "specaugment": True}

# === Model definition ===
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = F.softmax(torch.bmm(query, key), dim=1)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return self.gamma * out + x

class DynamicFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attention = SelfAttention(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.attention(x)
        return x

class Lightweight3BlockCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(64)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(256)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            SelfAttention(1024)
        )
        
        self.dynamic_filter = DynamicFilter(1024, 512)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dynamic_filter(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

# Load data before model creation to avoid pickling issues (moved to top level)
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

# Build the model here — num_classes is now defined at MODULE scope
model = Lightweight3BlockCNN(num_classes=num_classes, dropout=0.2)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

start = time.time()
results = {}
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
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

