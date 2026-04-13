# Task exp_004_task_02_generate_code

- **Experiment:** exp_004
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:29:39.755357+00:00
- **Completed:** 2026-04-10 13:30:07.300412+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Your task is to write a single, self-contained training
script that trains and evaluates a model for the BirdCLEF 2026 multi-label task.

STRICT RULES:
  - Use PyTorch (torch + torch.nn). Explicitly `import torch` at the top.
  - CPU-only. Do NOT reference torch.cuda, .cuda(), or any GPU-specific APIs.
  - Import the fixed data loader:
      from pipelines.data_loader import load_precomputed_dataset
    Call it with all required keyword arguments:
      load_precomputed_dataset(
          profile_path="data/processed/dataset_profile.json",
          spectrograms_dir="data/processed/spectrograms",
          labels_csv="data/processed/labels.csv",
          batch_size=<from hyperparams>,
          num_workers=0,
          augmentation=<dict from the proposal>,
      )
    It returns `(train_loader, val_loader, num_classes)`.
  - Instantiate the model with the EXACT `import_snippet` from the Model Registry.
    Use the exact class name and capitalization shown there (e.g. `CnnSmallV1`,
    NOT `cnn_small_v1`). Copy the snippet verbatim — do not rewrite it.
  - Multi-label task: use `nn.BCEWithLogitsLoss()`. The model outputs raw logits;
    apply `torch.sigmoid` only for prediction / metric computation.
  - Metric: macro-averaged ROC-AUC over classes with at least one positive.
    Use `sklearn.metrics.roc_auc_score` per column and average.
  - Respect the proposed `epochs` value but cap at 10 if it is larger — the
    orchestrator enforces a wall-clock budget and very long runs waste it.
  - Write final results to `results.json` in the CURRENT directory with this schema:
      {{"metrics": {{"roc_auc_macro": <float>, "loss": <float>}},
        "training_curves": {{"loss": [...], "roc_auc_macro": [...]}},
        "duration_seconds": <float>}}
  - Do NOT import: subprocess, urllib, requests, socket, os.system
  - Do NOT download anything from the internet
  - Do NOT write outside the current directory
  - Wrap training in try/except; on failure, still write results.json with an
    `error` key containing the exception message.


[USER]
## Architecture Proposal (JSON)
{
  "architecture": "cnn_small_v1",
  "pretrained_model": "cnn_small_v1",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.0
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.0,
    "specaugment": true
  }
}

## Dataset Profile (summary)
- num_classes: 206
- num_samples: 233101
- spectrogram_shape: (1, 128, 313)
- sample_rate: 32000 Hz
- imbalance_ratio: 6854.00 (min=1, max=6854)
- split: stratified_kfold (seed=42)
- train/val: 186478/46623
- most-populated classes:
    - coffal1: 6854
    - rufnig1: 4907
    - bncfly: 4630
- least-populated classes:
    - 116570: 1
    - 23724: 1
    - 23150: 3

## Your Task
Write a complete, runnable Python training script that implements the architecture
above. Respect every rule in the system prompt — especially:
  - `import torch` at the top
  - use the EXACT class name from the registry (e.g. `CnnSmallV1`)
  - call `load_precomputed_dataset(...)` with all required keyword arguments
  - write `results.json` at the end

Return ONLY the Python code. No explanations, no markdown fences, no prose.

```

## LLM Response
```
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import json
import time

# Hyperparameters
batch_size = 64
lr = 0.001
epochs = min(30, 10)  # cap at 10
optimizer = torch.optim.Adam
weight_decay = 0.0
dropout = 0.0

# Augmentation dict
augmentation = {
    "time_shift": True,
    "noise_injection": True,
    "mixup": 0.0,
    "specaugment": True,
}

# Load dataset
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation=augmentation,
)

# Instantiate model
model = CnnSmallV1(num_classes=num_classes, dropout=dropout)

# Loss
criterion = nn.BCEWithLogitsLoss()

# Optimizer
opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training curves
loss_curves = []
auc_curves = []

# Timing
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, y in train_loader:
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float())
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_samples += x.size(0)

    loss_curves.append(total_loss / n_samples)

    # Evaluate on validation set
    model.eval()
    all_logits = []
    all_labels = []
    for x, y in val_loader:
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.float().cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Predictions
    probs = torch.sigmoid(all_logits).numpy()
    # Per-class AUC
    per_class_auc = []
    for c in range(num_classes):
        y_true = all_labels[:, c].numpy()
        y_pred = probs[:, c]
        if (y_true == 1).any():
            per_class_auc.append(roc_auc_score(y_true, y_pred))
    # Macro avg only for classes with at least one positive
    if per_class_auc:
        macro_auc = sum(per_class_auc) / len(per_class_auc)
    else:
        macro_auc = 0.0
    auc_curves.append(macro_auc)

    loss_curves[-1] = total_loss / n_samples  # update last entry
    loss = total_loss / n_samples

# Duration
duration = time.time() - start_time

# Prepare results
results = {
    "metrics": {
        "roc_auc_macro": round(macro_auc, 4),
        "loss": round(loss, 4)
    },
    "training_curves": {
        "loss": loss_curves,
        "roc_auc_macro": auc_curves
    },
    "duration_seconds": round(duration, 2)
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Output
- **code:** import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import json
import time

# Hyperparameters
batch_size = 64
lr = 0.001
epochs = min(30, 10)  # cap at 10
optimizer = torch.optim.Adam
weight_decay = 0.0
dropout = 0.0

# Augmentation dict
augmentation = {
    "time_shift": True,
    "noise_injection": True,
    "mixup": 0.0,
    "specaugment": True,
}

# Load dataset
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation=augmentation,
)

# Instantiate model
model = CnnSmallV1(num_classes=num_classes, dropout=dropout)

# Loss
criterion = nn.BCEWithLogitsLoss()

# Optimizer
opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training curves
loss_curves = []
auc_curves = []

# Timing
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, y in train_loader:
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float())
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_samples += x.size(0)

    loss_curves.append(total_loss / n_samples)

    # Evaluate on validation set
    model.eval()
    all_logits = []
    all_labels = []
    for x, y in val_loader:
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.float().cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Predictions
    probs = torch.sigmoid(all_logits).numpy()
    # Per-class AUC
    per_class_auc = []
    for c in range(num_classes):
        y_true = all_labels[:, c].numpy()
        y_pred = probs[:, c]
        if (y_true == 1).any():
            per_class_auc.append(roc_auc_score(y_true, y_pred))
    # Macro avg only for classes with at least one positive
    if per_class_auc:
        macro_auc = sum(per_class_auc) / len(per_class_auc)
    else:
        macro_auc = 0.0
    auc_curves.append(macro_auc)

    loss_curves[-1] = total_loss / n_samples  # update last entry
    loss = total_loss / n_samples

# Duration
duration = time.time() - start_time

# Prepare results
results = {
    "metrics": {
        "roc_auc_macro": round(macro_auc, 4),
        "loss": round(loss, 4)
    },
    "training_curves": {
        "loss": loss_curves,
        "roc_auc_macro": auc_curves
    },
    "duration_seconds": round(duration, 2)
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

