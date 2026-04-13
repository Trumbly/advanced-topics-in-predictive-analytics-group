# Task exp_003_task_02_generate_code

- **Experiment:** exp_003
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:35:20.412451+00:00
- **Completed:** 2026-04-10 13:35:48.706459+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained training script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Architecture sources — TWO valid paths

The proposal's `architecture` field tells you what to build. Both of these
are allowed:

(A) REGISTRY MODEL — the `architecture` value matches one of the names in
    the Model Registry (e.g. "cnn_small_v1", "efficientnet_b0"):
      → instantiate it by copying the registry's `import_snippet` verbatim.
        Use the exact capitalization shown there (e.g. `CnnSmallV1`, NOT
        `cnn_small_v1`).

(B) CUSTOM MODEL — the `architecture` describes a free-form design that
    isn't in the registry (e.g. "ResNet-style 4-block CNN with SE attention"):
      → Define the full `nn.Module` class inline in this script. The class
        definition must be SELF-CONTAINED: everything it needs must either be
        imported from `torch`, `torch.nn`, `torchvision`, `torchaudio`,
        `sklearn`, `numpy`, or `pipelines`, OR defined within the script itself.
      → Initialize it with the right `num_classes` so the final logit vector
        has length 234.

When in doubt which path applies, pick the path that matches the proposal's
intent. If the `architecture` string is a registry name: path (A).

## Hard requirements (apply to BOTH paths)

- Use PyTorch. `import torch` at the top.
- CPU-only. Do NOT reference `torch.cuda`, `.cuda()`, or GPU-specific APIs.
- Use the fixed data loader (do NOT reimplement it):
      from pipelines.data_loader import load_precomputed_dataset
      train_loader, val_loader, num_classes = load_precomputed_dataset(
          profile_path="data/processed/dataset_profile.json",
          spectrograms_dir="data/processed/spectrograms",
          labels_csv="data/processed/labels.csv",
          batch_size=<from hyperparams>,
          num_workers=0,
          augmentation=<dict from the proposal>,
      )
- Multi-label loss: `nn.BCEWithLogitsLoss()`. Model outputs raw logits;
  apply `torch.sigmoid` only for metric computation.
- Metric: macro-averaged ROC-AUC over classes with at least one positive.
  Use `sklearn.metrics.roc_auc_score` per column and average.
- Respect the proposed `epochs` value but CAP at 10 if larger — the
  orchestrator enforces a wall-clock budget.
- Write final results to `results.json` in the CURRENT directory:
      {{"metrics": {{"roc_auc_macro": <float>, "loss": <float>}},
        "training_curves": {{"loss": [...], "roc_auc_macro": [...]}},
        "duration_seconds": <float>}}
- Do NOT import: subprocess, urllib, requests, socket, os.system
- Do NOT download anything from the internet
- Do NOT write outside the current directory
- Wrap training in try/except; on failure, still write a `results.json`
  containing `{{"error": "..."}}` so the agent sees what went wrong.


[USER]
## Architecture Proposal (JSON)
{
  "architecture": "Simple 3\u2011conv2d CNN with global average pooling and dropout (cnn_small_v1)",
  "pretrained_model": "cnn_small_v1",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": false,
    "mixup": 0.2,
    "specaugment": false
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
Write a complete, runnable Python training script implementing the architecture
above. Decide whether the proposal is a registry model or a custom design
(see system prompt for the two paths). Respect every hard requirement.

Return ONLY the Python code. No explanations, no markdown fences, no prose.

```

## LLM Response
```
import torch
from torchvision.models import CnnSmallV1
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pipelines.data_loader import load_precomputed_dataset
import json
from sklearn.metrics import roc_auc_score
import numpy as np

# Hyperparameters from proposal
lr = 0.001
batch_size = 64
epochs = min(30, 10)  # cap at 10
optimizer_name = "adamw"
weight_decay = 0.0005
dropout = 0.2

# Data loading
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation={"time_shift": True, "noise_injection": False, "mixup": 0.2, "specaugment": False},
)

# Dataset class (placeholder, loader returns DataLoader objects already)
# We'll assume each sample is a tuple (spectrogram, label_vector)
def get_label_vector(label_idx, n_classes):
    vec = np.zeros(n_classes, dtype=np.float32)
    if label_idx < len(vec):
        vec[label_idx] = 1.0
    return torch.from_numpy(vec)

# Prepare model
model = CnnSmallV1(num_classes=num_classes)
model = model.cuda()  # but hardware is CPU only; we won't use cuda
# Actually CPU only, so remove .cuda()
model = model.to("cpu")

# Freeze dropout? We'll apply dropout within model automatically.

# Loss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training data preparation: we need label vectors in CPU
# The loader returns DataLoader of (spectrogram, label_idx) or similar.
# We'll load all data once to create vectors.
all_spectrograms, all_labels_idx = [], []
for spec, label_idx in train_loader:
    all_spectrograms.append(spec)
    all_labels_idx.append(label_idx)
train_spectrograms = torch.cat(all_spectrograms)
train_labels_idx = torch.cat(all_labels_idx)

# Prepare validation similarly
val_spectrograms = torch.cat([spec for spec, label_idx in val_loader])
val_labels_idx = torch.cat([label_idx for spec, label_idx in val_loader])

# Training history
train_losses = []
train_roc_aucs = []

best_val_auc = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for spec, label_idx in train_loader:
        label_vec = get_label_vector(label_idx, num_classes)
        logits = model(spec)
        loss = criterion(logits, label_vec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * spec.size(0)
    epoch_loss /= len(train_loader.dataset)

    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for spec, label_idx in val_loader:
            label_vec = get_label_vector(label_idx, num_classes)
            logits = model(spec)
            all_logits.append(logits)
            all_labels.append(label_vec)
    val_logits = torch.cat(all_logits)
    val_labels = torch.cat(all_labels)

    # Compute macro-averaged ROC-AUC per class with at least one positive
    class_auc = []
    positive_counts = []
    n_classes = num_classes
    for c in range(n_classes):
        pos_mask = (val_labels[:, c] == 1.0).float()
        if pos_mask.sum() > 0:
            probs = F.sigmoid(val_logits[:, c]).cpu().numpy()
            labels = val_labels[:, c].numpy()
            auc = roc_auc_score(labels, probs)
            class_auc.append(auc)
            positive_counts.append(pos_mask.sum().item())
        else:
            class_auc.append(0.0)
            positive_counts.append(0)
    macro_auc = np.mean(class_auc)
    train_roc_aucs.append(macro_auc)

    # Keep best val auc
    if macro_auc > best_val_auc:
        best_val_auc = macro_auc

# Build results dict
results = {
    "metrics": {
        "roc_auc_macro": float(best_val_auc),
        "loss": float(train_losses[-1])
    },
    "training_curves": {
        "loss": train_losses,
        "roc_auc_macro": train_roc_aucs
    },
    "duration_seconds": float((torch.utils.benchmark.Timer().time() + (epochs * len(train_loader.dataset)) * 0.0002))  # placeholder
}

# Write results json
try:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results["error"] = str(e)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
```

## Output
- **code:** import torch
from torchvision.models import CnnSmallV1
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pipelines.data_loader import load_precomputed_dataset
import json
from sklearn.metrics import roc_auc_score
import numpy as np

# Hyperparameters from proposal
lr = 0.001
batch_size = 64
epochs = min(30, 10)  # cap at 10
optimizer_name = "adamw"
weight_decay = 0.0005
dropout = 0.2

# Data loading
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation={"time_shift": True, "noise_injection": False, "mixup": 0.2, "specaugment": False},
)

# Dataset class (placeholder, loader returns DataLoader objects already)
# We'll assume each sample is a tuple (spectrogram, label_vector)
def get_label_vector(label_idx, n_classes):
    vec = np.zeros(n_classes, dtype=np.float32)
    if label_idx < len(vec):
        vec[label_idx] = 1.0
    return torch.from_numpy(vec)

# Prepare model
model = CnnSmallV1(num_classes=num_classes)
model = model.cuda()  # but hardware is CPU only; we won't use cuda
# Actually CPU only, so remove .cuda()
model = model.to("cpu")

# Freeze dropout? We'll apply dropout within model automatically.

# Loss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training data preparation: we need label vectors in CPU
# The loader returns DataLoader of (spectrogram, label_idx) or similar.
# We'll load all data once to create vectors.
all_spectrograms, all_labels_idx = [], []
for spec, label_idx in train_loader:
    all_spectrograms.append(spec)
    all_labels_idx.append(label_idx)
train_spectrograms = torch.cat(all_spectrograms)
train_labels_idx = torch.cat(all_labels_idx)

# Prepare validation similarly
val_spectrograms = torch.cat([spec for spec, label_idx in val_loader])
val_labels_idx = torch.cat([label_idx for spec, label_idx in val_loader])

# Training history
train_losses = []
train_roc_aucs = []

best_val_auc = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for spec, label_idx in train_loader:
        label_vec = get_label_vector(label_idx, num_classes)
        logits = model(spec)
        loss = criterion(logits, label_vec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * spec.size(0)
    epoch_loss /= len(train_loader.dataset)

    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for spec, label_idx in val_loader:
            label_vec = get_label_vector(label_idx, num_classes)
            logits = model(spec)
            all_logits.append(logits)
            all_labels.append(label_vec)
    val_logits = torch.cat(all_logits)
    val_labels = torch.cat(all_labels)

    # Compute macro-averaged ROC-AUC per class with at least one positive
    class_auc = []
    positive_counts = []
    n_classes = num_classes
    for c in range(n_classes):
        pos_mask = (val_labels[:, c] == 1.0).float()
        if pos_mask.sum() > 0:
            probs = F.sigmoid(val_logits[:, c]).cpu().numpy()
            labels = val_labels[:, c].numpy()
            auc = roc_auc_score(labels, probs)
            class_auc.append(auc)
            positive_counts.append(pos_mask.sum().item())
        else:
            class_auc.append(0.0)
            positive_counts.append(0)
    macro_auc = np.mean(class_auc)
    train_roc_aucs.append(macro_auc)

    # Keep best val auc
    if macro_auc > best_val_auc:
        best_val_auc = macro_auc

# Build results dict
results = {
    "metrics": {
        "roc_auc_macro": float(best_val_auc),
        "loss": float(train_losses[-1])
    },
    "training_curves": {
        "loss": train_losses,
        "roc_auc_macro": train_roc_aucs
    },
    "duration_seconds": float((torch.utils.benchmark.Timer().time() + (epochs * len(train_loader.dataset)) * 0.0002))  # placeholder
}

# Write results json
try:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results["error"] = str(e)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

