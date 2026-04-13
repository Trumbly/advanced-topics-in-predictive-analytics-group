# Task exp_002_task_02_generate_code

- **Experiment:** exp_002
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:34:30.879611+00:00
- **Completed:** 2026-04-10 13:35:08.327866+00:00

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
  "architecture": "3-conv2d backbone with global average pooling, dropout, and linear head",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": false,
    "mixup": 0.0,
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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
import json
import time
from pipelines.data_loader import load_precomputed_dataset

# architecture definition
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(128, num_classes)  # 234 classes as per instruction

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# hyperparams
hyperparams = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
}
epochs = min(10, hyperparams["epochs"])  # cap at 10

# augmentation dict (all false, so ignore)

train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=hyperparams["batch_size"],
    num_workers=0,
    augmentation=hyperparams["augmentation"]
)

model = MyModel(num_classes=234)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

start = time.time()
train_losses = []
roc_aucs = []
macro_auc = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for spec, labels in train_loader:
        labels = labels.float()
        optimizer.zero_grad()
        logits = model(spec)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for spec, labels in val_loader:
            logits = model(spec)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    classes = np.unique(all_labels)
    valid_classes = classes[np.any(all_labels[:, np.newaxis, :] > 0.5, axis=1)]
    if len(valid_classes) > 0:
        roc_scores = [sklearn.metrics.roc_auc_score(all_labels[valid_classes], all_preds[valid_classes], multi_class='ovr', average='macro')]
        macro_auc = np.mean(roc_scores)
    else:
        macro_auc = 0.0
    roc_aucs.append(macro_auc)

    if epoch == epochs - 1:
        macro_auc = np.mean(roc_aucs)

# final metrics
final_loss = train_losses[-1]
final_roc_auc = macro_auc

# duration
duration_seconds = time.time() - start

# results
results = {
    "metrics": {"roc_auc_macro": float(final_roc_auc), "loss": float(final_loss)},
    "training_curves": {"loss": train_losses, "roc_auc_macro": roc_aucs},
    "duration_seconds": float(duration_seconds)
}

try:
    open("results.json", "w").write(json.dumps(results, indent=0))
except:
    pass
```

## Output
- **code:** import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
import json
import time
from pipelines.data_loader import load_precomputed_dataset

# architecture definition
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(128, num_classes)  # 234 classes as per instruction

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# hyperparams
hyperparams = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
}
epochs = min(10, hyperparams["epochs"])  # cap at 10

# augmentation dict (all false, so ignore)

train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=hyperparams["batch_size"],
    num_workers=0,
    augmentation=hyperparams["augmentation"]
)

model = MyModel(num_classes=234)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

start = time.time()
train_losses = []
roc_aucs = []
macro_auc = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for spec, labels in train_loader:
        labels = labels.float()
        optimizer.zero_grad()
        logits = model(spec)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for spec, labels in val_loader:
            logits = model(spec)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    classes = np.unique(all_labels)
    valid_classes = classes[np.any(all_labels[:, np.newaxis, :] > 0.5, axis=1)]
    if len(valid_classes) > 0:
        roc_scores = [sklearn.metrics.roc_auc_score(all_labels[valid_classes], all_preds[valid_classes], multi_class='ovr', average='macro')]
        macro_auc = np.mean(roc_scores)
    else:
        macro_auc = 0.0
    roc_aucs.append(macro_auc)

    if epoch == epochs - 1:
        macro_auc = np.mean(roc_aucs)

# final metrics
final_loss = train_losses[-1]
final_roc_auc = macro_auc

# duration
duration_seconds = time.time() - start

# results
results = {
    "metrics": {"roc_auc_macro": float(final_roc_auc), "loss": float(final_loss)},
    "training_curves": {"loss": train_losses, "roc_auc_macro": roc_aucs},
    "duration_seconds": float(duration_seconds)
}

try:
    open("results.json", "w").write(json.dumps(results, indent=0))
except:
    pass

