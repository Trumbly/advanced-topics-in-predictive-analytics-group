# Task exp_005_task_02_generate_code

- **Experiment:** exp_005
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:36:57.230405+00:00
- **Completed:** 2026-04-10 13:37:51.437326+00:00

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
  "architecture": "2\u2011conv small CNN with global average pooling, no pretrained weights",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.0001,
    "batch_size": 128,
    "epochs": 50,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
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
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
import time

try:
    # Load data
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        profile_path="data/processed/dataset_profile.json",
        spectrograms_dir="data/processed/spectrograms",
        labels_csv="data/processed/labels.csv",
        batch_size=128,
        num_workers=0,
        augmentation={"time_shift": True, "noise_injection": True, "mixup": 0.0, "specaugment": False},
    )
    epochs = min(50, 10)  # cap at 10
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train curves
    train_losses = []
    train_roc_aus = []

    # Warm up optimizer? Not needed.

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(spec)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_positive_counts = np.zeros(num_classes, dtype=int)
        with torch.no_grad():
            for spec, labels in val_loader:
                logits = model(spec)
                val_loss += criterion(logits, labels.float()).item() * spec.size(0)
                preds = torch.sigmoid(logits)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                pos_counts = np.sum(labels, axis=0)
                val_positive_counts += pos_counts

        val_loss = val_loss / n_batches

        # ROC AUC per class with at least one positive
        class_aus = []
        true_labels = np.concatenate(val_labels)
        pred_labels = np.concatenate(val_preds)
        for i in range(num_classes):
            if val_positive_counts[i] > 0:
                class_aus.append(roc_auc_score(true_labels[i], pred_labels[i]))
        train_roc_aus.append(np.mean(class_aus))

        optimizer.step()
        scheduler.step()

    duration = time.time() - start_time
    results = {
        "metrics": {"roc_auc_macro": train_roc_aus[-1], "loss": train_losses[-1]},
        "training_curves": {"loss": train_losses, "roc_auc_macro": train_roc_aus},
        "duration_seconds": duration,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

# Define custom model
class Model(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate model with correct number of classes
model = Model(num_classes=num_classes)

# Continue training as above (already defined)

# End of script
```

## Output
- **code:** import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
import time

try:
    # Load data
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        profile_path="data/processed/dataset_profile.json",
        spectrograms_dir="data/processed/spectrograms",
        labels_csv="data/processed/labels.csv",
        batch_size=128,
        num_workers=0,
        augmentation={"time_shift": True, "noise_injection": True, "mixup": 0.0, "specaugment": False},
    )
    epochs = min(50, 10)  # cap at 10
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train curves
    train_losses = []
    train_roc_aus = []

    # Warm up optimizer? Not needed.

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(spec)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_positive_counts = np.zeros(num_classes, dtype=int)
        with torch.no_grad():
            for spec, labels in val_loader:
                logits = model(spec)
                val_loss += criterion(logits, labels.float()).item() * spec.size(0)
                preds = torch.sigmoid(logits)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                pos_counts = np.sum(labels, axis=0)
                val_positive_counts += pos_counts

        val_loss = val_loss / n_batches

        # ROC AUC per class with at least one positive
        class_aus = []
        true_labels = np.concatenate(val_labels)
        pred_labels = np.concatenate(val_preds)
        for i in range(num_classes):
            if val_positive_counts[i] > 0:
                class_aus.append(roc_auc_score(true_labels[i], pred_labels[i]))
        train_roc_aus.append(np.mean(class_aus))

        optimizer.step()
        scheduler.step()

    duration = time.time() - start_time
    results = {
        "metrics": {"roc_auc_macro": train_roc_aus[-1], "loss": train_losses[-1]},
        "training_curves": {"loss": train_losses, "roc_auc_macro": train_roc_aus},
        "duration_seconds": duration,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

# Define custom model
class Model(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate model with correct number of classes
model = Model(num_classes=num_classes)

# Continue training as above (already defined)

# End of script

