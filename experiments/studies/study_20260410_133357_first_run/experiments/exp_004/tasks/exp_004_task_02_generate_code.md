# Task exp_004_task_02_generate_code

- **Experiment:** exp_004
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:36:00.565327+00:00
- **Completed:** 2026-04-10 13:36:35.685360+00:00

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
  "architecture": "3-convolutional baseline with global average pooling and dropout",
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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics
import json
from pipelines.data_loader import load_precomputed_dataset

hyperparams = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
}
lr = hyperparams["lr"]
batch_size = hyperparams["batch_size"]
epochs = min(hyperparams["epochs"], 10)  # cap at 10

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    logits_list = []
    labels_list = []
    for batch_idx, (spectrograms, labels) in enumerate(loader):
        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * spectrograms.size(0)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.cpu())
    return epoch_loss / len(loader.dataset), logits_list, labels_list

def evaluate_model(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    for batch_idx, (spectrograms, labels) in enumerate(loader):
        with torch.no_grad():
            logits = model(spectrograms)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    prob = torch.sigmoid(all_logits)
    labels_f = all_labels.float()
    num_classes = all_labels.shape[1]
    num_positive = 0
    roc_scores = []
    for i in range(num_classes):
        col = labels_f[:, i]
        pos = col > 0.5
        if pos.any():
            roc = sklearn.metrics.roc_auc_score(col, prob[:, i], pos=pos)
            roc_scores.append(roc)
            num_positive += 1
    macro_auc = sum(roc_scores) / num_positive if num_positive > 0 else 0.0
    return macro_auc

def main():
    try:
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            profile_path="data/processed/dataset_profile.json",
            spectrograms_dir="data/processed/spectrograms",
            labels_csv="data/processed/labels.csv",
            batch_size=hyperparams["batch_size"],
            num_workers=0,
            augmentation={"time_shift": False, "noise_injection": True, "mixup": 0.0, "specaugment": False}
        )
        loss_fn = nn.BCEWithLogitsLoss()
        model = ThreeConvBaseline(num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=hyperparams["weight_decay"])
        train_losses = []
        train_aucs = []
        dev_losses = []
        dev_aucs = []
        start = torch.utils.benchmark.benchmark.Timer()
        for epoch in range(1, epochs + 1):
            train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, None)
            train_losses.append(train_loss)
            train_logits, train_labels = train_logits_list, train_labels_list  # placeholder
            train_auc = evaluate_model(model, train_loader)
            train_aucs.append(train_auc)

            dev_loss, dev_logits, dev_labels = evaluate_model(model, val_loader)
            dev_losses.append(dev_loss)
            dev_aucs.append(dev_auc)

        duration = float(start.Timer().Time())
        results = {
            "metrics": {
                "roc_auc_macro": dev_aucs[-1],
                "loss": dev_losses[-1]
            },
            "training_curves": {
                "loss": train_losses,
                "roc_auc_macro": train_aucs
            },
            "duration_seconds": duration
        }
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        results = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

class ThreeConvBaseline(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=hyperparams["dropout"])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
```

## Output
- **code:** import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics
import json
from pipelines.data_loader import load_precomputed_dataset

hyperparams = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
}
lr = hyperparams["lr"]
batch_size = hyperparams["batch_size"]
epochs = min(hyperparams["epochs"], 10)  # cap at 10

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    logits_list = []
    labels_list = []
    for batch_idx, (spectrograms, labels) in enumerate(loader):
        optimizer.zero_grad()
        logits = model(spectrograms)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * spectrograms.size(0)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.cpu())
    return epoch_loss / len(loader.dataset), logits_list, labels_list

def evaluate_model(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    for batch_idx, (spectrograms, labels) in enumerate(loader):
        with torch.no_grad():
            logits = model(spectrograms)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    prob = torch.sigmoid(all_logits)
    labels_f = all_labels.float()
    num_classes = all_labels.shape[1]
    num_positive = 0
    roc_scores = []
    for i in range(num_classes):
        col = labels_f[:, i]
        pos = col > 0.5
        if pos.any():
            roc = sklearn.metrics.roc_auc_score(col, prob[:, i], pos=pos)
            roc_scores.append(roc)
            num_positive += 1
    macro_auc = sum(roc_scores) / num_positive if num_positive > 0 else 0.0
    return macro_auc

def main():
    try:
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            profile_path="data/processed/dataset_profile.json",
            spectrograms_dir="data/processed/spectrograms",
            labels_csv="data/processed/labels.csv",
            batch_size=hyperparams["batch_size"],
            num_workers=0,
            augmentation={"time_shift": False, "noise_injection": True, "mixup": 0.0, "specaugment": False}
        )
        loss_fn = nn.BCEWithLogitsLoss()
        model = ThreeConvBaseline(num_classes)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=hyperparams["weight_decay"])
        train_losses = []
        train_aucs = []
        dev_losses = []
        dev_aucs = []
        start = torch.utils.benchmark.benchmark.Timer()
        for epoch in range(1, epochs + 1):
            train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, None)
            train_losses.append(train_loss)
            train_logits, train_labels = train_logits_list, train_labels_list  # placeholder
            train_auc = evaluate_model(model, train_loader)
            train_aucs.append(train_auc)

            dev_loss, dev_logits, dev_labels = evaluate_model(model, val_loader)
            dev_losses.append(dev_loss)
            dev_aucs.append(dev_auc)

        duration = float(start.Timer().Time())
        results = {
            "metrics": {
                "roc_auc_macro": dev_aucs[-1],
                "loss": dev_losses[-1]
            },
            "training_curves": {
                "loss": train_losses,
                "roc_auc_macro": train_aucs
            },
            "duration_seconds": duration
        }
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        results = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

class ThreeConvBaseline(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=hyperparams["dropout"])
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

