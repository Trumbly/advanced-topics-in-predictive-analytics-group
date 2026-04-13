# Task exp_002_task_02_generate_code

- **Experiment:** exp_002
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:11:18.234869+00:00
- **Completed:** 2026-04-10 13:11:37.347793+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Your task is to write a single, self-contained training
script that trains and evaluates a model for the BirdCLEF 2026 multi-label task.

STRICT RULES:
  - Use PyTorch (unless the model registry entry specifies TensorFlow)
  - Import the fixed data loader: `from pipelines.data_loader import load_precomputed_dataset`
    This returns `(train_loader, val_loader, num_classes)`.
  - Import model classes from `pipelines.models` when available, or use the `import_snippet`
    from the registry exactly as given.
  - Write final results to `results.json` with this schema:
      {{"metrics": {{"roc_auc_macro": <float>, "loss": <float>}},
        "training_curves": {{"loss": [...], "roc_auc_macro": [...]}},
        "duration_seconds": <float>}}
  - Do NOT import: `os.system`, `subprocess`, `urllib`, `requests`
  - Do NOT download anything from the internet
  - Do NOT write outside the current directory
  - Catch exceptions and still write a results.json with an `error` field if training fails


[USER]
## Architecture Proposal
{
  "architecture": "resnet18",
  "pretrained_model": "resnet18",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 40,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "dropout": 0.2
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.2,
    "specaugment": true
  }
}

## Dataset Profile
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
Write a complete Python training script implementing the architecture above.
Return ONLY the Python code, no explanations, no markdown fences.

```

## LLM Response
```
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score
import json

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    model = import_snippet('resnet18', pretrained='resnet18', dropout=0.2)
    model = model(num_classes=num_classes)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_losses = []
    epoch_aucs = []
    start_time = time.time()

    for epoch in range(40):
        epoch_loss = 0.0
        model.train()
        for spec, y, y_hat in train_loader:
            optimizer.zero_grad()
            y_pred = model(spec)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for spec, y, y_hat in val_loader:
                y_pred = model(spec)
                all_preds.append(y_pred)
                all_targets.append(y)
        val_pred = torch.cat(all_preds)
        val_target = torch.cat(all_targets)
        val_loss = loss_fn(val_pred, val_target).item()
        macro_auc = roc_auc_score(val_target.numpy(), val_pred.numpy(), average='macro')
        epoch_aucs.append(macro_auc)
        scheduler.step()

    duration = time.time() - start_time
    final_metrics = {
        "roc_auc_macro": epoch_aucs[-1],
        "loss": epoch_losses[-1]
    }
    results = {
        "metrics": final_metrics,
        "training_curves": {"loss": epoch_losses, "roc_auc_macro": epoch_aucs},
        "duration_seconds": round(duration, 2)
    }

except Exception as e:
    results = {
        "error": str(e),
        "metrics": None,
        "training_curves": None,
        "duration_seconds": 0.0
    }

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
```

## Output
- **code:** from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score
import json

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    model = import_snippet('resnet18', pretrained='resnet18', dropout=0.2)
    model = model(num_classes=num_classes)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_losses = []
    epoch_aucs = []
    start_time = time.time()

    for epoch in range(40):
        epoch_loss = 0.0
        model.train()
        for spec, y, y_hat in train_loader:
            optimizer.zero_grad()
            y_pred = model(spec)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for spec, y, y_hat in val_loader:
                y_pred = model(spec)
                all_preds.append(y_pred)
                all_targets.append(y)
        val_pred = torch.cat(all_preds)
        val_target = torch.cat(all_targets)
        val_loss = loss_fn(val_pred, val_target).item()
        macro_auc = roc_auc_score(val_target.numpy(), val_pred.numpy(), average='macro')
        epoch_aucs.append(macro_auc)
        scheduler.step()

    duration = time.time() - start_time
    final_metrics = {
        "roc_auc_macro": epoch_aucs[-1],
        "loss": epoch_losses[-1]
    }
    results = {
        "metrics": final_metrics,
        "training_curves": {"loss": epoch_losses, "roc_auc_macro": epoch_aucs},
        "duration_seconds": round(duration, 2)
    }

except Exception as e:
    results = {
        "error": str(e),
        "metrics": None,
        "training_curves": None,
        "duration_seconds": 0.0
    }

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

