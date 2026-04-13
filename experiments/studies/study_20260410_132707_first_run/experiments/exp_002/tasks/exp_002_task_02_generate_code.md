# Task exp_002_task_02_generate_code

- **Experiment:** exp_002
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:28:00.159119+00:00
- **Completed:** 2026-04-10 13:28:31.503859+00:00

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
  "architecture": "efficientnet_b0",
  "pretrained_model": "efficientnet_b0",
  "hyperparams": {
    "lr": 0.0001,
    "batch_size": 32,
    "epochs": 30,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": false,
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
from pipelines.data_loader import load_precomputed_dataset
import time
from CnnSmallV1 import CnnSmallV1

hyperparams = {
    "batch_size": 32,
    "epochs": 30,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "adam",
    "dropout": 0.1
}
epochs = 10 if hyperparams["epochs"] > 10 else hyperparams["epochs"]

train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=hyperparams["batch_size"],
    num_workers=0,
    augmentation=hyperparams["augmentation"]
)

model = CnnSmallV1(num_classes=num_classes)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

loss_history = []
roc_auc_history = []
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    n = 0
    for data in train_loader:
        x = data["spectrogram"]
        y = data["label"]
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n += 1
    epoch_loss /= n
    loss_history.append(epoch_loss)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            x = data["spectrogram"]
            y = data["label"]
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.float().cpu().numpy())
    val_preds = torch.cat(all_preds).numpy()
    val_labels = torch.cat(all_labels).numpy()

    roc_aus = []
    for i in range(num_classes):
        if (val_labels[:, i] == 0).all():
            continue
        roc_aus.append(roc_auc_score(val_labels[:, i], val_preds[:, i]))
    epoch_auc = sum(roc_aus) / len(roc_aus) if roc_aus else 0.0
    roc_auc_history.append(epoch_auc)
    model.train()

overall_loss = sum(loss_history) / len(loss_history)
overall_auc = sum(roc_auc_history) / len(roc_auc_history)

results = {
    "metrics": {"roc_auc_macro": overall_auc, "loss": overall_loss},
    "training_curves": {"loss": loss_history, "roc_auc_macro": roc_auc_history},
    "duration_seconds": time.time() - start_time
}

try:
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        f.write(json.dumps(results))
```

## Output
- **code:** import torch
from pipelines.data_loader import load_precomputed_dataset
import time
from CnnSmallV1 import CnnSmallV1

hyperparams = {
    "batch_size": 32,
    "epochs": 30,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "adam",
    "dropout": 0.1
}
epochs = 10 if hyperparams["epochs"] > 10 else hyperparams["epochs"]

train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=hyperparams["batch_size"],
    num_workers=0,
    augmentation=hyperparams["augmentation"]
)

model = CnnSmallV1(num_classes=num_classes)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

loss_history = []
roc_auc_history = []
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    n = 0
    for data in train_loader:
        x = data["spectrogram"]
        y = data["label"]
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n += 1
    epoch_loss /= n
    loss_history.append(epoch_loss)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            x = data["spectrogram"]
            y = data["label"]
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.float().cpu().numpy())
    val_preds = torch.cat(all_preds).numpy()
    val_labels = torch.cat(all_labels).numpy()

    roc_aus = []
    for i in range(num_classes):
        if (val_labels[:, i] == 0).all():
            continue
        roc_aus.append(roc_auc_score(val_labels[:, i], val_preds[:, i]))
    epoch_auc = sum(roc_aus) / len(roc_aus) if roc_aus else 0.0
    roc_auc_history.append(epoch_auc)
    model.train()

overall_loss = sum(loss_history) / len(loss_history)
overall_auc = sum(roc_auc_history) / len(roc_auc_history)

results = {
    "metrics": {"roc_auc_macro": overall_auc, "loss": overall_loss},
    "training_curves": {"loss": loss_history, "roc_auc_macro": roc_auc_history},
    "duration_seconds": time.time() - start_time
}

try:
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        f.write(json.dumps(results))

