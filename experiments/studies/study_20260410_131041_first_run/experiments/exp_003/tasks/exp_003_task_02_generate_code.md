# Task exp_003_task_02_generate_code

- **Experiment:** exp_003
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:11:46.287699+00:00
- **Completed:** 2026-04-10 13:12:11.301658+00:00

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
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 40,
    "optimizer": "adam",
    "weight_decay": 0.0005,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.1,
    "specaugment": false
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
import json, time, torch, torch.nn as nn, torch.optim as optim, numpy as np
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet

architecture = "cnn_small_v1"
import_snippet = "cnn_small_v1"
model_class = import_snippet()
model = model_class()

lr = 0.001
batch_size = 128
epochs = 40
weight_decay = 0.0005
dropout = 0.1

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()

train_loader, val_loader, num_classes = load_precomputed_dataset()

train_losses = []
train_roc_aucs = []

start = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_total = 0
    all_preds = []
    all_labels = []
    for samples, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * samples.size(0)
        epoch_total += samples.size(0)
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.float().cpu())
    train_losses.append(epoch_loss / epoch_total)

    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for val_s, val_l in val_loader:
            preds = model(val_s).cpu().numpy()
            all_val_preds.append(preds)
            all_val_labels.append(val_l.numpy())
    preds = np.concatenate(all_val_preds).ravel()
    labels = np.concatenate(all_val_labels).ravel()
    auc = roc_auc_score(labels, preds, average='macro')
    train_roc_aucs.append(auc)

    model.train()

end = time.time()

with torch.no_grad():
    all_preds = []
    all_labels = []
    for val_s, val_l in val_loader:
        preds = model(val_s).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(val_l.numpy())
    preds = np.concatenate(all_preds).ravel()
    labels = np.concatenate(all_labels).ravel()
    final_auc = roc_auc_score(labels, preds, average='macro')

results = {
    "metrics": {"roc_auc_macro": final_auc, "loss": train_losses[-1]},
    "training_curves": {"loss": train_losses, "roc_auc_macro": train_roc_aucs},
    "duration_seconds": end - start
}
try:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
except Exception as e:
    results["error"] = str(e)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
```

## Output
- **code:** import json, time, torch, torch.nn as nn, torch.optim as optim, numpy as np
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet

architecture = "cnn_small_v1"
import_snippet = "cnn_small_v1"
model_class = import_snippet()
model = model_class()

lr = 0.001
batch_size = 128
epochs = 40
weight_decay = 0.0005
dropout = 0.1

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()

train_loader, val_loader, num_classes = load_precomputed_dataset()

train_losses = []
train_roc_aucs = []

start = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_total = 0
    all_preds = []
    all_labels = []
    for samples, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * samples.size(0)
        epoch_total += samples.size(0)
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.float().cpu())
    train_losses.append(epoch_loss / epoch_total)

    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for val_s, val_l in val_loader:
            preds = model(val_s).cpu().numpy()
            all_val_preds.append(preds)
            all_val_labels.append(val_l.numpy())
    preds = np.concatenate(all_val_preds).ravel()
    labels = np.concatenate(all_val_labels).ravel()
    auc = roc_auc_score(labels, preds, average='macro')
    train_roc_aucs.append(auc)

    model.train()

end = time.time()

with torch.no_grad():
    all_preds = []
    all_labels = []
    for val_s, val_l in val_loader:
        preds = model(val_s).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(val_l.numpy())
    preds = np.concatenate(all_preds).ravel()
    labels = np.concatenate(all_labels).ravel()
    final_auc = roc_auc_score(labels, preds, average='macro')

results = {
    "metrics": {"roc_auc_macro": final_auc, "loss": train_losses[-1]},
    "training_curves": {"loss": train_losses, "roc_auc_macro": train_roc_aucs},
    "duration_seconds": end - start
}
try:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
except Exception as e:
    results["error"] = str(e)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

