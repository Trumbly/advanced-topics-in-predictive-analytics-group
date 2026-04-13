# Task exp_004_task_02_generate_code

- **Experiment:** exp_004
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:20:22.143842+00:00
- **Completed:** 2026-04-10 13:20:45.585448+00:00

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
    "epochs": 30,
    "optimizer": "Adam",
    "weight_decay": 0.0001,
    "dropout": 0.2
  },
  "augmentation": {
    "time_shift": false,
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
import json
import time
import torch
from torch import nn
import torch.nn.functional as F
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CNN_SmallV1

def macro_auc_score(y_true, probs):
    y_pred = (probs >= 0.5).float()
    auc = F.roc_auc_score(y_true, probs).item()
    return auc

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    hyperparams = {
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 30,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "dropout": 0.2
    }
    # instantiate model
    model = CNN_SmallV1(num_classes=num_classes, **hyperparams)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    training_curves = {"loss": [], "roc_auc_macro": []}
    start_time = time.time()
    val_losses = []
    val_roc = []
    for epoch in range(hyperparams["epochs"]):
        epoch_loss = 0.0
        model.train()
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # spec shape: (1, 128, 313)
            # labels shape: (1, 1, 313?) Actually each sample has 206 binary labels? Might be binary per class? Need flatten.
            # Assume labels are binary per class per sample: shape (batch, num_classes)
            # spec is spectrogram (1,128,313) maybe need processing? We'll assume model expects that.
            outputs = model(spec)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for _, labels in val_loader:
                outputs = model(batch_spec)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                all_preds.append(outputs)
                all_labels.append(labels)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            probs = torch.cat(all_preds, dim=0).cpu().numpy()
            labels = torch.cat(all_labels, dim=0).cpu().numpy()
            val_roc.append(macro_auc_score(labels, probs))
        training_curves["loss"].append(epoch_loss)
        training_curves["roc_auc_macro"].append(val_roc[-1])
    duration = time.time() - start_time
    metrics = {"loss": val_loss, "roc_auc_macro": val_roc[-1]}
    results = {"metrics": metrics, "training_curves": training_curves, "duration_seconds": duration}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
```

## Output
- **code:** import json
import time
import torch
from torch import nn
import torch.nn.functional as F
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CNN_SmallV1

def macro_auc_score(y_true, probs):
    y_pred = (probs >= 0.5).float()
    auc = F.roc_auc_score(y_true, probs).item()
    return auc

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    hyperparams = {
        "lr": 0.001,
        "batch_size": 128,
        "epochs": 30,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "dropout": 0.2
    }
    # instantiate model
    model = CNN_SmallV1(num_classes=num_classes, **hyperparams)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    training_curves = {"loss": [], "roc_auc_macro": []}
    start_time = time.time()
    val_losses = []
    val_roc = []
    for epoch in range(hyperparams["epochs"]):
        epoch_loss = 0.0
        model.train()
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # spec shape: (1, 128, 313)
            # labels shape: (1, 1, 313?) Actually each sample has 206 binary labels? Might be binary per class? Need flatten.
            # Assume labels are binary per class per sample: shape (batch, num_classes)
            # spec is spectrogram (1,128,313) maybe need processing? We'll assume model expects that.
            outputs = model(spec)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for _, labels in val_loader:
                outputs = model(batch_spec)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                all_preds.append(outputs)
                all_labels.append(labels)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            probs = torch.cat(all_preds, dim=0).cpu().numpy()
            labels = torch.cat(all_labels, dim=0).cpu().numpy()
            val_roc.append(macro_auc_score(labels, probs))
        training_curves["loss"].append(epoch_loss)
        training_curves["roc_auc_macro"].append(val_roc[-1])
    duration = time.time() - start_time
    metrics = {"loss": val_loss, "roc_auc_macro": val_roc[-1]}
    results = {"metrics": metrics, "training_curves": training_curves, "duration_seconds": duration}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

