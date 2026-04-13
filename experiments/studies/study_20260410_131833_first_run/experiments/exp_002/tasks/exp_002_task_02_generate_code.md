# Task exp_002_task_02_generate_code

- **Experiment:** exp_002
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:19:12.248704+00:00
- **Completed:** 2026-04-10 13:19:37.433229+00:00

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
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "SGD",
    "weight_decay": 0.001,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": true,
    "mixup": 0.2,
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
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Any

from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from sklearn.metrics import roc_auc_score
import numpy as np

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb, reduction='mean')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
        preds = torch.softmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())
    epoch_loss /= len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy().ravel()
    macro_auc = roc_auc_score(labels, preds, average='macro')
    return epoch_loss, macro_auc, preds, labels

def main():
    try:
        start = time.time()
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cnn_small_v1(num_classes).to(device)
        optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = BCEWithLogitsLoss()
        train_curves = []
        for epoch in range(1, 101):
            train_loss, train_auc, _, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            train_curves.append({'loss': train_loss, 'roc_auc_macro': train_auc})
        final_loss = train_curves[-1]['loss']
        final_auc = train_curves[-1]['roc_auc_macro']
        duration = time.time() - start
        results = {
            "metrics": {"roc_auc_macro": float(final_auc), "loss": float(final_loss)},
            "training_curves": [{"loss": float(item['loss']), "roc_auc_macro": float(item['roc_auc_macro'])} for item in train_curves],
            "duration_seconds": float(duration)
        }
        with open('results.json', 'w') as f:
            json.dump(results, f)
    except Exception as e:
        results = {"error": str(e)}
        with open('results.json', 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()
```

## Output
- **code:** import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Any

from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from sklearn.metrics import roc_auc_score
import numpy as np

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb, reduction='mean')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
        preds = torch.softmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())
    epoch_loss /= len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy().ravel()
    macro_auc = roc_auc_score(labels, preds, average='macro')
    return epoch_loss, macro_auc, preds, labels

def main():
    try:
        start = time.time()
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cnn_small_v1(num_classes).to(device)
        optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = BCEWithLogitsLoss()
        train_curves = []
        for epoch in range(1, 101):
            train_loss, train_auc, _, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            train_curves.append({'loss': train_loss, 'roc_auc_macro': train_auc})
        final_loss = train_curves[-1]['loss']
        final_auc = train_curves[-1]['roc_auc_macro']
        duration = time.time() - start
        results = {
            "metrics": {"roc_auc_macro": float(final_auc), "loss": float(final_loss)},
            "training_curves": [{"loss": float(item['loss']), "roc_auc_macro": float(item['roc_auc_macro'])} for item in train_curves],
            "duration_seconds": float(duration)
        }
        with open('results.json', 'w') as f:
            json.dump(results, f)
    except Exception as e:
        results = {"error": str(e)}
        with open('results.json', 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()

