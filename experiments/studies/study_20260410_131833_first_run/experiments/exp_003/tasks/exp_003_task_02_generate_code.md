# Task exp_003_task_02_generate_code

- **Experiment:** exp_003
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:19:48.887500+00:00
- **Completed:** 2026-04-10 13:20:09.956747+00:00

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
    "lr": 0.01,
    "batch_size": 64,
    "epochs": 150,
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "dropout": 0.0
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": false,
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
import torch
import torch.nn as nn
import torch.optim as optim
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    model = import_snippet("cnn_small_v1")
    model = model.to('cpu')  # assuming CPU
    model.train()

    batch_size = 64
    epochs = 150
    lr = 0.01
    weight_decay = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    loss_history = []
    auc_history = []
    start = time.time()
    final_loss = None
    final_auc = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        model.train()
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            spec = spec.to('cpu')
            labels = labels.to('cpu')
            outputs = model(spec)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss /= (batch_idx + 1)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_preds = []
            val_labels = []
            for spec, labels in val_loader:
                spec = spec.to('cpu')
                labels = labels.to('cpu')
                outputs = model(spec)
                val_loss += loss_fn(outputs, labels).item()
                val_preds.append(outputs.detach().cpu().numpy())
                val_labels.append(labels.cpu().numpy())
            val_loss /= len(val_loader)
            val_loss = val_loss / len(val_loader)

            # Compute macro AUC
            val_preds_np = np.concatenate(val_preds, axis=0)
            val_labels_np = np.concatenate(val_labels, axis=0)
            valid_auc = roc_auc_score(val_labels_np, val_preds_np, multi_class='multilabel-roc')
            final_auc = valid_auc
            final_loss = val_loss

        epoch_auc = final_auc
        loss_history.append(epoch_loss)
        auc_history.append(final_auc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {final_loss:.4f}, Val AUC: {final_auc:.4f}")

    duration = time.time() - start
    final_metrics = {
        "roc_auc_macro": final_auc,
        "loss": final_loss
    }
    training_curves = {
        "loss": loss_history,
        "roc_auc_macro": auc_history
    }
    results = {
        "metrics": final_metrics,
        "training_curves": training_curves,
        "duration_seconds": duration
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    error_msg = f"Error occurred: {str(e)}"
    results = {"error": error_msg}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
```

## Output
- **code:** import json
import torch
import torch.nn as nn
import torch.optim as optim
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    model = import_snippet("cnn_small_v1")
    model = model.to('cpu')  # assuming CPU
    model.train()

    batch_size = 64
    epochs = 150
    lr = 0.01
    weight_decay = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    loss_history = []
    auc_history = []
    start = time.time()
    final_loss = None
    final_auc = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        model.train()
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            spec = spec.to('cpu')
            labels = labels.to('cpu')
            outputs = model(spec)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss /= (batch_idx + 1)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_preds = []
            val_labels = []
            for spec, labels in val_loader:
                spec = spec.to('cpu')
                labels = labels.to('cpu')
                outputs = model(spec)
                val_loss += loss_fn(outputs, labels).item()
                val_preds.append(outputs.detach().cpu().numpy())
                val_labels.append(labels.cpu().numpy())
            val_loss /= len(val_loader)
            val_loss = val_loss / len(val_loader)

            # Compute macro AUC
            val_preds_np = np.concatenate(val_preds, axis=0)
            val_labels_np = np.concatenate(val_labels, axis=0)
            valid_auc = roc_auc_score(val_labels_np, val_preds_np, multi_class='multilabel-roc')
            final_auc = valid_auc
            final_loss = val_loss

        epoch_auc = final_auc
        loss_history.append(epoch_loss)
        auc_history.append(final_auc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {final_loss:.4f}, Val AUC: {final_auc:.4f}")

    duration = time.time() - start
    final_metrics = {
        "roc_auc_macro": final_auc,
        "loss": final_loss
    }
    training_curves = {
        "loss": loss_history,
        "roc_auc_macro": auc_history
    }
    results = {
        "metrics": final_metrics,
        "training_curves": training_curves,
        "duration_seconds": duration
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    error_msg = f"Error occurred: {str(e)}"
    results = {"error": error_msg}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

