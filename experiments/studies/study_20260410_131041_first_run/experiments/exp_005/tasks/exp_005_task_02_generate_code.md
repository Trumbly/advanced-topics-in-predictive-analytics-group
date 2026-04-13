# Task exp_005_task_02_generate_code

- **Experiment:** exp_005
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:12:49.844166+00:00
- **Completed:** 2026-04-10 13:13:18.765275+00:00

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
    "lr": 5e-05,
    "batch_size": 256,
    "epochs": 60,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
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
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1

def main():
    try:
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        hyperparams = {
            "lr": 5e-05,
            "batch_size": 256,
            "epochs": 60,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "dropout": 0.1
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cnn_small_v1(num_classes=num_classes, drop_rate=hyperparams["dropout"]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )
        train_losses = []
        val_losses = []
        val_aucs = []
        start_time = time.time()
        for epoch in range(1, hyperparams["epochs"] + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch["spectrogram"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

            model.eval()
            val_loss_sum = 0.0
            all_pred = []
            all_true = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["spectrogram"].to(device)
                    y = batch["labels"].to(device)
                    logits = model(x)
                    val_loss_sum += criterion(logits, y).item()
                    probs = torch.sigmoid(logits)
                    pred = (probs > 0.5).float().cpu()
                    all_pred.append(pred)
                    all_true.append(y.cpu())
            val_losses.append(val_loss_sum / len(val_loader))
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            val_aucs.append(roc_auc_score(y_true, y_pred))

            scheduler.step(val_losses[-1])

        metrics = {
            "roc_auc_macro": val_aucs[-1],
            "loss": val_losses[-1]
        }
        training_curves = {
            "loss": train_losses,
            "roc_auc_macro": val_aucs
        }
        duration_seconds = time.time() - start_time
        result = {"metrics": metrics, "training_curves": training_curves, "duration_seconds": duration_seconds}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result = {"error": str(e), "metrics": None, "training_curves": None, "duration_seconds": 0.0}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
```

## Output
- **code:** import json
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1

def main():
    try:
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        hyperparams = {
            "lr": 5e-05,
            "batch_size": 256,
            "epochs": 60,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "dropout": 0.1
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cnn_small_v1(num_classes=num_classes, drop_rate=hyperparams["dropout"]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )
        train_losses = []
        val_losses = []
        val_aucs = []
        start_time = time.time()
        for epoch in range(1, hyperparams["epochs"] + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch["spectrogram"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

            model.eval()
            val_loss_sum = 0.0
            all_pred = []
            all_true = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["spectrogram"].to(device)
                    y = batch["labels"].to(device)
                    logits = model(x)
                    val_loss_sum += criterion(logits, y).item()
                    probs = torch.sigmoid(logits)
                    pred = (probs > 0.5).float().cpu()
                    all_pred.append(pred)
                    all_true.append(y.cpu())
            val_losses.append(val_loss_sum / len(val_loader))
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            val_aucs.append(roc_auc_score(y_true, y_pred))

            scheduler.step(val_losses[-1])

        metrics = {
            "roc_auc_macro": val_aucs[-1],
            "loss": val_losses[-1]
        }
        training_curves = {
            "loss": train_losses,
            "roc_auc_macro": val_aucs
        }
        duration_seconds = time.time() - start_time
        result = {"metrics": metrics, "training_curves": training_curves, "duration_seconds": duration_seconds}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result = {"error": str(e), "metrics": None, "training_curves": None, "duration_seconds": 0.0}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

