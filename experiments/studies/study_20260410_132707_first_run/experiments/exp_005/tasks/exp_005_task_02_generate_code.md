# Task exp_005_task_02_generate_code

- **Experiment:** exp_005
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:30:19.617319+00:00
- **Completed:** 2026-04-10 13:31:05.837526+00:00

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
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 20,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.0
  },
  "augmentation": {
    "time_shift": true,
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
import json
import time
from torch import nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

def main():
    start = time.time()
    try:
        hyperparams = {
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 20,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "dropout": 0.0
        }
        epochs = min(hyperparams["epochs"], 10)
        loss_fn = nn.BCEWithLogitsLoss()
        model = CnnSmallV1(num_classes=206)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        augment = hyperparams["augmentation"]
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            profile_path="data/processed/dataset_profile.json",
            spectrograms_dir="data/processed/spectrograms",
            labels_csv="data/processed/labels.csv",
            batch_size=hyperparams["batch_size"],
            num_workers=0,
            augmentation=augment,
        )
        training_losses = []
        roc_aucs = []
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for spec, labels in train_loader:
                optimizer.zero_grad()
                logits = model(spec)
                loss = loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_avg = epoch_loss / len(train_loader)
            training_losses.append(loss_avg)

            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for spec, labels in val_loader:
                    logits = model(spec)
                    all_logits.append(logits)
                    all_labels.append(labels.float())
            val_logits = torch.cat(all_logits)
            val_probs = torch.sigmoid(val_logits)
            val_bin = (val_probs > 0.5).float()
            per_class_roc = []
            for i in range(num_classes):
                pos_mask = val_bin[:, i] == 1
                if pos_mask.any():
                    per_class_roc.append(roc_auc_score(val_bin[:, i], val_probs[:, i]))
                else:
                    per_class_roc.append(0.0)
            roc_auc_macro = sum(per_class_roc) / len(per_class_roc) if per_class_roc else 0.0
            roc_aucs.append(roc_auc_macro)

        final_loss = training_losses[-1]
        final_roc = roc_aucs[-1]

        results = {
            "metrics": {"roc_auc_macro": final_roc, "loss": final_loss},
            "training_curves": {"loss": training_losses, "roc_auc_macro": roc_aucs},
            "duration_seconds": time.time() - start
        }
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        err = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(err, f, indent=2)

if __name__ == "__main__":
    main()
```

## Output
- **code:** import torch
import json
import time
from torch import nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

def main():
    start = time.time()
    try:
        hyperparams = {
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 20,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "dropout": 0.0
        }
        epochs = min(hyperparams["epochs"], 10)
        loss_fn = nn.BCEWithLogitsLoss()
        model = CnnSmallV1(num_classes=206)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        augment = hyperparams["augmentation"]
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            profile_path="data/processed/dataset_profile.json",
            spectrograms_dir="data/processed/spectrograms",
            labels_csv="data/processed/labels.csv",
            batch_size=hyperparams["batch_size"],
            num_workers=0,
            augmentation=augment,
        )
        training_losses = []
        roc_aucs = []
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for spec, labels in train_loader:
                optimizer.zero_grad()
                logits = model(spec)
                loss = loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_avg = epoch_loss / len(train_loader)
            training_losses.append(loss_avg)

            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for spec, labels in val_loader:
                    logits = model(spec)
                    all_logits.append(logits)
                    all_labels.append(labels.float())
            val_logits = torch.cat(all_logits)
            val_probs = torch.sigmoid(val_logits)
            val_bin = (val_probs > 0.5).float()
            per_class_roc = []
            for i in range(num_classes):
                pos_mask = val_bin[:, i] == 1
                if pos_mask.any():
                    per_class_roc.append(roc_auc_score(val_bin[:, i], val_probs[:, i]))
                else:
                    per_class_roc.append(0.0)
            roc_auc_macro = sum(per_class_roc) / len(per_class_roc) if per_class_roc else 0.0
            roc_aucs.append(roc_auc_macro)

        final_loss = training_losses[-1]
        final_roc = roc_aucs[-1]

        results = {
            "metrics": {"roc_auc_macro": final_roc, "loss": final_loss},
            "training_curves": {"loss": training_losses, "roc_auc_macro": roc_aucs},
            "duration_seconds": time.time() - start
        }
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        err = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(err, f, indent=2)

if __name__ == "__main__":
    main()

