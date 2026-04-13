# Task exp_003_task_02_generate_code

- **Experiment:** exp_003
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 13:28:41.559813+00:00
- **Completed:** 2026-04-10 13:29:05.494474+00:00

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
  "architecture": "resnet18",
  "pretrained_model": "resnet18",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 20,
    "optimizer": "adam",
    "weight_decay": 0.0001,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
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
from torch.nn import BCEWithLogitsLoss
import json
from sklearn.metrics import roc_auc_score
import time

# Hyperparameters from proposal
lr = 0.001
batch_size = 64
epochs = 20
optimizer_name = "adam"
weight_decay = 0.0001
dropout = 0.1
mixup_alpha = 0.2

# Augmentation dict
augmentation = {"time_shift": True, "noise_injection": False, "mixup": mixup_alpha, "specaugment": False}

# Cap epochs at 10
epochs = min(epochs, 10)

# Load data
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation=augmentation,
)

# Instantiate model using exact class name from registry
model = CnnSmallV1(num_classes=num_classes)

# Optimizer
if optimizer_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

# Loss function
criterion = BCEWithLogitsLoss()

# Storage for metrics
training_losses = []
training_roc_aucs = []

# Start timer
start_time = time.time()

try:
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_logits = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # Apply mixup if enabled
            if augmentation["mixup"] > 0:
                alpha = augmentation["mixup"]
                idx = torch.rand(1, device='cpu')
                idx = idx.item()
                idx = torch.tensor([idx], device='cpu')
                lam = torch.linspace(0, 1, 2).to('cpu').float()
                lam = lam / lam.sum() * idx  # convex combination
                lam = torch.tensor([lam[0].item(), lam[1].item()], device='cpu')

                # Mixup labels
                labels_mixup = lam[0] * labels.float() + lam[1] * inputs.float()

                # Mixup inputs
                alpha = torch.tensor([alpha, 1 - alpha], device='cpu').float()
                inputs_mixup = alpha[0] * inputs + alpha[1] * inputs_mixup

                outputs = model(inputs_mixup)
                loss = criterion(outputs, labels_mixup)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_logits.append(outputs.detach().cpu())

        # Compute epoch loss
        epoch_loss /= len(train_loader)
        training_losses.append(epoch_loss)

        # Validation: macro ROC AUC
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                if augmentation["mixup"] > 0:
                    alpha = augmentation["mixup"]
                    idx = torch.rand(1, device='cpu')
                    idx = idx.item()
                    idx = torch.tensor([idx], device='cpu')
                    lam = torch.linspace(0, 1, 2).to('cpu').float()
                    lam = lam / lam.sum() * idx
                    lam = torch.tensor([lam[0].item(), lam[1].item()], device='cpu')
                    labels_mixup = lam[0] * labels.float() + lam[1] * inputs.float()
                    inputs_mixup = alpha[0] * inputs + alpha[1] * inputs_mixup
                    outputs = model(inputs_mixup)
                    logits = outputs
                    labels_mixup_tensor = torch.cat([labels_mixup, torch.zeros(labels.shape[0], device='cpu')])
                else:
                    outputs = model(inputs)
                    logits = outputs
                    labels_mixup_tensor = torch.cat([labels.float(), torch.zeros(labels.shape[0], device='cpu')])

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels_mixup_tensor.detach().cpu())

        val_logits = torch.cat(all_logits)
        val_labels = torch.cat(all_labels)

        # Compute macro ROC AUC per class with at least one positive
        macro_auc = []
        for c in range(num_classes):
            mask = val_labels[:, c].float() > 0.5
            cnt = mask.sum().item()
            if cnt > 0:
                probs = val_logits[:, c].cpu().numpy()
                auc = roc_auc_score(val_labels[:, c].cpu().numpy(), probs)
                macro_auc.append(auc)
            else:
                macro_auc.append(0.0)
        training_roc_aucs.append(sum(macro_auc) / len(macro_auc))

    # Final metrics
    final_loss = training_losses[-1]
    final_roc_auc = training_roc_aucs[-1]

    # Write results.json
    results = {
        "metrics": {"roc_auc_macro": float(final_roc_auc), "loss": float(final_loss)},
        "training_curves": {
            "loss": training_losses,
            "roc_auc_macro": training_roc_aucs
        },
        "duration_seconds": float(time.time() - start_time)
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    error = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(error, f, indent=2)
```

## Output
- **code:** import torch
from pipelines.data_loader import load_precomputed_dataset
from torch.nn import BCEWithLogitsLoss
import json
from sklearn.metrics import roc_auc_score
import time

# Hyperparameters from proposal
lr = 0.001
batch_size = 64
epochs = 20
optimizer_name = "adam"
weight_decay = 0.0001
dropout = 0.1
mixup_alpha = 0.2

# Augmentation dict
augmentation = {"time_shift": True, "noise_injection": False, "mixup": mixup_alpha, "specaugment": False}

# Cap epochs at 10
epochs = min(epochs, 10)

# Load data
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation=augmentation,
)

# Instantiate model using exact class name from registry
model = CnnSmallV1(num_classes=num_classes)

# Optimizer
if optimizer_name == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

# Loss function
criterion = BCEWithLogitsLoss()

# Storage for metrics
training_losses = []
training_roc_aucs = []

# Start timer
start_time = time.time()

try:
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_logits = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # Apply mixup if enabled
            if augmentation["mixup"] > 0:
                alpha = augmentation["mixup"]
                idx = torch.rand(1, device='cpu')
                idx = idx.item()
                idx = torch.tensor([idx], device='cpu')
                lam = torch.linspace(0, 1, 2).to('cpu').float()
                lam = lam / lam.sum() * idx  # convex combination
                lam = torch.tensor([lam[0].item(), lam[1].item()], device='cpu')

                # Mixup labels
                labels_mixup = lam[0] * labels.float() + lam[1] * inputs.float()

                # Mixup inputs
                alpha = torch.tensor([alpha, 1 - alpha], device='cpu').float()
                inputs_mixup = alpha[0] * inputs + alpha[1] * inputs_mixup

                outputs = model(inputs_mixup)
                loss = criterion(outputs, labels_mixup)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_logits.append(outputs.detach().cpu())

        # Compute epoch loss
        epoch_loss /= len(train_loader)
        training_losses.append(epoch_loss)

        # Validation: macro ROC AUC
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                if augmentation["mixup"] > 0:
                    alpha = augmentation["mixup"]
                    idx = torch.rand(1, device='cpu')
                    idx = idx.item()
                    idx = torch.tensor([idx], device='cpu')
                    lam = torch.linspace(0, 1, 2).to('cpu').float()
                    lam = lam / lam.sum() * idx
                    lam = torch.tensor([lam[0].item(), lam[1].item()], device='cpu')
                    labels_mixup = lam[0] * labels.float() + lam[1] * inputs.float()
                    inputs_mixup = alpha[0] * inputs + alpha[1] * inputs_mixup
                    outputs = model(inputs_mixup)
                    logits = outputs
                    labels_mixup_tensor = torch.cat([labels_mixup, torch.zeros(labels.shape[0], device='cpu')])
                else:
                    outputs = model(inputs)
                    logits = outputs
                    labels_mixup_tensor = torch.cat([labels.float(), torch.zeros(labels.shape[0], device='cpu')])

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels_mixup_tensor.detach().cpu())

        val_logits = torch.cat(all_logits)
        val_labels = torch.cat(all_labels)

        # Compute macro ROC AUC per class with at least one positive
        macro_auc = []
        for c in range(num_classes):
            mask = val_labels[:, c].float() > 0.5
            cnt = mask.sum().item()
            if cnt > 0:
                probs = val_logits[:, c].cpu().numpy()
                auc = roc_auc_score(val_labels[:, c].cpu().numpy(), probs)
                macro_auc.append(auc)
            else:
                macro_auc.append(0.0)
        training_roc_aucs.append(sum(macro_auc) / len(macro_auc))

    # Final metrics
    final_loss = training_losses[-1]
    final_roc_auc = training_roc_aucs[-1]

    # Write results.json
    results = {
        "metrics": {"roc_auc_macro": float(final_roc_auc), "loss": float(final_loss)},
        "training_curves": {
            "loss": training_losses,
            "roc_auc_macro": training_roc_aucs
        },
        "duration_seconds": float(time.time() - start_time)
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    error = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(error, f, indent=2)

