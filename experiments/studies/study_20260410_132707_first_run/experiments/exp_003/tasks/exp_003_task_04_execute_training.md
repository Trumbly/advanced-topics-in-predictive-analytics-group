# Task exp_003_task_04_execute_training

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-10 13:29:05.502938+00:00
- **Completed:** 2026-04-10 13:29:28.050763+00:00

## Code Used
```python
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
- **exit_code:** 1
- **duration_seconds:** 22.54643945902353
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_132707_first_run/exp_003
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** FileNotFound
- **message:** FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/dataset_profile.json'

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_132707_first_run/exp_003/code.py", line 24, in <module>
    train_loader, val_loader, num_classes = load_precomputed_dataset(
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/pipelines/data_loader.py", line 141, in load_precomputed_dataset
    profile = DatasetProfile.from_json_file(Path(profile_path))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/agent/models.py", line 59, in from_json_file
    return cls.model_validate_json(path.read_text())
                                   ^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/pathlib.py", line 1058, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/pathlib.py", line 1044, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/dataset_profile.json'
```
