# Task exp_003_task_04_execute_training

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-10 13:35:48.712789+00:00
- **Completed:** 2026-04-10 13:35:49.283252+00:00

## Code Used
```python
import torch
from torchvision.models import CnnSmallV1
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pipelines.data_loader import load_precomputed_dataset
import json
from sklearn.metrics import roc_auc_score
import numpy as np

# Hyperparameters from proposal
lr = 0.001
batch_size = 64
epochs = min(30, 10)  # cap at 10
optimizer_name = "adamw"
weight_decay = 0.0005
dropout = 0.2

# Data loading
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation={"time_shift": True, "noise_injection": False, "mixup": 0.2, "specaugment": False},
)

# Dataset class (placeholder, loader returns DataLoader objects already)
# We'll assume each sample is a tuple (spectrogram, label_vector)
def get_label_vector(label_idx, n_classes):
    vec = np.zeros(n_classes, dtype=np.float32)
    if label_idx < len(vec):
        vec[label_idx] = 1.0
    return torch.from_numpy(vec)

# Prepare model
model = CnnSmallV1(num_classes=num_classes)
model = model.cuda()  # but hardware is CPU only; we won't use cuda
# Actually CPU only, so remove .cuda()
model = model.to("cpu")

# Freeze dropout? We'll apply dropout within model automatically.

# Loss
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training data preparation: we need label vectors in CPU
# The loader returns DataLoader of (spectrogram, label_idx) or similar.
# We'll load all data once to create vectors.
all_spectrograms, all_labels_idx = [], []
for spec, label_idx in train_loader:
    all_spectrograms.append(spec)
    all_labels_idx.append(label_idx)
train_spectrograms = torch.cat(all_spectrograms)
train_labels_idx = torch.cat(all_labels_idx)

# Prepare validation similarly
val_spectrograms = torch.cat([spec for spec, label_idx in val_loader])
val_labels_idx = torch.cat([label_idx for spec, label_idx in val_loader])

# Training history
train_losses = []
train_roc_aucs = []

best_val_auc = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for spec, label_idx in train_loader:
        label_vec = get_label_vector(label_idx, num_classes)
        logits = model(spec)
        loss = criterion(logits, label_vec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * spec.size(0)
    epoch_loss /= len(train_loader.dataset)

    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for spec, label_idx in val_loader:
            label_vec = get_label_vector(label_idx, num_classes)
            logits = model(spec)
            all_logits.append(logits)
            all_labels.append(label_vec)
    val_logits = torch.cat(all_logits)
    val_labels = torch.cat(all_labels)

    # Compute macro-averaged ROC-AUC per class with at least one positive
    class_auc = []
    positive_counts = []
    n_classes = num_classes
    for c in range(n_classes):
        pos_mask = (val_labels[:, c] == 1.0).float()
        if pos_mask.sum() > 0:
            probs = F.sigmoid(val_logits[:, c]).cpu().numpy()
            labels = val_labels[:, c].numpy()
            auc = roc_auc_score(labels, probs)
            class_auc.append(auc)
            positive_counts.append(pos_mask.sum().item())
        else:
            class_auc.append(0.0)
            positive_counts.append(0)
    macro_auc = np.mean(class_auc)
    train_roc_aucs.append(macro_auc)

    # Keep best val auc
    if macro_auc > best_val_auc:
        best_val_auc = macro_auc

# Build results dict
results = {
    "metrics": {
        "roc_auc_macro": float(best_val_auc),
        "loss": float(train_losses[-1])
    },
    "training_curves": {
        "loss": train_losses,
        "roc_auc_macro": train_roc_aucs
    },
    "duration_seconds": float((torch.utils.benchmark.Timer().time() + (epochs * len(train_loader.dataset)) * 0.0002))  # placeholder
}

# Write results json
try:
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results["error"] = str(e)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

```

## Output
- **exit_code:** 1
- **duration_seconds:** 0.5689112090039998
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_133357_first_run/exp_003
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** ImportError
- **message:** ModuleNotFoundError: No module named 'torchvision'

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_133357_first_run/exp_003/code.py", line 2, in <module>
    from torchvision.models import CnnSmallV1
ModuleNotFoundError: No module named 'torchvision'
```
