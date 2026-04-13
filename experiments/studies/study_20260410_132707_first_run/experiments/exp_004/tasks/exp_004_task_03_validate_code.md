# Task exp_004_task_03_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:30:07.300921+00:00
- **Completed:** 2026-04-10 13:30:07.301724+00:00

## Code Used
```python
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import json
import time

# Hyperparameters
batch_size = 64
lr = 0.001
epochs = min(30, 10)  # cap at 10
optimizer = torch.optim.Adam
weight_decay = 0.0
dropout = 0.0

# Augmentation dict
augmentation = {
    "time_shift": True,
    "noise_injection": True,
    "mixup": 0.0,
    "specaugment": True,
}

# Load dataset
train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=batch_size,
    num_workers=0,
    augmentation=augmentation,
)

# Instantiate model
model = CnnSmallV1(num_classes=num_classes, dropout=dropout)

# Loss
criterion = nn.BCEWithLogitsLoss()

# Optimizer
opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training curves
loss_curves = []
auc_curves = []

# Timing
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, y in train_loader:
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float())
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_samples += x.size(0)

    loss_curves.append(total_loss / n_samples)

    # Evaluate on validation set
    model.eval()
    all_logits = []
    all_labels = []
    for x, y in val_loader:
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.float().cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Predictions
    probs = torch.sigmoid(all_logits).numpy()
    # Per-class AUC
    per_class_auc = []
    for c in range(num_classes):
        y_true = all_labels[:, c].numpy()
        y_pred = probs[:, c]
        if (y_true == 1).any():
            per_class_auc.append(roc_auc_score(y_true, y_pred))
    # Macro avg only for classes with at least one positive
    if per_class_auc:
        macro_auc = sum(per_class_auc) / len(per_class_auc)
    else:
        macro_auc = 0.0
    auc_curves.append(macro_auc)

    loss_curves[-1] = total_loss / n_samples  # update last entry
    loss = total_loss / n_samples

# Duration
duration = time.time() - start_time

# Prepare results
results = {
    "metrics": {
        "roc_auc_macro": round(macro_auc, 4),
        "loss": round(loss, 4)
    },
    "training_curves": {
        "loss": loss_curves,
        "roc_auc_macro": auc_curves
    },
    "duration_seconds": round(duration, 2)
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

```

## Output
- **validation:** passed
- **code_bytes:** 2684
