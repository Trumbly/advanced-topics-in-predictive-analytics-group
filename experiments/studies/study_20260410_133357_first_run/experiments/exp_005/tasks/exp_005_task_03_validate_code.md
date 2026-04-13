# Task exp_005_task_03_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:37:51.439024+00:00
- **Completed:** 2026-04-10 13:37:51.442024+00:00

## Code Used
```python
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
import time

try:
    # Load data
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        profile_path="data/processed/dataset_profile.json",
        spectrograms_dir="data/processed/spectrograms",
        labels_csv="data/processed/labels.csv",
        batch_size=128,
        num_workers=0,
        augmentation={"time_shift": True, "noise_injection": True, "mixup": 0.0, "specaugment": False},
    )
    epochs = min(50, 10)  # cap at 10
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train curves
    train_losses = []
    train_roc_aus = []

    # Warm up optimizer? Not needed.

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (spec, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(spec)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_positive_counts = np.zeros(num_classes, dtype=int)
        with torch.no_grad():
            for spec, labels in val_loader:
                logits = model(spec)
                val_loss += criterion(logits, labels.float()).item() * spec.size(0)
                preds = torch.sigmoid(logits)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
                pos_counts = np.sum(labels, axis=0)
                val_positive_counts += pos_counts

        val_loss = val_loss / n_batches

        # ROC AUC per class with at least one positive
        class_aus = []
        true_labels = np.concatenate(val_labels)
        pred_labels = np.concatenate(val_preds)
        for i in range(num_classes):
            if val_positive_counts[i] > 0:
                class_aus.append(roc_auc_score(true_labels[i], pred_labels[i]))
        train_roc_aus.append(np.mean(class_aus))

        optimizer.step()
        scheduler.step()

    duration = time.time() - start_time
    results = {
        "metrics": {"roc_auc_macro": train_roc_aus[-1], "loss": train_losses[-1]},
        "training_curves": {"loss": train_losses, "roc_auc_macro": train_roc_aus},
        "duration_seconds": duration,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

# Define custom model
class Model(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate model with correct number of classes
model = Model(num_classes=num_classes)

# Continue training as above (already defined)

# End of script

```

## Output
- **validation:** passed
- **code_bytes:** 3797
