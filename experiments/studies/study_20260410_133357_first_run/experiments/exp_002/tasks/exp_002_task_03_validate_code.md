# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:35:08.332708+00:00
- **Completed:** 2026-04-10 13:35:08.335720+00:00

## Code Used
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
import json
import time
from pipelines.data_loader import load_precomputed_dataset

# architecture definition
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(128, num_classes)  # 234 classes as per instruction

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# hyperparams
hyperparams = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "optimizer": "adamw",
    "weight_decay": 0.0005,
    "dropout": 0.2
}
epochs = min(10, hyperparams["epochs"])  # cap at 10

# augmentation dict (all false, so ignore)

train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=hyperparams["batch_size"],
    num_workers=0,
    augmentation=hyperparams["augmentation"]
)

model = MyModel(num_classes=234)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

start = time.time()
train_losses = []
roc_aucs = []
macro_auc = 0.0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for spec, labels in train_loader:
        labels = labels.float()
        optimizer.zero_grad()
        logits = model(spec)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for spec, labels in val_loader:
            logits = model(spec)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    classes = np.unique(all_labels)
    valid_classes = classes[np.any(all_labels[:, np.newaxis, :] > 0.5, axis=1)]
    if len(valid_classes) > 0:
        roc_scores = [sklearn.metrics.roc_auc_score(all_labels[valid_classes], all_preds[valid_classes], multi_class='ovr', average='macro')]
        macro_auc = np.mean(roc_scores)
    else:
        macro_auc = 0.0
    roc_aucs.append(macro_auc)

    if epoch == epochs - 1:
        macro_auc = np.mean(roc_aucs)

# final metrics
final_loss = train_losses[-1]
final_roc_auc = macro_auc

# duration
duration_seconds = time.time() - start

# results
results = {
    "metrics": {"roc_auc_macro": float(final_roc_auc), "loss": float(final_loss)},
    "training_curves": {"loss": train_losses, "roc_auc_macro": roc_aucs},
    "duration_seconds": float(duration_seconds)
}

try:
    open("results.json", "w").write(json.dumps(results, indent=0))
except:
    pass

```

## Output
- **validation:** passed
- **code_bytes:** 3670
