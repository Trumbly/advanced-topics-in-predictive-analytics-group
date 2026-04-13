# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 16:23:29.648514+00:00
- **Completed:** 2026-04-10 16:23:29.665948+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DROPOUT = 0.2
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.3, "specaugment": False}

# === Model definition (custom_conv_3block_1channel) ===
class CustomModel(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(DROPOUT)
        self.flatten = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.flatten(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === Load data loader ===
start = time.time()
train_loader, val_loader, num_classes = load_precomputed_dataset(
    batch_size=BATCH_SIZE,
    num_workers=0,
    augmentation=AUGMENTATION,
)

# === Build model and set up training ===
model = CustomModel(num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()

# === Training curves ===
curves = {"loss": [], "roc_auc_macro": []}
for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))

    model.eval()
    all_probs, all_targs = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_probs.append(torch.sigmoid(model(x)).numpy())
            all_targs.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    targs = np.concatenate(all_targs, axis=0)

    # Macro ROC-AUC over columns with at least one positive
    aucs = []
    for c in range(targs.shape[1]):
        if targs[:, c].sum() > 0:
            aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
    val_auc = float(np.mean(aucs)) if aucs else 0.0
    epoch_loss = float(np.mean(epoch_losses))

    curves["loss"].append(epoch_loss)
    curves["roc_auc_macro"].append(val_auc)

# === Results ===
results = {
    "metrics": {
        "roc_auc_macro": curves["roc_auc_macro"][-1],
        "loss": curves["loss"][-1],
    },
    "training_curves": curves,
    "duration_seconds": time.time() - start,
}

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **validation:** passed
- **code_bytes:** 3040
