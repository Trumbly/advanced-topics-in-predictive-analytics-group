# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-10 14:09:46.708352+00:00
- **Completed:** 2026-04-10 14:09:46.710230+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
BATCH_SIZE = 64
EPOCHS = 20                 # cap at 10
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.2, "specaugment": True}

# === Model definition ===
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, (1, 1))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class CustomCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.se4 = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# === Build the model ===
model = CustomCNN(num_classes=206, dropout=0.3)

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    curves = {"loss": [], "roc_auc_macro": []}
    for epoch in range(min(EPOCHS, 10)):
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

    results = {
        "metrics": {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "loss": curves["loss"][-1],
        },
        "training_curves": curves,
        "duration_seconds": time.time() - start,
    }
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **validation:** failed

## Error
- **type:** ForbiddenPattern
- **message:** Forbidden pattern found: 'eval('
