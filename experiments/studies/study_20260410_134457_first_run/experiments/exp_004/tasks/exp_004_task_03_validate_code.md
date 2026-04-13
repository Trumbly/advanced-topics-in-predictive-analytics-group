# Task exp_004_task_03_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:47:41.960219+00:00
- **Completed:** 2026-04-10 13:47:41.966561+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

BATCH_SIZE = 128
EPOCHS = 10  # cap at 10
LR = 0.0005
weight_decay = 0.0001
OPTIMIZER = 'Adam'
DROPOUT = 0.1
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.2, "specaugment": False}

class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(channels, 1, 1))
        self.alpha = nn.Parameter(torch.clamp(self.alpha, min=0, max=1))
        self.gamma = nn.Parameter(torch.clamp(self.gamma, min=0))
        self.beta = nn.Parameter(torch.clamp(self.beta, min=0))

    def forward(self, x):
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.var(x, dim=[2, 3], keepdim=True, unbiased=False)
        scale = self.gamma * self.beta + self.alpha * (1 - self.alpha)
        weight = torch.nn.functional.relu(scale - 1)
        weight = weight / weight.mean()
        return x * weight

class MyModel(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SE(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.se2 = SE(32)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.se1(x)
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.se2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    model = MyModel(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

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
- **validation:** passed
- **code_bytes:** 3927
