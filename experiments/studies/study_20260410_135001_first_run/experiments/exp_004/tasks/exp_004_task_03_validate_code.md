# Task exp_004_task_03_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:53:10.926523+00:00
- **Completed:** 2026-04-10 13:53:10.931029+00:00

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
EPOCHS = 50  # capped at 10
LR = 0.0001
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.0, "specaugment": True}

num_classes = 234

class MyModel(nn.Module):
    def __init__(self, num_classes=234):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=1)
        self.conv1_1x1 = nn.Conv2d(64, 2, kernel_size=1)
        self.conv2 = nn.Conv2d(2, 64, kernel_size=3, stride=3, padding=1)
        self.conv2_1x1 = nn.Conv2d(64, 2, kernel_size=1)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, stride=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_1x1(x)
        x = self.conv2(x)
        x = self.conv2_1x1(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    model = MyModel(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    curves = {"loss": [], "roc_auc_macro": []}
    epochs_to_run = min(EPOCHS, 10)
    for epoch in range(epochs_to_run):
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
- **code_bytes:** 3016
