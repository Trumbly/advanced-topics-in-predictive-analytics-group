# Task exp_001_task_04_execute_training

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** execute_training
- **Status:** timeout
- **Started:** 2026-04-10 14:30:45.135803+00:00
- **Completed:** 2026-04-10 15:47:02.491060+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import CnnSmallV1

BATCH_SIZE = 32
EPOCHS = 2
LR = 0.001
AUGMENTATION = {"time_shift": True, "noise_injection": True}

num_classes = 206

model = CnnSmallV1(num_classes=num_classes, in_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
criterion = nn.BCEWithLogitsLoss()

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

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
- **exit_code:** -1
- **duration_seconds:** 600.0559659170103
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_143004_first_run/exp_001
- **results_json_path:** None
- **timed_out:** True

## Error
- **type:** Timeout
- **message:** Process exceeded configured timeout
