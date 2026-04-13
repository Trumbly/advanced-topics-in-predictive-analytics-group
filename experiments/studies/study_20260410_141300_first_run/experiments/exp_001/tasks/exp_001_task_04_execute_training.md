# Task exp_001_task_04_execute_training

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-10 14:13:41.572890+00:00
- **Completed:** 2026-04-10 14:13:44.529873+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from torch.nn import Dropout

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
BATCH_SIZE = 32
EPOCHS = 2                # cap at 10
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition ===
from pipelines.models import CnnSmallV1
model = CnnSmallV1(num_classes=num_classes, in_channels=1)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

curves = {"loss": [], "roc_auc_macro": []}
start = time.time()
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            logits = Dropout(0.1)(logits)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        all_probs, all_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                probs = torch.sigmoid(model(x)).numpy()
                all_probs.append(probs)
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
- **exit_code:** 1
- **duration_seconds:** 2.9552735000033863
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_141300_first_run/exp_001
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** NameError: name 'num_classes' is not defined

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_141300_first_run/exp_001/code.py", line 18, in <module>
    model = CnnSmallV1(num_classes=num_classes, in_channels=1)
                                   ^^^^^^^^^^^
NameError: name 'num_classes' is not defined
```
