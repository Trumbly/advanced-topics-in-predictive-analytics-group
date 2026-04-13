# Task exp_001_task_04_execute_training

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-11 08:46:58.074776+00:00
- **Completed:** 2026-04-11 08:46:59.712873+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

print("loading data...", flush=True)
train_loader, val_loader, num_classes = load_precomputed_dataset(
    augmentation=AUGMENTATION,
)
print(
    f"data loaded: {num_classes} classes, "
    f"{len(train_loader.dataset)} train samples, "
    f"{len(val_loader.dataset)} val samples",
    flush=True,
)

model = CnnSmallV1(num_classes=num_classes, in_channels=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train_batches = len(train_loader)
log_every = max(1, n_train_batches // 10)
curves = {"loss": [], "roc_auc_macro": []}
start = time.time()

for epoch in range(EPOCHS):
    print(f"epoch {epoch + 1}/{EPOCHS} starting ({n_train_batches} batches)...", flush=True)
    model.train()
    epoch_losses = []
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))

        if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
            pct = 100.0 * (batch_idx + 1) / n_train_batches
            print(
                f"  epoch {epoch + 1} [{pct:5.1f}%] "
                f"batch {batch_idx + 1}/{n_train_batches} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    print(f"epoch {epoch + 1}: running validation...", flush=True)
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
    print(f"epoch {epoch + 1}/{EPOCHS} done: loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}", flush=True)

results = {
    "metrics": {
        "roc_auc_macro": curves["roc_auc_macro"][-1],
        "loss": curves["loss"][-1],
    },
    "training_curves": curves,
    "duration_seconds": time.time() - start,
}
print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **exit_code:** 1
- **duration_seconds:** 1.6366385830333456
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_084632_first_run/exp_001
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** NameError: name 'CnnSmallV1' is not defined

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_084632_first_run/exp_001/code.py", line 27, in <module>
    model = CnnSmallV1(num_classes=num_classes, in_channels=1)
            ^^^^^^^^^^
NameError: name 'CnnSmallV1' is not defined
```
