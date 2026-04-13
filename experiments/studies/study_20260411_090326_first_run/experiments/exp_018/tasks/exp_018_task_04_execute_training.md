# Task exp_018_task_04_execute_training

- **Experiment:** exp_018
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-11 10:47:58.479762+00:00
- **Completed:** 2026-04-11 10:48:15.574748+00:00

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
from pipelines.models import CnnSmallV1

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
print(f"device: {device}", flush=True)
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0, "specaugment": False}

if __name__ == "__main__":
    print("loading data...", flush=True)
    train_loader, val_loader, num_classes = load_precomputed_dataset(augmentation=AUGMENTATION)
    print(f"data loaded: {num_classes} classes, {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples", flush=True)

    model = CnnSmallV1(num_classes=234).to(device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    n_train_batches = len(train_loader)
    log_every = max(1, n_train_batches // 10)
    curves = {"loss": [], "roc_auc_macro": []}
    for epoch in range(EPOCHS):
        print(f"epoch {epoch + 1}/{EPOCHS} starting ...", flush=True)
        model.train()
        epoch_losses = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                pct = 100.0 * (batch_idx + 1) / n_train_batches
                print(f"  epoch {epoch + 1} [{pct:5.1f}%] "
                      f"batch {batch_idx + 1}/{n_train_batches} "
                      f"loss={loss.item():.4f}", flush=True)

        print(f"epoch {epoch + 1}: running validation...", flush=True)
        model.eval()
        all_probs, all_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
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
        print(f"epoch {epoch + 1}/{EPOCHS} done: "
              f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}", flush=True)

    results = {
        "metrics": {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "loss": curves["loss"][-1],
        },
        "training_curves": curves,
        "duration_seconds": time.time() - time.time()
    }
    print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)
    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **exit_code:** 1
- **duration_seconds:** 17.09353725000983
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_090326_first_run/exp_018
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** ValueError
- **message:** raise ValueError(

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_090326_first_run/exp_018/code.py", line 46, in <module>
    loss = criterion(logits, y)
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/loss.py", line 834, in forward
    return F.binary_cross_entropy_with_logits(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/functional.py", line 3638, in binary_cross_entropy_with_logits
    raise ValueError(
ValueError: Target size (torch.Size([512, 206])) must be the same as input size (torch.Size([512, 234]))
```
