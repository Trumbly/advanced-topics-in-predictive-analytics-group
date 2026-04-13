# Task exp_001_task_08_execute_training

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-11 08:48:05.349738+00:00
- **Completed:** 2026-04-11 08:48:07.470901+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

torch.set_num_threads(os.cpu_count() or 4)

class CnnSmallV1(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(nn.AdaptiveAvgPool2d((8, 8))(self.conv2(self.relu(self.conv1(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

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
- **duration_seconds:** 2.1201087500085123
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_084632_first_run/exp_001
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** AttributeError: Can't pickle local object 'load_precomputed_dataset.<locals>._TorchAdapter'

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_084632_first_run/exp_001/code.py", line 61, in <module>
    for batch_idx, (x, y) in enumerate(train_loader):
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 495, in __iter__
    self._iterator = self._get_iterator()
                     ^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 433, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1192, in __init__
    w.start()
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/context.py", line 288, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
AttributeError: Can't pickle local object 'load_precomputed_dataset.<locals>._TorchAdapter'
```
