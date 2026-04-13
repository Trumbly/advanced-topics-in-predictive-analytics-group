# Task exp_003_task_04_execute_training

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-10 17:15:29.743742+00:00
- **Completed:** 2026-04-10 17:15:34.619961+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

start = time.time()
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.001
weight_decay = 0.0001
DROPOUT = 0.2
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.05, "specaugment": False}

print("loading data...", flush=True)
train_loader, val_loader, num_classes = load_precomputed_dataset(
    batch_size=BATCH_SIZE,
    num_workers=0,
    augmentation=AUGMENTATION,
)
print(
    f"data loaded: {num_classes} classes, "
    f"{len(train_loader.dataset)} train samples, "
    f"{len(val_loader.dataset)} val samples",
    flush=True,
)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.mean(dim=[2, 3])
        y = self.avgpool(y).squeeze(-1).squeeze(-1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.dropout(y)
        x = x * y
        return x

class ResNetStyle(nn.Module):
    def __init__(self, in_channels=1, num_classes=206):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.se = SEBlock(256)
        self.dropout = nn.Dropout(DROPOUT)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNetStyle(in_channels=1, num_classes=num_classes)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()

n_train_batches = len(train_loader)
log_every = max(1, n_train_batches // 10)

curves = {"loss": [], "roc_auc_macro": []}
for epoch in range(EPOCHS):
    print(
        f"epoch {epoch + 1}/{EPOCHS} starting "
        f"({n_train_batches} batches)...",
        flush=True,
    )
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
            all_probs

```

## Output
- **exit_code:** 1
- **duration_seconds:** 4.874974083038978
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_163024_first_run/exp_003
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** ValueError
- **message:** raise ValueError(f"Input dimension should be at least {len(out_size) + 1}")

```
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_163024_first_run/exp_003/code.py", line 68, in forward
    x = self.se(x)
        ^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_163024_first_run/exp_003/code.py", line 43, in forward
    y = self.avgpool(y).squeeze(-1).squeeze(-1)
        ^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/pooling.py", line 1510, in forward
    return F.adaptive_avg_pool2d(input, self.output_size)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/functional.py", line 1398, in adaptive_avg_pool2d
    _output_size = _list_with_default(output_size, input.size())
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/miniconda3/envs/birdclef/lib/python3.11/site-packages/torch/nn/modules/utils.py", line 41, in _list_with_default
    raise ValueError(f"Input dimension should be at least {len(out_size) + 1}")
ValueError: Input dimension should be at least 3
```
