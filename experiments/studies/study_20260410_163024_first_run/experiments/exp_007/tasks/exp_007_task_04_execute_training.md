# Task exp_007_task_04_execute_training

- **Experiment:** exp_007
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-10 19:01:10.815280+00:00
- **Completed:** 2026-04-10 19:01:18.231968+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
weight_decay = 0.0001
dropout = 0.1
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.2, "specaugment": False}

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

class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv1d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, c, 1)
        y = self.relu(self.fc(y))
        y = self.sigmoid(y)
        return x * y

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.se = SE(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.se(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = MyModel(num_classes)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()

n_train_batches = len(train_loader)
log_every = max(1, n_train_batches // 10)

curves = {"loss": [], "roc_auc_macro": []}
start = time.time()
try:
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
                print(f"  epoch {epoch + 1} [{pct:5.1f}%] "
                      f"batch {batch_idx + 1}/{n_train_batches} "
                      f"loss={loss.item():.4f}", flush=True)

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
        print(f"epoch {epoch + 1}/{EPOCHS} done: "
              f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}", flush=True)

    results = {
        "metrics": {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "loss": curves["loss"][-1],
        },
        "training_curves": curves,
        "duration_seconds": time.time() - start,
    }
    print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **exit_code:** 0
- **duration_seconds:** 7.4147926670266315
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_163024_first_run/exp_007
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_163024_first_run/exp_007/results.json
- **timed_out:** False
