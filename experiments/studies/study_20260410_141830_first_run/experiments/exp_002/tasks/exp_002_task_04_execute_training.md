# Task exp_002_task_04_execute_training

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-10 14:20:34.478541+00:00
- **Completed:** 2026-04-10 14:20:45.194473+00:00

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
EPOCHS = 10
LR = 0.0005
WEIGHT_DECAY = 0.001

AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.2, "specaugment": True}

class ChannelSE(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(C, C, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(C, C, kernel_size=1)
        self.silu = nn.SiLU()
    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.silu(y)
        y = self.relu(y)
        return x + y

class MyModel(nn.Module):
    def __init__(self, num_classes=234, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.se = ChannelSE(128)
        self.dropout = nn.Dropout(0.1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128*128*313, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.se(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

train_loader, val_loader, num_classes = load_precomputed_dataset(
    batch_size=BATCH_SIZE,
    num_workers=0,
    augmentation=AUGMENTATION,
)

model = MyModel(num_classes=num_classes, in_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()

curves = {"loss": [], "roc_auc_macro": []}
start = time.time()
results = {}
try:
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
                probs = torch.sigmoid(logits).numpy()
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
        "metrics": {"roc_auc_macro": curves["roc_auc_macro"][-1], "loss": curves["loss"][-1]},
        "training_curves": curves,
        "duration_seconds": time.time() - start,
    }
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **exit_code:** 0
- **duration_seconds:** 10.714257458981592
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_141830_first_run/exp_002
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_141830_first_run/exp_002/results.json
- **timed_out:** False
