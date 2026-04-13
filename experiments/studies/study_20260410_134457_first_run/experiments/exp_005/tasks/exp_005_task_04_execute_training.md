# Task exp_005_task_04_execute_training

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-10 13:48:51.421961+00:00
- **Completed:** 2026-04-10 13:48:52.744115+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
BATCH_SIZE = 64
EPOCHS = 30  # cap at 10
LR = 0.0001
AUGMENTATION = {"time_shift": False, "noise_injection": True}

# === Model definition ===
class MyModel(nn.Module):
    def __init__(self, num_classes=234):
        super().__init__()
        self.backbone = CnnSmallV1(num_classes=num_classes, in_channels=1)
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1, bias=False)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.class_dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.class_dropout(x)
        return x

model = MyModel(num_classes=234)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
criterion = nn.BCEWithLogitsLoss()

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    # Cap epochs
    EPOCHS = min(EPOCHS, 10)

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
- **exit_code:** 1
- **duration_seconds:** 1.3204684999655
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_134457_first_run/exp_005
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** NameError: name 'CnnSmallV1' is not defined

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_134457_first_run/exp_005/code.py", line 34, in <module>
    model = MyModel(num_classes=234)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260410_134457_first_run/exp_005/code.py", line 19, in __init__
    self.backbone = CnnSmallV1(num_classes=num_classes, in_channels=1)
                    ^^^^^^^^^^
NameError: name 'CnnSmallV1' is not defined
```
