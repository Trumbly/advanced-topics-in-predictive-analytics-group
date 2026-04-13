# Task exp_003_task_03_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:56:38.560194+00:00
- **Completed:** 2026-04-10 13:56:38.561143+00:00

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

BATCH_SIZE = 128
LR = 0.0005
EPOCHS = 10
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.2, "specaugment": True}

train_loader, val_loader, num_classes = load_precomputed_dataset(
    batch_size=BATCH_SIZE,
    num_workers=0,
    augmentation=AUGMENTATION,
)

model = CnnSmallV1(num_classes=num_classes, in_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-05)
criterion = nn.BCEWithLogitsLoss()

curves = {"loss": [], "roc_auc_macro": []}
start = time.time()
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
- **validation:** passed
- **code_bytes:** 2118
