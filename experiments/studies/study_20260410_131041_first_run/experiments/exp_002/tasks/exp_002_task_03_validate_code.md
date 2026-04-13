# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-10 13:11:37.349570+00:00
- **Completed:** 2026-04-10 13:11:37.350116+00:00

## Code Used
```python
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import import_snippet
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score
import json

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    model = import_snippet('resnet18', pretrained='resnet18', dropout=0.2)
    model = model(num_classes=num_classes)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_losses = []
    epoch_aucs = []
    start_time = time.time()

    for epoch in range(40):
        epoch_loss = 0.0
        model.train()
        for spec, y, y_hat in train_loader:
            optimizer.zero_grad()
            y_pred = model(spec)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for spec, y, y_hat in val_loader:
                y_pred = model(spec)
                all_preds.append(y_pred)
                all_targets.append(y)
        val_pred = torch.cat(all_preds)
        val_target = torch.cat(all_targets)
        val_loss = loss_fn(val_pred, val_target).item()
        macro_auc = roc_auc_score(val_target.numpy(), val_pred.numpy(), average='macro')
        epoch_aucs.append(macro_auc)
        scheduler.step()

    duration = time.time() - start_time
    final_metrics = {
        "roc_auc_macro": epoch_aucs[-1],
        "loss": epoch_losses[-1]
    }
    results = {
        "metrics": final_metrics,
        "training_curves": {"loss": epoch_losses, "roc_auc_macro": epoch_aucs},
        "duration_seconds": round(duration, 2)
    }

except Exception as e:
    results = {
        "error": str(e),
        "metrics": None,
        "training_curves": None,
        "duration_seconds": 0.0
    }

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

```

## Output
- **validation:** failed

## Error
- **type:** ForbiddenPattern
- **message:** Forbidden pattern found: 'eval('
