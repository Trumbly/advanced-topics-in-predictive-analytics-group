# Task exp_005_task_03_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:21:55.811550+00:00
- **Completed:** 2026-04-10 13:21:55.814275+00:00

## Code Used
```python
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import EfficientNetB0

try:
    train_loader, val_loader, num_classes = load_precomputed_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {
        "lr": 0.0001,
        "batch_size": 64,
        "epochs": 30,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "dropout": 0.2
    }
    model = EfficientNetB0(num_classes=206, pretrained=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses = []
    train_roc_aucs = []
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    elapsed = 0.0
    macro_auc_all = []

    for epoch in range(args["epochs"]):
        model.train()
        epoch_loss = 0.0
        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(specs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader.dataset))
        train_roc_aucs.append(0.0)

        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                labels = labels.to(device)
                logits = model(specs)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_auc = torchmetrics.MultiLabelAUCRate(num_classes, torch.tensor(all_preds).float(),
                                                torch.tensor(all_labels).float, average='macro').item()
        macro_auc_all.append(val_auc)
        train_losses[-1] = epoch_loss / len(train_loader.dataset)
        train_roc_aucs[-1] = val_auc

        if start:
            start.elapsed_time(end)
            elapsed = start.elapsed_time(end)
        else:
            elapsed = 0.0

    final_loss = train_losses[-1]
    final_auc = train_roc_aucs[-1]

    results = {
        "metrics": {"roc_auc_macro": final_auc, "loss": final_loss},
        "training_curves": {"loss": train_losses, "roc_auc_macro": train_roc_aucs},
        "duration_seconds": elapsed
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

except Exception as e:
    results = {"error": str(e), "metrics": {"roc_auc_macro": 0.0, "loss": 0.0},
               "training_curves": {"loss": [], "roc_auc_macro": []},
               "duration_seconds": 0.0}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

```

## Output
- **validation:** passed
- **code_bytes:** 3454
