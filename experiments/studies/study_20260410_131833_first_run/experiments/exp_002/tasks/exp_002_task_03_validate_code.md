# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:19:37.436511+00:00
- **Completed:** 2026-04-10 13:19:37.438518+00:00

## Code Used
```python
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Any

from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from sklearn.metrics import roc_auc_score
import numpy as np

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb, reduction='mean')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
        preds = torch.softmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())
    epoch_loss /= len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy().ravel()
    macro_auc = roc_auc_score(labels, preds, average='macro')
    return epoch_loss, macro_auc, preds, labels

def main():
    try:
        start = time.time()
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cnn_small_v1(num_classes).to(device)
        optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_fn = BCEWithLogitsLoss()
        train_curves = []
        for epoch in range(1, 101):
            train_loss, train_auc, _, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            train_curves.append({'loss': train_loss, 'roc_auc_macro': train_auc})
        final_loss = train_curves[-1]['loss']
        final_auc = train_curves[-1]['roc_auc_macro']
        duration = time.time() - start
        results = {
            "metrics": {"roc_auc_macro": float(final_auc), "loss": float(final_loss)},
            "training_curves": [{"loss": float(item['loss']), "roc_auc_macro": float(item['roc_auc_macro'])} for item in train_curves],
            "duration_seconds": float(duration)
        }
        with open('results.json', 'w') as f:
            json.dump(results, f)
    except Exception as e:
        results = {"error": str(e)}
        with open('results.json', 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()

```

## Output
- **validation:** passed
- **code_bytes:** 2450
