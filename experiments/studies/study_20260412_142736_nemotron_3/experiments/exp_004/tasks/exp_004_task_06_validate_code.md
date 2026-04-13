# Task exp_004_task_06_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 12:42:29.051399+00:00
- **Completed:** 2026-04-12 12:42:29.054933+00:00

## Code Used
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os
import json
import numpy as np

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.LazyLinear(num_classes)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

def load_precomputed_dataset():
    from pipelines.data_loader import load_precomputed_dataset
    return load_precomputed_dataset()

def compute_pos_weight():
    from pipelines.data_loader import compute_pos_weight
    return compute_pos_weight()

# hyperparams
EPOCHS = 1
lr = 0.001
batch_size = 128
optimizer = optim.Adam
weight_decay = 0.0
dropout = 0.1

def train():
    model = MyModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = load_precomputed_dataset()
    num_workers = os.getenv("BIRDCLEF_NUM_WORKERS", "0").strip() or 2
    prefetch_factor = os.getenv("BIRDCLEF_PREFACTOR", "2").strip() or 2
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=True, prefetch_factor=prefetch_factor)

    pos_weight = compute_pos_weight().to(device)
    results = {
        "loss": 0.0,
        "auc": 0.0,
        "epochs": 0,
        "hyperparams": {"lr": lr, "batch_size": batch_size, "epochs": EPOCHS,
                        "optimizer": "adam", "weight_decay": weight_decay, "dropout": dropout}
    }

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        results["loss"] = epoch_loss / len(dataset)
        results["epochs"] = epoch + 1

        # validation AUC
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, dtype=torch.float32)
                yb = yb.to(device, dtype=torch.long)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        from sklearn.metrics import roc_auc_score
        results["auc"] = roc_auc_score(all_labels, all_logits)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {results['loss']:.4f} - AUC: {results['auc']:.4f}", flush=True)
        model.train()

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    train()

```

## Output
- **validation:** passed
- **code_bytes:** 3182
