# Task exp_003_task_03_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 14:21:37.692557+00:00
- **Completed:** 2026-04-10 14:21:37.695420+00:00

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
EPOCHS = 10  # capped at 10 as per requirement
LR = 0.0005
WEIGHT_DECAY = 0.001
DROPOUT = 0.2

# === Model definition ===
class MyModel(nn.Module):
    def __init__(self, num_classes=206, dropout=0.2, weight_decay=0.001):
        super(MyModel, self).__init__()
        # 3-conv CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # SE attention: compute per-channel mean and multiply
        self.se_attn = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)  # output weights
        )
        self.fc = nn.Linear(128, num_classes)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)

        # SE attention
        se_out = self.se_attn(x)
        se_out = se_out.view(x.size(0), 128, 1)
        x = x * se_out

        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# === Model definition ===
model = MyModel(num_classes=206, dropout=DROPOUT, weight_decay=WEIGHT_DECAY)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()

curves = {"loss": [], "roc_auc_macro": []}
start = time.time()
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

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
                probs = torch.sigmoid(model(x)).numpy()
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
- **validation:** passed
- **code_bytes:** 3581
