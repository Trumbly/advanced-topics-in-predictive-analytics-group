# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:46:02.977999+00:00
- **Completed:** 2026-04-10 13:46:02.980473+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

BATCH_SIZE = 64
EPOCHS = 10  # capped at 10
LR = 0.001
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.2, "specaugment": True}

train_loader, val_loader, num_classes = load_precomputed_dataset(
    batch_size=BATCH_SIZE,
    num_workers=0,
    augmentation=AUGMENTATION,
)

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv1d(out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        attn = self.attn(x)
        attn = attn.squeeze(1)
        attn = attn.unsqueeze(1)
        x = x * attn
        return self.proj(x)

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.ca = ChannelAttention(128, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.ca(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = MyModel(num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
- **code_bytes:** 3269
