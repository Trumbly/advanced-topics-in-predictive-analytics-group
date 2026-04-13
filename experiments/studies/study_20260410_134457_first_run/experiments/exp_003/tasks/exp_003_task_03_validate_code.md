# Task exp_003_task_03_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:46:49.255736+00:00
- **Completed:** 2026-04-10 13:46:49.256753+00:00

## Code Used
```python
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# Hardcode hyperparameters from the proposal (NOT a hyperparams dict)
BATCH_SIZE = 128
EPOCHS = min(50, 10)  # cap at 10 per requirement
LR = 0.001
optimizer_name = "Adam"
weight_decay = 0.0001
dropout = 0.2
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# Model definition
class FourBlockCNN(nn.Module):
    def __init__(self, num_classes=206):
        super(FourBlockCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)  # (128,313) -> (64,157)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)  # (64,157) -> (32,77)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)  # (32,77) -> (16,39)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)  # (16,39) -> (8,19)
        # Global average pooling over the time dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # (8,19) -> (8,1)
        self.flatten = nn.Flatten()
        # SE attention block (simple projection + residual)
        self.se_attn = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)  # (batch, 8, 1)
        x = self.flatten(x)   # (batch, 8)
        x = self.flatten(x).view(-1, 256)  # (batch, 256)
        x = self.se_attn(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# Model, optimizer, criterion
model = FourBlockCNN(num_classes=234)  # dataset has 234 classes (note: hyperparams said 206, but dataset summary says 234)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()

# Data loading
train_loader, val_loader, num_classes = load_precomputed_dataset(
    batch_size=BATCH_SIZE,
    num_workers=0,
    augmentation=AUGMENTATION,
)

# Training
start = time.time()
results = {}
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
    losses = np.mean(epoch_losses)
    curves["loss"].append(losses)

    model.eval()
    all_probs, all_targs = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(logits := model(x)).numpy()
            all_probs.append(probs)
            all_targs.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    targs = np.concatenate(all_targs, axis=0)

    aucs = []
    for c in range(targs.shape[1]):
        if targs[:, c].sum() > 0:
            aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
    val_auc = float(np.mean(aucs)) if aucs else 0.0
    curves["roc_auc_macro"].append(val_auc)

results = {
    "metrics": {
        "roc_auc_macro": curves["roc_auc_macro"][-1],
        "loss": curves["loss"][-1],
    },
    "training_curves": curves,
    "duration_seconds": time.time() - start,
}

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **validation:** passed
- **code_bytes:** 3553
