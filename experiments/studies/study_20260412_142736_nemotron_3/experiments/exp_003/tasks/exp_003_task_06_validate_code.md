# Task exp_003_task_06_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-12 12:36:54.366818+00:00
- **Completed:** 2026-04-12 12:36:54.368442+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 0.001

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.scale = np.sqrt(dim)

    def forward(self, x):
        B, N, D = x.shape
        xyz = self.qkv(x).reshape(B, N, 3, D)
        xyz = xyz.permute(0, 2, 1, 3)  # B, 3, N, D
        attn_weights = (xyz @ xyz.transpose(1, 2)).squeeze(-1) / self.scale
        attn_weights = torch.softmax(attn_weights, dim=1)
        v = xyz[:, :, :, 2]  # V shape B, 3, N, D
        out = (attn_weights @ v).squeeze(-1)  # B, 3, N, D -> flatten to B, 3, N
        out = out.reshape(B, 3, N).mean(dim=1)  # average over 3 tokens? Actually we want N
        return out

class TwoConvWithSelfAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.self_attn = SelfAttention(64)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu2(self.bn2(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(x).flatten(1)  # (B, 64, 1, 1) -> (B, 64)
        x = self.self_attn(x)
        x = self.dropout(x)
        return self.head(x)

if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            augmentation={"time_shift": False, "noise_injection": False, "mixup": 0.2, "specaugment": False},
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        model = TwoConvWithSelfAttention(num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

        curves = {
            "loss": [],
            "roc_auc_macro": [],
            "cmap_at_5": [],
            "f1_macro": [],
        }
        for epoch in range(EPOCHS):
            print(f"epoch {epoch + 1}/{EPOCHS} starting ...", flush=True)
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y

```

## Output
- **validation:** failed

## Error
- **type:** SyntaxError
- **message:** Line 51: '(' was never closed
