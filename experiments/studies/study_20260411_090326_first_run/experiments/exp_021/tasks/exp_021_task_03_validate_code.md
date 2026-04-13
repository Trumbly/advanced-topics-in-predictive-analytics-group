# Task exp_021_task_03_validate_code

- **Experiment:** exp_021
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-11 11:11:08.308698+00:00
- **Completed:** 2026-04-11 11:11:08.313884+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

EPOCHS = 1
LR = 0.0005
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0, "specaugment": False}

class SEAttention2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg(x).view(b, c, 1, 1)
        y = self.conv(y).view(b, c, 1, 1)
        weight = torch.sigmoid(y)
        return x * weight

class MyModel(nn.Module):
    def __init__(self, num_classes=206):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=None).features
        self.backbone = nn.Sequential(*backbone)
        self.backbone[0] = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.se_blocks = []
        self.dropout = nn.Dropout(0.4)
        for i in range(0, 20, 5):
            self.se_blocks.append(SEAttention2d(self.backbone[i].out_channels))
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        for i in range(0, len(self.backbone), 5):
            x = self.se_blocks[i // 5](x)
        x = x.view(x.size(0), -1)
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
            augmentation=AUGMENTATION,
        )
        print(
            f"data loaded: {num_classes} classes, "
            f"{len(train_loader.dataset)} train samples, "
            f"{len(val_loader.dataset)} val samples",
            flush=True,
        )

        model = MyModel(num_classes=num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)

        curves = {"loss": [], "roc_auc_macro": []}
        for epoch in range(EPOCHS):
            print(
                f"epoch {epoch + 1}/{EPOCHS} starting "
                f"({n_train_batches} batches)...",
                flush=True,
            )
            model.train()
            epoch_losses = []
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

                if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                    pct = 100.0 * (batch_idx + 1) / n_train_batches
                    print(
                        f"  epoch {epoch + 1} [{pct:5.1f}%] "
                        f"batch {batch_idx + 1}/{n_train_batches} "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            print(f"epoch {epoch + 1}: running validation...", flush=True)
            model.eval()
            all_probs, all_targs = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    probs = torch.sigmoid(model(x)).cpu().numpy()
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
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
                flush=True,
            )

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}", flush=True)
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **validation:** passed
- **code_bytes:** 5498
