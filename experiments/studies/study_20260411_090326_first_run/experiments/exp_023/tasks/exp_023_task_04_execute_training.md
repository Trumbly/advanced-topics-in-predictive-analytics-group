# Task exp_023_task_04_execute_training

- **Experiment:** exp_023
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-11 11:14:57.933389+00:00
- **Completed:** 2026-04-11 11:15:15.461125+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": False, "mixup": 0.2, "specaugment": False}

class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.sigmoid(y)
        return x * y

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.ModuleList()
        # Block 1: 32 channels
        self.features.append(
            spectral_norm(nn.Conv2d(1, 32, 3, padding=1)))
        self.features.append(SEBlock(32))
        self.features.append(
            spectral_norm(nn.Conv2d(32, 32, 3, padding=1)))
        self.features.append(SEBlock(32))
        self.features.append(
            spectral_norm(nn.Conv2d(32, 32, 3, padding=1)))
        self.features.append(SEBlock(32))

        # Block 2: 64 channels
        self.features.append(
            spectral_norm(nn.Conv2d(32, 64, 3, padding=1)))
        self.features.append(SEBlock(64))
        self.features.append(
            spectral_norm(nn.Conv2d(64, 64, 3, padding=1)))
        self.features.append(SEBlock(64))
        self.features.append(
            spectral_norm(nn.Conv2d(64, 64, 3, padding=1)))
        self.features.append(SEBlock(64))

        # Block 3: 128 channels
        self.features.append(
            spectral_norm(nn.Conv2d(64, 128, 3, padding=1)))
        self.features.append(SEBlock(128))
        self.features.append(
            spectral_norm(nn.Conv2d(128, 128, 3, padding=1)))
        self.features.append(SEBlock(128))
        self.features.append(
            spectral_norm(nn.Conv2d(128, 128, 3, padding=1)))
        self.features.append(SEBlock(128))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x).flatten(1)
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

        model = MyModel(num_classes=num_classes)
        model = model.to(device)
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
                    targs = y.numpy()
                    all_probs.append(probs)
                    all_targs.append(targs)
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
        print(
            f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **exit_code:** 0
- **duration_seconds:** 17.525458458985668
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_090326_first_run/exp_023
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_090326_first_run/exp_023/results.json
- **timed_out:** False
