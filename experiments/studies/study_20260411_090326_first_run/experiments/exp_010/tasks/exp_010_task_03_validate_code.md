# Task exp_010_task_03_validate_code

- **Experiment:** exp_010
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-11 10:15:57.316202+00:00
- **Completed:** 2026-04-11 10:15:57.320879+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

EPOCHS = 1
LR = 0.001
weight_decay = 0.001
dropout = 0.1

AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.2, "specaugment": False}

class Custom3ConvCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        self.h_attn = nn.Linear(256 * 128 * 313, 256)
        self.attn_proj = nn.Linear(256, 256)
        self.V_proj = nn.Linear(256, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        # flatten to (B, 128, 256*313)
        B, C, H, W = x.shape
        x = x.view(B, H, C * W)

        # self-attention
        Q = self.h_attn(x)
        K = self.h_attn(x)
        V = self.h_attn(x)
        S = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(256)
        S = torch.softmax(self.attn_proj(S), dim=-1)
        x = torch.matmul(S, V)  # (B, 128, 256)
        x = x.view(B, -1)

        x = self.global_pool(x).flatten(1)
        logits = self.head(x)
        return logits

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

        model = Custom3ConvCNN(num_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
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
- **validation:** passed
- **code_bytes:** 5813
