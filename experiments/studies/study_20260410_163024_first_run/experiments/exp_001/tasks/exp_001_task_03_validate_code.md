# Task exp_001_task_03_validate_code

- **Experiment:** exp_001
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 16:30:47.015154+00:00
- **Completed:** 2026-04-10 16:30:47.017564+00:00

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
BATCH_SIZE = 32
EPOCHS = 2
LR = 0.001
AUGMENTATION = {"time_shift": True, "noise_injection": True}

# === Model definition ===
from pipelines.models import CnnSmallV1
model = CnnSmallV1(num_classes=206, in_channels=1)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

start = time.time()
results = {}
try:
    print("loading data...", flush=True)
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )
    print(
        f"data loaded: {num_classes} classes, "
        f"{len(train_loader.dataset)} train samples, "
        f"{len(val_loader.dataset)} val samples",
        flush=True,
    )

    # Build the model here — num_classes is now defined at MODULE scope
    model = CnnSmallV1(num_classes=num_classes, in_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    n_train_batches = len(train_loader)
    log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

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
                all_probs.append(torch.sigmoid(model(x)).numpy())
                all_targs.append(y.numpy())
        probs = np.concatenate(all_probs, axis=0)
        targs = np.concatenate(all_targs, axis=0)

        # Macro ROC-AUC over columns with at least one positive
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
- **code_bytes:** 4095
