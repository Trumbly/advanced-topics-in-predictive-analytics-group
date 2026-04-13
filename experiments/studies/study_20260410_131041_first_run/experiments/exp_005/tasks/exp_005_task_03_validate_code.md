# Task exp_005_task_03_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-10 13:13:18.766535+00:00
- **Completed:** 2026-04-10 13:13:18.768333+00:00

## Code Used
```python
import json
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset
from pipelines.models import cnn_small_v1

def main():
    try:
        train_loader, val_loader, num_classes = load_precomputed_dataset()
        hyperparams = {
            "lr": 5e-05,
            "batch_size": 256,
            "epochs": 60,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "dropout": 0.1
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cnn_small_v1(num_classes=num_classes, drop_rate=hyperparams["dropout"]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )
        train_losses = []
        val_losses = []
        val_aucs = []
        start_time = time.time()
        for epoch in range(1, hyperparams["epochs"] + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch["spectrogram"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))

            model.eval()
            val_loss_sum = 0.0
            all_pred = []
            all_true = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["spectrogram"].to(device)
                    y = batch["labels"].to(device)
                    logits = model(x)
                    val_loss_sum += criterion(logits, y).item()
                    probs = torch.sigmoid(logits)
                    pred = (probs > 0.5).float().cpu()
                    all_pred.append(pred)
                    all_true.append(y.cpu())
            val_losses.append(val_loss_sum / len(val_loader))
            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            val_aucs.append(roc_auc_score(y_true, y_pred))

            scheduler.step(val_losses[-1])

        metrics = {
            "roc_auc_macro": val_aucs[-1],
            "loss": val_losses[-1]
        }
        training_curves = {
            "loss": train_losses,
            "roc_auc_macro": val_aucs
        }
        duration_seconds = time.time() - start_time
        result = {"metrics": metrics, "training_curves": training_curves, "duration_seconds": duration_seconds}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)

    except Exception as e:
        result = {"error": str(e), "metrics": None, "training_curves": None, "duration_seconds": 0.0}
        with open("results.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

```

## Output
- **validation:** failed

## Error
- **type:** ForbiddenPattern
- **message:** Forbidden pattern found: 'eval('
