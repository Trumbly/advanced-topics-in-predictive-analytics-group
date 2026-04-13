# Task exp_002_task_03_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:28:31.505817+00:00
- **Completed:** 2026-04-10 13:28:31.506518+00:00

## Code Used
```python
import torch
from pipelines.data_loader import load_precomputed_dataset
import time
from CnnSmallV1 import CnnSmallV1

hyperparams = {
    "batch_size": 32,
    "epochs": 30,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "adam",
    "dropout": 0.1
}
epochs = 10 if hyperparams["epochs"] > 10 else hyperparams["epochs"]

train_loader, val_loader, num_classes = load_precomputed_dataset(
    profile_path="data/processed/dataset_profile.json",
    spectrograms_dir="data/processed/spectrograms",
    labels_csv="data/processed/labels.csv",
    batch_size=hyperparams["batch_size"],
    num_workers=0,
    augmentation=hyperparams["augmentation"]
)

model = CnnSmallV1(num_classes=num_classes)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

loss_history = []
roc_auc_history = []
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    n = 0
    for data in train_loader:
        x = data["spectrogram"]
        y = data["label"]
        logits = model(x)
        loss = criterion(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n += 1
    epoch_loss /= n
    loss_history.append(epoch_loss)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in val_loader:
            x = data["spectrogram"]
            y = data["label"]
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.float().cpu().numpy())
    val_preds = torch.cat(all_preds).numpy()
    val_labels = torch.cat(all_labels).numpy()

    roc_aus = []
    for i in range(num_classes):
        if (val_labels[:, i] == 0).all():
            continue
        roc_aus.append(roc_auc_score(val_labels[:, i], val_preds[:, i]))
    epoch_auc = sum(roc_aus) / len(roc_aus) if roc_aus else 0.0
    roc_auc_history.append(epoch_auc)
    model.train()

overall_loss = sum(loss_history) / len(loss_history)
overall_auc = sum(roc_auc_history) / len(roc_auc_history)

results = {
    "metrics": {"roc_auc_macro": overall_auc, "loss": overall_loss},
    "training_curves": {"loss": loss_history, "roc_auc_macro": roc_auc_history},
    "duration_seconds": time.time() - start_time
}

try:
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
except Exception as e:
    results = {"error": str(e)}
    with open("results.json", "w") as f:
        f.write(json.dumps(results))

```

## Output
- **validation:** passed
- **code_bytes:** 2673
