# Task exp_005_task_03_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-10 13:31:05.838399+00:00
- **Completed:** 2026-04-10 13:31:05.840808+00:00

## Code Used
```python
import torch
import json
import time
from torch import nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

def main():
    start = time.time()
    try:
        hyperparams = {
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 20,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "dropout": 0.0
        }
        epochs = min(hyperparams["epochs"], 10)
        loss_fn = nn.BCEWithLogitsLoss()
        model = CnnSmallV1(num_classes=206)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
        augment = hyperparams["augmentation"]
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            profile_path="data/processed/dataset_profile.json",
            spectrograms_dir="data/processed/spectrograms",
            labels_csv="data/processed/labels.csv",
            batch_size=hyperparams["batch_size"],
            num_workers=0,
            augmentation=augment,
        )
        training_losses = []
        roc_aucs = []
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for spec, labels in train_loader:
                optimizer.zero_grad()
                logits = model(spec)
                loss = loss_fn(logits, labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_avg = epoch_loss / len(train_loader)
            training_losses.append(loss_avg)

            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for spec, labels in val_loader:
                    logits = model(spec)
                    all_logits.append(logits)
                    all_labels.append(labels.float())
            val_logits = torch.cat(all_logits)
            val_probs = torch.sigmoid(val_logits)
            val_bin = (val_probs > 0.5).float()
            per_class_roc = []
            for i in range(num_classes):
                pos_mask = val_bin[:, i] == 1
                if pos_mask.any():
                    per_class_roc.append(roc_auc_score(val_bin[:, i], val_probs[:, i]))
                else:
                    per_class_roc.append(0.0)
            roc_auc_macro = sum(per_class_roc) / len(per_class_roc) if per_class_roc else 0.0
            roc_aucs.append(roc_auc_macro)

        final_loss = training_losses[-1]
        final_roc = roc_aucs[-1]

        results = {
            "metrics": {"roc_auc_macro": final_roc, "loss": final_loss},
            "training_curves": {"loss": training_losses, "roc_auc_macro": roc_aucs},
            "duration_seconds": time.time() - start
        }
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        err = {"error": str(e)}
        with open("results.json", "w") as f:
            json.dump(err, f, indent=2)

if __name__ == "__main__":
    main()

```

## Output
- **validation:** passed
- **code_bytes:** 3099
