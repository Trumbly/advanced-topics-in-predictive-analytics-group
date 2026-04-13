# Task exp_017_task_04_validate_code

- **Experiment:** exp_017
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-11 10:46:08.601438+00:00
- **Completed:** 2026-04-11 10:46:08.606034+00:00

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

torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Linear(64, 64)
        self.se_act = nn.Sigmoid()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x2 = self.relu2(self.conv2(x))
        x = self.pool2(x2)
        x3 = self.relu3(self.conv3(x))
        x = self.pool3(x3)
        x = x3 + x2
        x = self.se_act(self.se(x)).unsqueeze(-1).unsqueeze(-1)
        x = x * x3
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        for epoch in range(EPOCHS):
            model.train()
            for step, (imgs, y) in enumerate(train_loader):
                imgs = imgs.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = loss_fn(logits, y)
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}", flush=True)

            val_logits = [model(x).cpu().numpy() for x in val_loader]
            val_labels = [y.numpy() for _, y in val_loader]
            val_pred = (val_logits > 0.5).astype(int).flatten()
            val_true = val_labels.flatten()
            val_auc = roc_auc_score(val_true, val_logits.ravel())
            results['val_auc'] = val_auc
            print(f"val_auc: {val_auc:.4f}", flush=True)

        # Save results
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        results['error'] = str(e)
        print("error: failed, details:", results['error'], flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"training completed in {time.time() - start:.2f}s", flush=True)

```

## Output
- **validation:** passed
- **code_bytes:** 3646
