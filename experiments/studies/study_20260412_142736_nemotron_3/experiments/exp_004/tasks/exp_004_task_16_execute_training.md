# Task exp_004_task_16_execute_training

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-12 12:45:12.202423+00:00
- **Completed:** 2026-04-12 12:45:24.130082+00:00

## Code Used
```python
import json
import torch
import torch.nn as nn
import json as json_mod
import os
import numpy as np
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lazy_linear = nn.LazyLinear(2)

    def forward(self, x):
        return self.lazy_linear(x)

EPOCHS = 1

hyperparams = {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.1
}

if __name__ == "__main__":
    dataset, num_classes = load_precomputed_dataset()[:2]
    pos_weight = compute_pos_weight().to(device)

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])

    print(f"Training start, epochs={EPOCHS}, batch_size={hyperparams['batch_size']}", flush=True)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}", flush=True)
        for batch in dataset:
            x = batch.x.to(device, dtype=torch.float32)
            y = batch.y.to(device, dtype=torch.float32)
            y_hat = torch.sigmoid(model(x))
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probs = []
            for batch in dataset:
                x = batch.x.to(device, dtype=torch.float32)
                p = torch.sigmoid(model(x)).cpu().numpy()
                probs.append(p)
            probs = np.concatenate(probs)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(probs, y.cpu().numpy())
            print(f"Validation AUC: {auc:.4f}", flush=True)

    results = {"error": None, "auc": auc, "epochs": EPOCHS}
    with open("results.json", "w") as f:
        json.dump(results, f)

```

## Output
- **exit_code:** 1
- **duration_seconds:** 11.926213000086136
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_004
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** libc++abi: terminating due to uncaught exception of type std::__1::system_error: Broken pipe

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_004/code.py", line 46, in <module>
    x = batch.x.to(device, dtype=torch.float32)
        ^^^^^^^
AttributeError: 'list' object has no attribute 'x'
libc++abi: terminating due to uncaught exception of type std::__1::system_error: Broken pipe
```
