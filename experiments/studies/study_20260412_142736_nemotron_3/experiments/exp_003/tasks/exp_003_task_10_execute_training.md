# Task exp_003_task_10_execute_training

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-12 12:38:15.974245+00:00
- **Completed:** 2026-04-12 12:38:17.466430+00:00

## Code Used
```python
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1

class ProposalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.1)
        self.lazyhead = nn.LazyLinear(output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x.view(-1, self.lazyhead.in_features)
        x = self.lazyhead(x)
        return x

def load_precomputed_dataset():
    from pipelines.data_loader import load_precomputed_dataset
    # Returns dataset (X, y) and num_classes
    data, num_classes = load_precomputed_dataset()
    return data, num_classes

if __name__ == "__main__":
    data, num_classes = load_precomputed_dataset()
    input_dim = data.shape[1]  # assuming (N, input_dim)
    output_dim = num_classes

    model = ProposalNet(input_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight().to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    X, y = data
    y = y.float()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        preds = model(X.to(device))
        loss = criterion(preds, y.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X.to(device))).cpu().numpy()
        auc = roc_auc_score(y.numpy(), probs, multioutput='uniform_average')
    results = {"macro_auc": auc}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

```

## Output
- **exit_code:** 1
- **duration_seconds:** 1.49111087503843
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_003
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** ValueError
- **message:** ValueError: too many values to unpack (expected 2)

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_003/code.py", line 38, in <module>
    data, num_classes = load_precomputed_dataset()
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_003/code.py", line 34, in load_precomputed_dataset
    data, num_classes = load_precomputed_dataset()
    ^^^^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)
```
