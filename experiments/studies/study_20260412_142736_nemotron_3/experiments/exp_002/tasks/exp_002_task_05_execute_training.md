# Task exp_002_task_05_execute_training

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-12 12:31:37.897693+00:00
- **Completed:** 2026-04-12 12:31:39.169921+00:00

## Code Used
```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datasets
import torch.utils.data as transforms
import torch.nn.functional as F

def load_precomputed_dataset():
    # Placeholder for loading precomputed dataset
    dataset = datasets.TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
    return datasets.DataLoader(dataset, batch_size=32, shuffle=True)

def train():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.NLLLoss()
    for epoch in range(10):
        for x, y in load_precomputed_dataset():
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} complete')

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in load_precomputed_dataset():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f'Accuracy: {100.*correct/total:.2f}%')

if __name__ == "__main__":
    train()
    evaluate()

```

## Output
- **exit_code:** 1
- **duration_seconds:** 1.2711988339433447
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** NameError: name 'model' is not defined

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002/code.py", line 47, in <module>
    evaluate()
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002/code.py", line 34, in evaluate
    model.eval()
    ^^^^^
NameError: name 'model' is not defined
```
