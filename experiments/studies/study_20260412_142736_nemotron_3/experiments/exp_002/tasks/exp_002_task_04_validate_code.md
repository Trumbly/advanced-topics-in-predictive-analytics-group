# Task exp_002_task_04_validate_code

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 12:31:37.893219+00:00
- **Completed:** 2026-04-12 12:31:37.896129+00:00

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
- **validation:** passed
- **code_bytes:** 1366
