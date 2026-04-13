# Task exp_002_codegen_retry_01

- **Experiment:** exp_002
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-12 12:30:51.330074+00:00
- **Completed:** 2026-04-12 12:31:37.891228+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained Python script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Input tensor shape —...

[USER]
e (line 21) without an `if __name__ == "__main__":` guard. PyTorch DataLoader with num_workers > 0 uses spawn workers that re-import the script; without the guard each worker recursively spawns more workers and Python raises a bootstrapping RuntimeError. Move the training block (everything that actually RUNS, not class definitions) inside `if __name__ == "__main__":`.

Fix this specific issue and return the COMPLETE corrected script.
Return ONLY Python code — no explanations, no markdown fences.
```

## LLM Response
```
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
- **code:** import os
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

