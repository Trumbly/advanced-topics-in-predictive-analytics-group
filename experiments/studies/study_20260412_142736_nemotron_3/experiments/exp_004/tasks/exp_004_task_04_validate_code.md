# Task exp_004_task_04_validate_code

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 12:41:24.553538+00:00
- **Completed:** 2026-04-12 12:41:24.555100+00:00

## Code Used
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class MyDataset(data.Dataset):
    def __init__(self, transform=None):
        self.dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform: img = self.transform(img)
        return img, label

def train():
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.01)
    train_set = MyDataset()
    loader = data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    for epoch in range(3):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()

```

## Output
- **validation:** passed
- **code_bytes:** 1246
