# Task exp_003_task_07_validate_code

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 12:37:48.280522+00:00
- **Completed:** 2026-04-12 12:37:48.281914+00:00

## Code Used
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ProposalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        return self.fc2(nn.ReLU()(self.fc1(x)))

if __name__ == '__main__':
    input_dim = 10
    output_dim = 5
    model = ProposalNet(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X = torch.randn(100, input_dim)
    y = torch.randint(0, output_dim, (100,))

    for epoch in range(5):
        preds = model(X)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

```

## Output
- **validation:** passed
- **code_bytes:** 816
