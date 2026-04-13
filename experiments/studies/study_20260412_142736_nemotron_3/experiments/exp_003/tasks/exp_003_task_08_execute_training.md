# Task exp_003_task_08_execute_training

- **Experiment:** exp_003
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-12 12:37:48.283162+00:00
- **Completed:** 2026-04-12 12:37:49.242766+00:00

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
- **exit_code:** 0
- **duration_seconds:** 0.9584100419888273
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_003
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** NoResultsFile
- **message:** Script exited with status 0 but did not produce a results.json file in the working directory. Make sure your code writes `results.json` at the end of training.

```
1.7044872045516968
1.6948124170303345
1.6854249238967896
1.6763219833374023
1.6674928665161133
```
