# Task exp_010_task_08_validate_code

- **Experiment:** exp_010
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-11 10:17:44.641034+00:00
- **Completed:** 2026-04-11 10:17:44.643847+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

torch.set_num_threads(os.cpu_count() or 4)

train_loader, val_loader, num_classes = load_precomputed_dataset()

EPOCHS = 1

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.1)
        self.self_attn = torch.nn.Linear(128, 128)
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.self_attn(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    train_loss /= len(train_loader)
    print(f'Train loss {train_loss:.4f}', flush=True)

    model.eval()
    with torch.no_grad():
        val_outputs = model(val_loader[0])
        val_probs = F.sigmoid(val_outputs).cpu().numpy()
        val_labels = val_loader[1].numpy()
        auc = roc_auc_score(val_labels, val_probs)
    print(f'Val ROC-AUC {auc:.4f}', flush=True)

    results = {
        "train_loss": train_loss,
        "val_auc": auc,
        "epoch": epoch + 1,
        "error": None
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

print('Training finished, results.json written.', flush=True)

```

## Output
- **validation:** failed

## Error
- **type:** MissingMainGuard
- **message:** `load_precomputed_dataset(...)` called at module scope (line 14) without an `if __name__ == "__main__":` guard. PyTorch DataLoader with num_workers > 0 uses spawn workers that re-import the script; without the guard each worker recursively spawns more workers and Python raises a bootstrapping RuntimeError. Move the training block (everything that actually RUNS, not class definitions) inside `if __name__ == "__main__":`.
