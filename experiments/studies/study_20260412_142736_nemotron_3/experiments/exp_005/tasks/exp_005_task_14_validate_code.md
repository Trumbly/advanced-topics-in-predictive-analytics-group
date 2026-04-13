# Task exp_005_task_14_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** completed
- **Started:** 2026-04-12 12:50:40.013139+00:00
- **Completed:** 2026-04-12 12:50:40.016427+00:00

## Code Used
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os
import json
import random
from torch.nn.functional import relu
from sklearn.metrics import roc_auc_score
import numpy as np

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
torch.set_num_threads(os.cpu_count() or 4) if DEVICE_NAME == "cpu" else None

EPOCHS = 1
lr = 0.001
batch_size = 128
weight_decay = 0.0
dropout = 0.1

def load_precomputed_dataset():
    from pipelines.data_loader import load_precomputed_dataset as load_fn
    train_loader, val_loader = load_fn()
    return train_loader, val_loader

class MNISTNet(nn.Module):
    def __init__(self, num_classes, dropout, device):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(num_classes)
        self.device = device

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = self.pool(relu(self.conv3(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def compute_pos_weight():
    from pipelines.data_loader import compute_pos_weight
    return compute_pos_weight()

def main():
    global device
    loader = load_precomputed_dataset()
    train_loader, val_loader = loader
    data = next(iter(train_loader))
    num_classes = data[1].shape[1]
    pos_weight = compute_pos_weight()
    model = MNISTNet(num_classes, dropout, device).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    results = {}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_total += inputs.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            true = labels.cpu().numpy()
            pred_labels = (preds > 0.5).astype(int)
            train_acc += np.mean(pred_labels == true)
        train_accuracy = train_acc / train_total if train_total > 0 else 0.0
        results['train_loss'] = train_loss / train_total
        results['train_acc'] = train_accuracy

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0
        val_auc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)
                pred = torch.sigmoid(outputs).cpu().numpy()
                true = labels.cpu().numpy()
                pred_labels = (pred > 0.5).astype(int)
                val_acc += np.mean(pred_labels == true)
                for label_idx in range(num_classes):
                    val_auc += roc_auc_score(true[:, label_idx], pred[:, label_idx], multi_class='binary', average='macro')
        val_auc = val_auc / num_classes if num_classes > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_acc / val_total if val_total > 0 else 0.0
        results['val_loss'] = val_loss
        results['val_acc'] = val_acc
        results['val_auc'] = val_auc

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {results['train_loss']:.4f} Acc: {results['train_acc']:.4f} | "
              f"Val Loss: {results['val_loss']:.4f} Acc: {results['val_acc']:.4f} | Val AUC: {results['val_auc']:.4f}", flush=True)

    results['num_classes'] = num_classes
    results['pos_weight'] = pos_weight
    results['macro_auc'] = val_auc
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

```

## Output
- **validation:** passed
- **code_bytes:** 4763
