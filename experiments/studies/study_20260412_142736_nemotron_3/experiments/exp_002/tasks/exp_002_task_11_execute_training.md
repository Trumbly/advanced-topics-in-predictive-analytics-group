# Task exp_002_task_11_execute_training

- **Experiment:** exp_002
- **Type:** predefined
- **Name:** execute_training
- **Status:** failed
- **Started:** 2026-04-12 12:33:37.915991+00:00
- **Completed:** 2026-04-12 12:33:40.321032+00:00

## Code Used
```python
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = 1
lr = 0.001
batch_size = 128
weight_decay = 0.0
dropout = 0.1

class TorchvisionAdapter(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        backbone = getattr(models, backbone_name)
        self.backbone = backbone(pretrained=False).features
        self.classifier = nn.LazyLinear(num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x
        if x.shape[2] == 28:
            x = T.ToTensor()(x)
            x = T.Resize((224, 224)).transform(x)
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def load_precomputed_dataset():
    dataset = datasets.TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
    data_loader = datasets.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=0, prefetch_factor=None)
    pos_weight = compute_pos_weight()
    return data_loader, pos_weight

if __name__ == "__main__":
    data_loader, pos_weight = load_precomputed_dataset()
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = TorchvisionAdapter("mobilenet_v3_small", num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc = 0.0
    all_probs = []
    all_labels = []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y in data_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
        epoch_loss /= len(data_loader)
        print(f'Epoch {epoch+1} loss: {epoch_loss:.4f}')
        if epoch == 0:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f}')
        else:
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)
            auc = roc_auc_score(labels, probs, average='macro')
            best_auc = max(best_auc, auc)
            print(f'Epoch {epoch+1} macro-ROC-AUC: {auc:.4f} (best: {best_auc:.4f})')
    results = {"macro_roc_auc": best_auc}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

```

## Output
- **exit_code:** 1
- **duration_seconds:** 2.403983124997467
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002
- **results_json_path:** None
- **timed_out:** False

## Error
- **type:** UnknownError
- **message:** NameError: name 'datasets' is not defined. Did you mean: 'dataset'?

```
Traceback (most recent call last):
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002/code.py", line 49, in <module>
    data_loader, pos_weight = load_precomputed_dataset()
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_142736_nemotron_3/exp_002/code.py", line 43, in load_precomputed_dataset
    dataset = datasets.TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
              ^^^^^^^^
NameError: name 'datasets' is not defined. Did you mean: 'dataset'?
```
