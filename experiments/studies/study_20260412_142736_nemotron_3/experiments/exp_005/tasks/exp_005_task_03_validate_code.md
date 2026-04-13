# Task exp_005_task_03_validate_code

- **Experiment:** exp_005
- **Type:** predefined
- **Name:** validate_code
- **Status:** failed
- **Started:** 2026-04-12 12:46:29.767311+00:00
- **Completed:** 2026-04-12 12:46:29.769436+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)
if DEVICE_NAME == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 0.001
train_loader, val_loader, num_classes = load_precomputed_dataset(augmentation={"time_shift": False, "noise_injection": False})
pos_weight = compute_pos_weight().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(pretrained=True).features
        self.backbone = backbone
        self.head = nn.Sequential(F.adaptive_avg_pool2d, nn.LazyLinear(num_classes))
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.flatten(1)
        return x

model = MobileNetV3Small(num_classes).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model built: {n_params:,} parameters", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
log_every = max(1, len(train_loader) // 10)
curves = {"loss": [], "roc_auc_macro": [], "cmap_at_5": [], "f1_macro": []}
start = time.time()
results = {}
try:
    print("loading data...", flush=True)
    print(f"data loaded: {num_classes} classes, {len(train_loader.dataset)} train, {len(val_loader.dataset)} val", flush=True)
    for epoch in range(EPOCHS):
        print(f"epoch {epoch+1}/{EPOCHS} starting ...", flush=True)
        model.train()
        epoch_losses = []
        for b, (x, y) in enumerate(train_loader):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if (b+1) % log_every == 0 or b+1 == len(train_loader):
                pct = 100*(b+1)/len(train_loader)
                print(f"  epoch {epoch+1} [{pct:5.1f}%] batch {b+1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        model.eval()
        all_probs, all_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                all_probs.append(probs)
                all_targs.append(y.numpy())
        probs = np.concatenate(all_probs, axis=0)
        targs = np.concatenate(all_targs, axis=0)
        aucs = []
        for c in range(targs.shape[1]):
            if targs[:,c].sum()>0:
                aucs.append(roc_auc_score(targs[:,c], probs[:,c]))
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        aps = []
        for c in range(targs.shape[1]):
            if targs[:,c].sum()>0:
                aps.append(average_precision_score(targs[:,c], probs[:,c]))
        val_cmap5 = float(np.mean(aps)) if aps else 0.0
        preds = (probs >= 0.5).astype(np.float32)
        val_f1 = float(f1_score(targs, preds, average='macro', zero_division=0))
        epoch_loss = float(np.mean(epoch_losses))
        curves["loss"].append(epoch_loss)
        curves["roc_auc_macro"].append(val_auc)
        curves["cmap_at_5"].append(val_cmap5)
        curves["f1_macro"].append(val_f1)
        print(f"epoch {epoch+1}/{EPOCHS} done: loss={epoch_loss:.4f} roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f} f1={val_f1:.4f}", flush=True)
    results = {"metrics": {"roc_auc_macro":curves["roc_auc_macro"][-1],"cmap_at_5":curves["cmap_at_5"][-1],"f1_macro":curves["f1_macro"][-1],"loss":curves["loss"][-1]},
              "training_curves":curves,
              "duration_seconds":time.time()-start}
    print(f"all done: roc_auc={curves['roc_auc_macro'][-1]:.4f} cmap@5={curves['cmap_at_5'][-1]:.4f} f1={curves['f1_macro'][-1]:.4f}", flush=True)
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"training failed: {type(exc).__name__}: {exc}", flush=True)
with open("results.json","w") as fh:
    json.dump(results,fh)

```

## Output
- **validation:** failed

## Error
- **type:** MissingMainGuard
- **message:** `load_precomputed_dataset(...)` called at module scope (line 18) without an `if __name__ == "__main__":` guard. PyTorch DataLoader with num_workers > 0 uses spawn workers that re-import the script; without the guard each worker recursively spawns more workers and Python raises a bootstrapping RuntimeError. Move the training block (everything that actually RUNS, not class definitions) inside `if __name__ == "__main__":`.
