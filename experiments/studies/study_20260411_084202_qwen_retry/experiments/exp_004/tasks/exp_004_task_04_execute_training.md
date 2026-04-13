# Task exp_004_task_04_execute_training

- **Experiment:** exp_004
- **Type:** predefined
- **Name:** execute_training
- **Status:** completed
- **Started:** 2026-04-11 08:46:23.497504+00:00
- **Completed:** 2026-04-11 08:46:25.681448+00:00

## Code Used
```python
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

# Saturate every CPU core for matrix ops
torch.set_num_threads(os.cpu_count() or 4)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
# come from config/config.yaml `training:` via env vars — do NOT
# hardcode them here. The data loader picks them up automatically.
EPOCHS = 1                 # CAP at 1 — fast-iteration mode
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True, "mixup": 0.2, "specaugment": True}

# === Model definition ===
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H, W)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q * (self.head_dim ** -0.5)).transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        return x

class DynamicFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.attention = nn.Conv2d(out_channels, 1, 1)
        
    def forward(self, x):
        x = self.conv(x)
        attention_weights = torch.sigmoid(self.attention(x))
        return x * attention_weights

class Custom4BlockCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            MultiHeadSelfAttention(32, num_heads=4),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            MultiHeadSelfAttention(64, num_heads=4),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            MultiHeadSelfAttention(128, num_heads=4),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            MultiHeadSelfAttention(256, num_heads=4),
        )
        
        self.dynamic_filter = DynamicFilter(256, 256)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.dynamic_filter(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

# === Model definition ===
model = Custom4BlockCNN(num_classes=206, dropout=0.3)

start = time.time()
results = {}
try:
    print("loading data...", flush=True)
    # batch_size / num_workers / persistent_workers / prefetch_factor
    # are read from BIRDCLEF_* env vars (sourced from config.yaml).
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        augmentation=AUGMENTATION,
    )
    print(
        f"data loaded: {num_classes} classes, "
        f"{len(train_loader.dataset)} train samples, "
        f"{len(val_loader.dataset)} val samples",
        flush=True,
    )

    # Build the model here — num_classes is now defined at MODULE scope
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    n_train_batches = len(train_loader)
    log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

    curves = {"loss": [], "roc_auc_macro": []}
    for epoch in range(EPOCHS):
        print(
            f"epoch {epoch + 1}/{EPOCHS} starting "
            f"({n_train_batches} batches)...",
            flush=True,
        )
        model.train()
        epoch_losses = []
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_train_batches:
                pct = 100.0 * (batch_idx + 1) / n_train_batches
                print(
                    f"  epoch {epoch + 1} [{pct:5.1f}%] "
                    f"batch {batch_idx + 1}/{n_train_batches} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        print(f"epoch {epoch + 1}: running validation...", flush=True)
        model.eval()
        all_probs, all_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_probs.append(torch.sigmoid(model(x)).numpy())
                all_targs.append(y.numpy())
        probs = np.concatenate(all_probs, axis=0)
        targs = np.concatenate(all_targs, axis=0)

        # Macro ROC-AUC over columns with at least one positive
        aucs = []
        for c in range(targs.shape[1]):
            if targs[:, c].sum() > 0:
                aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        epoch_loss = float(np.mean(epoch_losses))

        curves["loss"].append(epoch_loss)
        curves["roc_auc_macro"].append(val_auc)
        print(
            f"epoch {epoch + 1}/{EPOCHS} done: "
            f"loss={epoch_loss:.4f} roc_auc_macro={val_auc:.4f}",
            flush=True,
        )

    results = {
        "metrics": {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "loss": curves["loss"][-1],
        },
        "training_curves": curves,
        "duration_seconds": time.time() - start,
    }
    print(
        f"all done: final roc_auc_macro={curves['roc_auc_macro'][-1]:.4f}",
        flush=True,
    )
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

with open("results.json", "w") as fh:
    json.dump(results, fh)

```

## Output
- **exit_code:** 0
- **duration_seconds:** 2.182760459021665
- **workdir:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_084202_qwen_retry/exp_004
- **results_json_path:** /Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/sandbox/study_20260411_084202_qwen_retry/exp_004/results.json
- **timed_out:** False
