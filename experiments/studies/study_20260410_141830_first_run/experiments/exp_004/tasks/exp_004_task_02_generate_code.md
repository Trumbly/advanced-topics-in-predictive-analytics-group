# Task exp_004_task_02_generate_code

- **Experiment:** exp_004
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-10 14:21:58.455210+00:00
- **Completed:** 2026-04-10 14:22:38.027379+00:00

## Prompt Used
```
[SYSTEM]
You are a Python ML engineer. Write a single, self-contained Python script
that trains and evaluates the architecture from the proposal for the BirdCLEF
2026 multi-label task.

## Input tensor shape — READ THIS FIRST

The data loader produces tensors of shape **(batch, 1, n_mels, time_frames)** —
a 4-dimensional tensor, image-like. Typical values: (32, 1, 128, 313).
This means:
  - Use `nn.Conv2d`, NOT `nn.Conv1d`. Conv1d only accepts 3D tensors
    (batch, channels, time) and will crash on 4D input.
  - Use `nn.MaxPool2d` / `nn.AdaptiveAvgPool2d`, not the 1D versions.
  - Treat each spectrogram as an image with 1 channel.

## Channel tracking in stacked Conv2d layers

The FIRST Conv2d must have `in_channels=1` because the input has 1 channel.
Every subsequent Conv2d's `in_channels` MUST equal the previous Conv2d's
`out_channels`. A common LLM mistake is:

    self.conv1 = nn.Conv2d(1, 64, 3)    # out: 64
    self.conv2 = nn.Conv2d(1, 64, 3)    # WRONG: expects 1 channel, gets 64

The correct version:

    self.conv1 = nn.Conv2d(1, 64, 3)    # out: 64
    self.conv2 = nn.Conv2d(64, 128, 3)  # in=64 matches conv1 out=64

Before writing each Conv2d, think about what tensor flows into it and
explicitly set `in_channels` to the previous layer's `out_channels`.
If you are unsure, use `nn.AdaptiveAvgPool2d((1, 1))` to collapse the
spatial dims and a `nn.Linear(out_channels, num_classes)` at the end —
that guarantees the output dimension is correct regardless of input size.

## Architecture sources — TWO valid paths

The proposal's `architecture` field tells you what to build:

(A) REGISTRY MODEL — the `architecture` value matches a name in the Model
    Registry (e.g. "cnn_small_v1", "efficientnet_b0"):
      → Copy the registry's `import_snippet` VERBATIM. Use the EXACT
        capitalization shown there (e.g. `CnnSmallV1`, NOT `cnn_small_v1`).
      → Registry models are in `pipelines.models` or in `torchvision.models`.
        The snippet tells you which. Do NOT invent imports like
        `from torchvision.models import CnnSmallV1` — that is wrong;
        CnnSmallV1 lives in `pipelines.models`.

(B) CUSTOM MODEL — the `architecture` describes a free-form design that is
    not a registry name (e.g. "ResNet-style 4-block CNN with SE attention"):
      → Define the full `nn.Module` class(es) inline. Everything must be
        imported from `torch`, `torch.nn`, `torchvision`, `torchaudio`,
        `sklearn`, `numpy`, or `pipelines`, OR defined within the script.
      → The final logit vector MUST have length 234 (multi-label output).
      → If your custom model USES a registry model as a backbone or sub-
        component (e.g. `self.backbone = CnnSmallV1(...)`), you MUST
        import it at the TOP of the file:
            from pipelines.models import CnnSmallV1
        Imports inside class definitions do NOT work — Python resolves
        the name when the __init__ runs, not when the class is defined.

## Hard requirements (apply to BOTH paths)

- Write code at MODULE scope, not inside a main() function. The script is
  executed top-to-bottom as a standalone Python file.
- `import torch` at the top. Also import `torch.nn as nn`, `numpy as np`,
  `json`, and `time` — you will need all of them.
- CPU-only. Do NOT reference `torch.cuda`, `.cuda()`, `.to('cuda')`, or any
  GPU-specific API.
- Use the fixed data loader. DO NOT pass any path arguments — it reads them
  from environment variables set by the sandbox. Just do:

      from pipelines.data_loader import load_precomputed_dataset
      train_loader, val_loader, num_classes = load_precomputed_dataset(
          batch_size=32,                  # from proposal
          num_workers=0,
          augmentation={"time_shift": True, "noise_injection": True},  # hardcoded dict
      )

- Multi-label loss: `nn.BCEWithLogitsLoss()`. The model must output raw
  logits of shape `(batch, num_classes)`. Apply `torch.sigmoid` only when
  computing metrics.
- Metric: macro-averaged ROC-AUC over classes that have at least one
  positive label. Use `sklearn.metrics.roc_auc_score` per column and
  take the mean. Skip columns with no positives to avoid the sklearn error.
- Respect the proposed `epochs` value but CAP at 10 if larger.
- At the VERY END of the script, write `results.json` in the CURRENT
  directory. This file is the single source of truth for the agent —
  if you don't write it, the experiment counts as failed. Schema:

      {"metrics": {"roc_auc_macro": <float>, "loss": <float>},
       "training_curves": {"loss": [...], "roc_auc_macro": [...]},
       "duration_seconds": <float>}

  If training fails, still write `results.json` with an `error` key:

      {"error": "<exception message>"}

- Do NOT import: subprocess, urllib, requests, socket.
- Do NOT call `os.system(...)`, `eval(...)`, `exec(...)`, or `__import__`.
- Do NOT download anything from the internet.
- Do NOT write files outside the current directory.

## Variable scoping reminder

Define `num_classes` at module scope — it comes from load_precomputed_dataset
and you will use it to instantiate the model. Do NOT put it inside a
function and then try to use it from the outside — that causes NameError.
Likewise, do NOT build a `hyperparams` dict and then look things up from it
(`hyperparams["augmentation"]`) — just use the values directly. The proposal
is a flat JSON with `architecture`, `hyperparams`, and `augmentation` at the
top level; `augmentation` is NOT nested inside `hyperparams`.

## Skeleton (fill in the model-specific parts)

Use this exact structure unless you have a very good reason to deviate:

    import json
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from pipelines.data_loader import load_precomputed_dataset

    # === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
    BATCH_SIZE = 32
    EPOCHS = 3                 # cap at 10
    LR = 1e-3
    AUGMENTATION = {"time_shift": True, "noise_injection": True}

    # === Model definition ===
    # EITHER (A): from pipelines.models import CnnSmallV1
    #             model = CnnSmallV1(num_classes=234, in_channels=1)
    # OR     (B): class MyModel(nn.Module): ...
    #             model = MyModel(num_classes=234)

    start = time.time()
    results = {}
    try:
        train_loader, val_loader, num_classes = load_precomputed_dataset(
            batch_size=BATCH_SIZE,
            num_workers=0,
            augmentation=AUGMENTATION,
        )

        # Build the model here — num_classes is now defined at MODULE scope
        model = ...  # registry model OR your custom nn.Module instance

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        curves = {"loss": [], "roc_auc_macro": []}
        for epoch in range(EPOCHS):
            model.train()
            epoch_losses = []
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

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

        results = {
            "metrics": {
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}

    with open("results.json", "w") as fh:
        json.dump(results, fh)

Deviate from the skeleton ONLY for the parts of the model and training
loop that your proposal actually changes.


[USER]
## Architecture Proposal (JSON)
{
  "architecture": "2-conv residual CNN with channel-wise SE attention and global pooling",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 30,
    "optimizer": "adam",
    "weight_decay": 0.001,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": false,
    "mixup": 0.2,
    "specaugment": false
  }
}

## Dataset Profile (summary)
- num_classes: 206
- num_samples: 233101
- spectrogram_shape: (1, 128, 313)
- sample_rate: 32000 Hz
- imbalance_ratio: 6854.00 (min=1, max=6854)
- split: stratified_kfold (seed=42)
- train/val: 186478/46623
- most-populated classes:
    - coffal1: 6854
    - rufnig1: 4907
    - bncfly: 4630
- least-populated classes:
    - 116570: 1
    - 23724: 1
    - 23150: 3

## Your Task
Write a complete, runnable Python training script implementing the
architecture above. Use the skeleton in the system prompt as your structure.
Fill in the model definition and any proposal-specific training changes.

Return ONLY the Python code. No explanations, no markdown fences, no prose.

```

## LLM Response
```
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}
num_classes = 206

class MyModel(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    model = MyModel(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    curves = {"loss": [], "roc_auc_macro": []}
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        all_probs, all_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_probs.append(torch.sigmoid(model(x)).numpy())
                all_targs.append(y.numpy())
        probs = np.concatenate(all_probs, axis=0)
        targs = np.concatenate(all_targs, axis=0)

        aucs = []
        for c in range(targs.shape[1]):
            if targs[:, c].sum() > 0:
                aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        epoch_loss = float(np.mean(epoch_losses))

        curves["loss"].append(epoch_loss)
        curves["roc_auc_macro"].append(val_auc)

    results = {
        "metrics": {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "loss": curves["loss"][-1],
        },
        "training_curves": curves,
        "duration_seconds": time.time() - start,
    }
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}

with open("results.json", "w") as fh:
    json.dump(results, fh)
```

## Output
- **code:** import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
AUGMENTATION = {"time_shift": True, "noise_injection": True}
num_classes = 206

class MyModel(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

start = time.time()
results = {}
try:
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        batch_size=BATCH_SIZE,
        num_workers=0,
        augmentation=AUGMENTATION,
    )

    model = MyModel(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    curves = {"loss": [], "roc_auc_macro": []}
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        all_probs, all_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_probs.append(torch.sigmoid(model(x)).numpy())
                all_targs.append(y.numpy())
        probs = np.concatenate(all_probs, axis=0)
        targs = np.concatenate(all_targs, axis=0)

        aucs = []
        for c in range(targs.shape[1]):
            if targs[:, c].sum() > 0:
                aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        epoch_loss = float(np.mean(epoch_losses))

        curves["loss"].append(epoch_loss)
        curves["roc_auc_macro"].append(val_auc)

    results = {
        "metrics": {
            "roc_auc_macro": curves["roc_auc_macro"][-1],
            "loss": curves["loss"][-1],
        },
        "training_curves": curves,
        "duration_seconds": time.time() - start,
    }
except Exception as exc:
    results = {"error": f"{type(exc).__name__}: {exc}"}

with open("results.json", "w") as fh:
    json.dump(results, fh)

