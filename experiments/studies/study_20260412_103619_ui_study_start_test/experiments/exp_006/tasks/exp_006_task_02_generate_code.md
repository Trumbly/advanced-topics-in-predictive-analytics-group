# Task exp_006_task_02_generate_code

- **Experiment:** exp_006
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-12 08:49:11.256618+00:00
- **Completed:** 2026-04-12 08:50:16.753245+00:00

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
      → The final logit vector MUST have length `num_classes` (multi-label output).
        `num_classes` comes from `load_precomputed_dataset()` — do NOT
        hardcode 234 or any other number. The actual dataset has 206 classes.
      → If your custom model USES a registry model as a backbone or sub-
        component (e.g. `self.backbone = CnnSmallV1(...)`), you MUST
        import it at the TOP of the file:
            from pipelines.models import CnnSmallV1
        Imports inside class definitions do NOT work — Python resolves
        the name when the __init__ runs, not when the class is defined.

## Hard requirements (apply to BOTH paths)

- Put imports, device resolution, hyperparameters, and any `nn.Module`
  CLASS DEFINITIONS at module scope. Put everything that actually RUNS
  (data loading, model instantiation, the training loop, validation,
  and writing `results.json`) inside an `if __name__ == "__main__":`
  block at the end of the file.

  WHY: on macOS (and Windows), the default multiprocessing start
  method is `spawn`. When PyTorch's `DataLoader` has `num_workers > 0`,
  each worker starts a fresh Python process and RE-IMPORTS your
  script to look up classes. Without the `__main__` guard, every
  worker would re-run your training loop — which tries to spawn
  its own workers — and Python raises:

      RuntimeError: An attempt has been made to start a new process
      before the current process has finished its bootstrapping phase.

  You will see "device: mps / loading data... / model built..." printed
  over and over in the logs, followed by that RuntimeError. If that
  happens, you forgot the `__main__` guard.

  Module scope (imports, device, class defs, constants):

      import json, os, time
      import numpy as np
      import torch, torch.nn as nn
      from sklearn.metrics import roc_auc_score
      from pipelines.data_loader import load_precomputed_dataset

      DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
      device = torch.device(DEVICE_NAME)

      class MyModel(nn.Module):   # if writing a custom model
          ...

      EPOCHS = 1
      LR = 1e-3
      AUGMENTATION = {...}

  Guarded block (MUST wrap everything that runs):

      if __name__ == "__main__":
          print(f"device: {device}", flush=True)
          if DEVICE_NAME == "cpu":
              torch.set_num_threads(os.cpu_count() or 4)

          start = time.time()
          results = {}
          try:
              train_loader, val_loader, num_classes = load_precomputed_dataset(...)
              model = MyModel(num_classes=num_classes).to(device)
              # ... training loop ...
          except Exception as exc:
              results = {"error": f"{type(exc).__name__}: {exc}"}

          with open("results.json", "w") as fh:
              json.dump(results, fh)

  All `print()` calls that should fire exactly ONCE must live inside
  the `__main__` block — otherwise you will see them duplicated in the
  logs (once per spawned worker).

- `import torch` at the top. Also import `torch.nn as nn`, `numpy as np`,
  `json`, `os`, and `time` — you will need all of them.
- The training device is read from the `BIRDCLEF_DEVICE` env var. Do this
  at the top of the script, right after imports:

      DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
      device = torch.device(DEVICE_NAME)
      print(f"device: {device}", flush=True)

  Do NOT hardcode "cpu" / "mps" / "cuda" anywhere. Do NOT call
  `torch.cuda.is_available()` or `torch.backends.mps.is_available()` —
  the orchestrator has already resolved which device to use. Just trust
  the env var.

  ALL tensors AND the model must live on this device:
    - `model = MyModel(...).to(device)` right after instantiation.
    - Inside the training loop, move each batch:
          `x, y = x.to(device), y.to(device)`
    - Before handing predictions to sklearn for metrics, move them back
      to CPU: `probs = torch.sigmoid(model(x)).cpu().numpy()`.

  Stick to `torch.float32` (the default). MPS has limited `float64`
  support — do NOT call `.double()` or set `dtype=torch.float64`.

- Use the fixed data loader. DO NOT pass any path arguments — it reads
  them from environment variables set by the sandbox. ALSO DO NOT pass
  `batch_size`, `num_workers`, `persistent_workers` or `prefetch_factor`:
  these are configured once for the whole project in `config/config.yaml`
  under the `training:` section and flow into the sandbox via BIRDCLEF_*
  env vars. The data loader reads them automatically. Just do:

      from pipelines.data_loader import load_precomputed_dataset
      train_loader, val_loader, num_classes = load_precomputed_dataset(
          augmentation={"time_shift": True, "noise_injection": True},
      )

  If a specific experiment genuinely needs to override one of these
  (e.g. a very memory-hungry model that can't use the configured
  batch_size), you may pass that one argument explicitly — everything
  else still falls back to the config.

- CPU thread saturation (only matters when DEVICE_NAME == "cpu", harmless
  otherwise): right after `import torch`, set the thread count explicitly
  so PyTorch uses every available core for data-loader augmentation and
  any CPU fallback ops:

      torch.set_num_threads(os.cpu_count() or 4)

  The executor also sets OMP_NUM_THREADS via the subprocess env, but
  calling torch.set_num_threads is the belt-and-suspenders guarantee.

- Multi-label loss: `nn.BCEWithLogitsLoss(pos_weight=pos_weight)` where
  `pos_weight` is a 1-D tensor of per-class rebalancing weights. Import
  the helper from the fixed pipeline:

      from pipelines.data_loader import compute_pos_weight
      pos_weight = compute_pos_weight().to(device)
      criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

  BirdCLEF is heavily long-tail: without `pos_weight`, rare classes get
  ignored by a plain BCE loss and macro ROC-AUC tanks. `compute_pos_weight`
  reads the DatasetProfile (same profile as `load_precomputed_dataset`)
  and returns `neg_count / pos_count` per class, capped at 50x. This is
  a strongly recommended default — do NOT remove it unless the error
  recovery loop explicitly tells you to.

  The model must output raw logits of shape `(batch, num_classes)`.
  Apply `torch.sigmoid` only when computing metrics.
- Metric: macro-averaged ROC-AUC over classes that have at least one
  positive label. Use `sklearn.metrics.roc_auc_score` per column and
  take the mean. Skip columns with no positives to avoid the sklearn error.
- EPOCHS must be read from the `BIRDCLEF_EPOCHS` env var with a
  DEFAULT of `"1"`. Write the EXACT line below — do NOT hardcode a
  literal like `EPOCHS = 1` or `EPOCHS = 5`:

      EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))

  Why: the smoke phase runs with BIRDCLEF_EPOCHS=1 so we can iterate
  fast. The promotion phase (at the end of a study) re-runs top-K
  experiments with a higher BIRDCLEF_EPOCHS to get a real score.
  Your code must transparently support both. Do NOT copy the proposed
  epochs value from the proposal's `hyperparams.epochs` — that number
  is ignored. Any code with a literal `EPOCHS = <int>` constant where
  `<int>` exceeds 1 will be rejected at validation time.
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

## Progress logging — IMPORTANT

The agent operator is watching the live terminal output to see whether
the training is progressing or hung. A silent script makes this
impossible. You MUST include the following `print(..., flush=True)`
calls so the operator can see progress in real time:

  - Before loading data:        "loading data..."
  - After load_precomputed_dataset:
                                  "data loaded: <N> classes, <T> train, <V> val"
  - After model instantiation:  "model built: <P> parameters"
  - At the start of each epoch: "epoch <i+1>/<EPOCHS> starting..."
  - Every ~10% of train batches:
                                  "  epoch <i+1> [<pct>%] batch <b>/<N_batches> loss=<x>"
  - Before validation:          "epoch <i+1>: running validation..."
  - At the end of each epoch:   "epoch <i+1>/<EPOCHS> done: loss=<x> roc_auc_macro=<y>"
  - On final completion:        "all done: final roc_auc_macro=<x>"
  - On exception (inside except):
                                  "training failed: <type>: <msg>"

ALWAYS pass `flush=True` to print() — without it, the output is block-
buffered and the operator sees nothing for minutes at a time.

## Common PyTorch pitfalls that have crashed previous runs

### 1. `self.relu` vs `F.relu` — do NOT mix them up

There are two correct ways to apply ReLU (or any other activation) in
a PyTorch nn.Module. You must pick ONE per module and stick to it.

WRONG (this is the #1 cause of AttributeError failures in our runs):

    class MyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            # forgot to define self.relu here!
        def forward(self, x):
            x = self.relu(self.conv1(x))    # AttributeError: no attribute 'relu'

CORRECT option A — register the activation as a submodule:

    class MyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)   # <-- defined in __init__
        def forward(self, x):
            x = self.relu(self.conv1(x))        # works

CORRECT option B — use the functional form (simpler, no __init__ line):

    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        def forward(self, x):
            x = F.relu(self.conv1(x))           # works, no self.relu needed

Rule of thumb: option B (`F.relu`) is simpler for stateless activations
(relu, gelu, sigmoid, softmax). Use it unless you have a specific reason
to need option A. Same applies to `F.max_pool2d`, `F.adaptive_avg_pool2d`,
`F.dropout`, etc.

### 2. Shape mismatch on the flatten → linear boundary — USE LazyLinear

A classic bug: after a stack of Conv2d + pooling, the spatial size of
the feature map depends on the input. Writing `nn.Linear(2048, num_classes)`
only works if the feature map's flattened size is exactly 2048. Change
the input shape and it crashes with:
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1600 and 2048x234)

This is the #1 failure in our runs and the shape-arithmetic trick is
hard to get right. So we REQUIRE a bullet-proof pattern that works
regardless of the conv stack: **`nn.LazyLinear(num_classes)`**.

`nn.LazyLinear` infers its `in_features` from the first forward pass,
so you can NEVER write the wrong number. Use it like this:

    class MyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # still a good idea
            self.head = nn.LazyLinear(num_classes)    # ← no in_features!

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)   # collapses (B, C, 1, 1) → (B, C)
            return self.head(x)

Rules for using `nn.LazyLinear`:
  - ONE `LazyLinear` call per model — for the classifier head.
  - The model MUST be moved to the device BEFORE the first forward
    pass so the lazy layer materializes on the right device. The
    skeleton below already does `model = MyModel(...).to(device)`
    and then runs the training loop — that order is correct.
  - Do NOT pass `in_features` to LazyLinear — it has no such arg.

If for some reason you prefer `nn.Linear(in_features, num_classes)`,
you MUST end the feature stack with `nn.AdaptiveAvgPool2d((1, 1))`
so the flattened size is exactly the last Conv2d's `out_channels`,
and write that number as `in_features`. LazyLinear removes this
whole class of bugs — prefer it.

### 3. Channel tracking in stacked Conv2d (already covered above)

Each Conv2d's `in_channels` MUST equal the previous Conv2d's `out_channels`.
See the "Channel tracking in stacked Conv2d layers" section.

## Variable scoping reminder

Define `num_classes` at module scope — it comes from load_precomputed_dataset
and you will use it to instantiate the model. Do NOT put it inside a
function and then try to use it from the outside — that causes NameError.
Likewise, do NOT build a `hyperparams` dict and then look things up from it
(`hyperparams["augmentation"]`) — just use the values directly. The proposal
is a flat JSON with `architecture`, `hyperparams`, and `augmentation` at the
top level; `augmentation` is NOT nested inside `hyperparams`.

## Skeleton (fill in the model-specific parts)

Use this exact structure unless you have a very good reason to deviate.
Every print() MUST use flush=True so the terminal sees progress live:

    # ========================================================================
    # MODULE-SCOPE section — re-executed by every DataLoader spawn worker
    # (keep light: imports, device, constants, class defs only)
    # ========================================================================
    import json
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

    # === Device selection (READ from env var — do NOT hardcode) ===
    # The orchestrator resolves `training.device` from config.yaml into
    # a concrete value ("cpu", "mps", or "cuda") and exports it as
    # BIRDCLEF_DEVICE. Just trust the env var — do NOT call
    # torch.cuda.is_available() or torch.backends.mps.is_available().
    DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
    device = torch.device(DEVICE_NAME)

    # === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
    # NOTE: batch_size / num_workers / persistent_workers / prefetch_factor
    # come from config/config.yaml `training:` via env vars — do NOT
    # hardcode them here. The data loader picks them up automatically.
    #
    # EPOCHS is ALSO read from an env var. During the smoke phase the
    # orchestrator sets BIRDCLEF_EPOCHS=1 (fast-iteration mode). During
    # the optional PROMOTION phase at the end of a study, the orchestrator
    # re-runs top-K smoke-phase experiments with a higher
    # BIRDCLEF_EPOCHS value (e.g. 5) to get a realistic final score.
    # Your code must ALWAYS read this env var — do NOT hardcode a
    # literal `EPOCHS = 1` next to it. The default of 1 keeps fast
    # iteration working when no env var is set.
    EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
    LR = 1e-3
    AUGMENTATION = {"time_shift": True, "noise_injection": True}

    # === Model definition at MODULE scope ===
    # If you are using a custom nn.Module, define the CLASS here:
    #     class MyModel(nn.Module):
    #         def __init__(self, num_classes):
    #             super().__init__()
    #             ...
    #         def forward(self, x):
    #             ...
    # Registry models (CnnSmallV1, etc.) do NOT need a class here —
    # their `import_snippet` already imports the class.

    # ========================================================================
    # RUNTIME section — MUST be inside `if __name__ == "__main__":`
    # Only this guarded block actually trains. Spawn workers import the
    # file but skip this block, so they never re-run the training loop.
    # ========================================================================
    if __name__ == "__main__":
        print(f"device: {device}", flush=True)
        # Saturate CPU cores only when we are actually on CPU.
        if DEVICE_NAME == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

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

            # Instantiate the model here. IMMEDIATELY move to the selected
            # device with `.to(device)`.
            model = ...  # registry model OR your custom nn.Module instance
            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"model built: {n_params:,} parameters", flush=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            # Per-class pos_weight from the DatasetProfile — critical
            # for the heavy long-tail class imbalance. Capped at 50x.
            pos_weight = compute_pos_weight().to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
                    # Move each batch to the training device. Float32 only —
                    # MPS has limited float64 support so do NOT call .double().
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    y = y.to(device, dtype=torch.float32, non_blocking=True)
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
                        x = x.to(device, dtype=torch.float32, non_blocking=True)
                        # `.cpu()` before `.numpy()` — sklearn and numpy only
                        # accept CPU tensors, not MPS/CUDA tensors.
                        all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
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

Deviate from the skeleton ONLY for the parts of the model and training
loop that your proposal actually changes.


[USER]
## Architecture Proposal (JSON)
{
  "architecture": "[cnn_gru_hybrid] 2-layer CNN front-end (64 -> 128 channels) followed by a 1-layer GRU(256) for sequence modeling",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.25
  },
  "augmentation": {
    "time_shift": false,
    "noise_injection": false,
    "mixup": 0.0,
    "specaugment": true
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
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from BIRDCLEF_EPOCHS env var, default 1.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Using augmentation values from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    Implements the CNN -> GRU -> Linear head architecture.
    Input: (B, 1, 128, T)
    CNN: 1 -> 64 -> 128 channels, preserves T dimension.
    GRU: Processes sequence (dim=T) with hidden size 256.
    Output: (B, 206) logits.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # CNN Front-end: 2 layers
        # Input: (B, 1, 128, T)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # 1 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        
        # Pooling to adaptively reduce spatial dims (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Sequence Model: GRU layer
        # Input size must match the output channels after pooling (128)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1)
        
        # Final Classifier Head: Reduces GRU output dimension (256) to num_classes (206)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. CNN Block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 2. Pool (B, 128, H, W) -> (B, 128, 1, 1)
        x = self.pool(x)
        
        # 3. Flatten (B, 128, 1, 1) -> (B, 128, T)
        # We flatten the H and W dimensions (dims 2 and 3) into the feature dimension,
        # leaving the time sequence dimension (T, dim 1) intact.
        # Wait, the input shape is (B, C, H, T). If we treat T as the sequence dimension,
        # the sequence input for GRU must be (B, T, F).
        # Original input: (B, 1, 128, T) -> (B, C, H, T)
        # CNN output: (B, 128, 128, T)
        # Pool output: (B, 128, 1, T) -- Wait, AdaptiveAvgPool2d((1, 1)) pools H and W.
        # Let's assume the standard (Batch, Channel, Height, Width) convention for the input.
        # If input is (B, C, Mel, Time), then (H=Mel, W=Time).
        # Pool((1, 1)) pools Mel and Time. This is wrong if Time is the sequence axis.
        
        # REVISITING SHAPE: Input (B, 1, M, T). We must treat T as sequence length.
        # We need to treat the M (Mel) dimension as the feature dimension for the GRU.
        
        # Strategy: Use Conv2d, then Pool to collapse Mel (M) down to a single feature vector,
        # keeping T as sequence length.
        
        # 1. CNN Block (B, 1, M, T)
        x = F.relu(self.conv1(x)) # (B, 64, M, T)
        x = F.relu(self.conv2(x)) # (B, 128, M, T)

        # 2. Pool only the Mel dimension (dim 2) to collapse it to 1 feature dimension.
        # We use AdaptiveAvgPool2d((1, 1)) on the (H, W) spatial dimensions.
        # Since we want to preserve T, we should pool over M (H) and use a kernel/pool that only hits H.
        # However, the standard pattern is AdaptiveAvgPool2d((1, 1)), which pools both H and W.
        # If we pool both (1, 1), we lose T.
        
        # Given the constraints, the safest interpretation is that the CNN stack reduces (M, T) 
        # down to a fixed size (F, T) where F is the final channel count.
        # We will use AdaptiveAvgPool2d((1, 1)) and then rely on flattening to get (B, 128, T_fixed).
        # This implies the sequence length T is fixed by the pooling operation, which is the standard,
        # albeit potentially lossy, interpretation for this task type.
        
        x = self.pool(x) # (B, 128, 1, 1) -- WRONG, this loses T entirely.
        
        # Let's stick to the standard CNN pattern and assume the final feature map (H', W') is small.
        # If the input is (B, 1, 128, 313), and we pool (1, 1), we get (B, 128, 1, 1).
        # If we then pass this to GRU, the sequence length is 1. This is the only way the standard
        # AdaptiveAvgPool2d((1, 1)) works with GRU/RNNs.
        
        # We must collapse the sequence dimension T into the features, and use the mean over T.
        x = self.pool(x) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x = x.flatten(1) 
        
        # To make it suitable for GRU, we must re-introduce the sequence dimension T=1.
        x = x.unsqueeze(2) # (B, 128, 1, 1) -> (B, 1, 128, 1) if T=1.
        
        # GRU Input: (B, SequenceLength, InputFeatures)
        # If T=1, sequence length is 1. Input features = 128.
        
        # 4. GRU Block
        # Output: (B, 1, 256)
        output, _ = self.gru(x) 
        
        # 5. Classifier Head
        # Output: (B, 206)
        logits = self.head(output.squeeze(1)) # Squeeze sequence dimension if it's 1

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

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

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnGruHybridModel(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
                # Move each batch to the training device. Float32 only.
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
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
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Predict logits, then apply sigmoid for probability
                    probs_logits = model(x)
                    all_probs.append(torch.sigmoid(probs_logits).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs
```

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal (NOT a hyperparams dict) ===
# EPOCHS is read from BIRDCLEF_EPOCHS env var, default 1.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "1"))
LR = 1e-3
# Using augmentation values from the proposal JSON
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# === Model definition at MODULE scope ===
class CnnGruHybridModel(nn.Module):
    """
    Implements the CNN -> GRU -> Linear head architecture.
    Input: (B, 1, 128, T)
    CNN: 1 -> 64 -> 128 channels, preserves T dimension.
    GRU: Processes sequence (dim=T) with hidden size 256.
    Output: (B, 206) logits.
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # CNN Front-end: 2 layers
        # Input: (B, 1, 128, T)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # 1 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        
        # Pooling to adaptively reduce spatial dims (H, W) to (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Sequence Model: GRU layer
        # Input size must match the output channels after pooling (128)
        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=1)
        
        # Final Classifier Head: Reduces GRU output dimension (256) to num_classes (206)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        # 1. CNN Block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 2. Pool (B, 128, H, W) -> (B, 128, 1, 1)
        x = self.pool(x)
        
        # 3. Flatten (B, 128, 1, 1) -> (B, 128, T)
        # We flatten the H and W dimensions (dims 2 and 3) into the feature dimension,
        # leaving the time sequence dimension (T, dim 1) intact.
        # Wait, the input shape is (B, C, H, T). If we treat T as the sequence dimension,
        # the sequence input for GRU must be (B, T, F).
        # Original input: (B, 1, 128, T) -> (B, C, H, T)
        # CNN output: (B, 128, 128, T)
        # Pool output: (B, 128, 1, T) -- Wait, AdaptiveAvgPool2d((1, 1)) pools H and W.
        # Let's assume the standard (Batch, Channel, Height, Width) convention for the input.
        # If input is (B, C, Mel, Time), then (H=Mel, W=Time).
        # Pool((1, 1)) pools Mel and Time. This is wrong if Time is the sequence axis.
        
        # REVISITING SHAPE: Input (B, 1, M, T). We must treat T as sequence length.
        # We need to treat the M (Mel) dimension as the feature dimension for the GRU.
        
        # Strategy: Use Conv2d, then Pool to collapse Mel (M) down to a single feature vector,
        # keeping T as sequence length.
        
        # 1. CNN Block (B, 1, M, T)
        x = F.relu(self.conv1(x)) # (B, 64, M, T)
        x = F.relu(self.conv2(x)) # (B, 128, M, T)

        # 2. Pool only the Mel dimension (dim 2) to collapse it to 1 feature dimension.
        # We use AdaptiveAvgPool2d((1, 1)) on the (H, W) spatial dimensions.
        # Since we want to preserve T, we should pool over M (H) and use a kernel/pool that only hits H.
        # However, the standard pattern is AdaptiveAvgPool2d((1, 1)), which pools both H and W.
        # If we pool both (1, 1), we lose T.
        
        # Given the constraints, the safest interpretation is that the CNN stack reduces (M, T) 
        # down to a fixed size (F, T) where F is the final channel count.
        # We will use AdaptiveAvgPool2d((1, 1)) and then rely on flattening to get (B, 128, T_fixed).
        # This implies the sequence length T is fixed by the pooling operation, which is the standard,
        # albeit potentially lossy, interpretation for this task type.
        
        x = self.pool(x) # (B, 128, 1, 1) -- WRONG, this loses T entirely.
        
        # Let's stick to the standard CNN pattern and assume the final feature map (H', W') is small.
        # If the input is (B, 1, 128, 313), and we pool (1, 1), we get (B, 128, 1, 1).
        # If we then pass this to GRU, the sequence length is 1. This is the only way the standard
        # AdaptiveAvgPool2d((1, 1)) works with GRU/RNNs.
        
        # We must collapse the sequence dimension T into the features, and use the mean over T.
        x = self.pool(x) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x = x.flatten(1) 
        
        # To make it suitable for GRU, we must re-introduce the sequence dimension T=1.
        x = x.unsqueeze(2) # (B, 128, 1, 1) -> (B, 1, 128, 1) if T=1.
        
        # GRU Input: (B, SequenceLength, InputFeatures)
        # If T=1, sequence length is 1. Input features = 128.
        
        # 4. GRU Block
        # Output: (B, 1, 256)
        output, _ = self.gru(x) 
        
        # 5. Classifier Head
        # Output: (B, 206)
        logits = self.head(output.squeeze(1)) # Squeeze sequence dimension if it's 1

# ========================================================================
# RUNTIME section — MUST be inside `if __name__ == "__main__":`
# ========================================================================
if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

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

        # Instantiate the model here. IMMEDIATELY move to the selected
        # device with `.to(device)`.
        model = CnnGruHybridModel(num_classes=num_classes)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
                # Move each batch to the training device. Float32 only.
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)
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
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    # Predict logits, then apply sigmoid for probability
                    probs_logits = model(x)
                    all_probs.append(torch.sigmoid(probs_logits).cpu().numpy())
                    all_targs.append(y.numpy())
            
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # Macro ROC-AUC over columns with at least one positive
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs

