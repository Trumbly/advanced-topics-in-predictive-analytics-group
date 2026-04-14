# Task exp_003_task_02_generate_code

- **Experiment:** exp_003
- **Type:** llm
- **Name:** generate_code
- **Status:** completed
- **Started:** 2026-04-14 23:09:02.211474+00:00
- **Completed:** 2026-04-14 23:12:29.019392+00:00

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

      EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
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

              # --- CPU/GPU performance optimizations (free speedups) ---
              # channels_last: MKL-DNN is optimized for NHWC layout;
              # gives 10-30% speedup on CPU conv-heavy models, also
              # benefits MPS/CUDA. Apply to model AND input tensors.
              model = model.to(memory_format=torch.channels_last)

              # torch.compile: graph-mode optimization (PyTorch 2.0+).
              # Fuses ops, removes Python overhead. ~10-20% speedup.
              # Skip if torch version < 2.0.
              if hasattr(torch, "compile"):
                  model = torch.compile(model)
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

- **Performance optimizations** — these are FREE speedups with no accuracy
  loss. Apply them ALWAYS, for both CPU and GPU:

  1. **channels_last memory format** — convert the model and input tensors
     to NHWC layout. MKL-DNN (CPU) and cuDNN (GPU) are optimized for this:

         model = model.to(memory_format=torch.channels_last)
         # In the training loop, after moving to device:
         x = x.to(device, memory_format=torch.channels_last)

  2. **torch.compile** — PyTorch 2.0+ graph compilation. Fuses ops and
     eliminates Python overhead:

         if hasattr(torch, "compile"):
             model = torch.compile(model)

  3. **torch.inference_mode for validation** — faster than torch.no_grad
     because it also disables autograd view tracking:

         with torch.inference_mode():
             for x, y in val_loader:
                 x = x.to(device, memory_format=torch.channels_last)
                 ...

     Use `torch.inference_mode()` (not `torch.no_grad()`) in the
     validation loop.

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
- Metrics: compute THREE metrics per epoch:

  1. **macro-F1** (PRIMARY — this is the competition target):
     Do NOT use a fixed 0.5 threshold. Instead, optimize the
     threshold PER CLASS on the validation set to maximize F1.
     For each class with at least one positive, sweep thresholds
     from 0.05 to 0.95 in steps of 0.05 and pick the one that
     maximizes that class's F1. Then compute macro-averaged F1
     across all classes using those per-class optimal thresholds.

     Implementation pattern:

         from sklearn.metrics import f1_score as sk_f1

         best_thresholds = np.full(targs.shape[1], 0.5)
         for c in range(targs.shape[1]):
             if targs[:, c].sum() == 0:
                 continue
             best_f1_c = 0.0
             for thr in np.arange(0.05, 1.0, 0.05):
                 preds_c = (probs[:, c] >= thr).astype(np.float32)
                 f1_c = sk_f1(targs[:, c], preds_c, zero_division=0)
                 if f1_c > best_f1_c:
                     best_f1_c = f1_c
                     best_thresholds[c] = thr
         preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
         val_f1 = float(sk_f1(targs, preds_opt, average="macro", zero_division=0))

     Save `best_thresholds` to `best_thresholds.npy` whenever the
     model improves (alongside best_model.pt):

         np.save("best_thresholds.npy", best_thresholds)

  2. **macro-ROC-AUC**: `sklearn.metrics.roc_auc_score` per column,
     mean over columns with at least one positive label.
  3. **cmap@5** (class-mean average precision at k=5): for each class
     with at least one positive, compute average precision, then mean
     across classes.

  All three go into `results.json` under `metrics` and into
  `training_curves`. The `f1_macro` is the PRIMARY metric — the
  agent uses it to rank experiments and decide which is best.
- EPOCHS must be read from the `BIRDCLEF_EPOCHS` env var with a
  DEFAULT of `"4"`. Write the EXACT line below:

      EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))

  4 epochs with early stopping (patience=2) is the recommended setup.
  The model will stop earlier if validation F1 stops improving.
- **Learning rate scheduler** — ALWAYS add a cosine annealing scheduler
  for better convergence:

      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=EPOCHS
      )

  Call `scheduler.step()` at the END of each epoch (after validation).
  This is critical for reaching high F1 scores.
- **Early stopping** — stop training when validation F1 has not
  improved for `patience` consecutive epochs. This saves time when the
  model has already converged:

      PATIENCE = 2
      best_f1 = 0.0
      epochs_no_improve = 0

  At the end of each epoch, after computing `val_f1` (with optimized
  per-class thresholds):

      if val_f1 > best_f1:
          best_f1 = val_f1
          epochs_no_improve = 0
          # Save best model weights + thresholds for Kaggle submission
          torch.save(model.state_dict(), "best_model.pt")
          np.save("best_thresholds.npy", best_thresholds)
      else:
          epochs_no_improve += 1
      if epochs_no_improve >= PATIENCE:
          print(f"early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
          break

  **Model checkpoint** — Always save the best model weights to
  `best_model.pt` whenever `val_f1` improves (as shown above). This
  file is required for Kaggle submission. Also save a class mapping
  file so the submission notebook knows the class order:

      import json as _json
      _json.dump({
          "num_classes": num_classes,
          "class_ids": [s["class_id"] for s in
                        _json.load(open(os.environ.get(
                            "BIRDCLEF_DATASET_PROFILE",
                            "data/processed/dataset_profile.json"
                        )))["class_stats"]],
      }, open("class_mapping.json", "w"))

  Write this mapping file ONCE, right before the training loop starts.

  The final `results.json` should report the LAST epoch's metrics (not
  the best epoch's), since that is what the agent compares across runs.
- At the VERY END of the script, write `results.json` in the CURRENT
  directory. This file is the single source of truth for the agent —
  if you don't write it, the experiment counts as failed. Schema:

      {"metrics": {"f1_macro": <float>, "roc_auc_macro": <float>,
                   "cmap_at_5": <float>, "loss": <float>},
       "training_curves": {"loss": [...], "f1_macro": [...],
                           "roc_auc_macro": [...], "cmap_at_5": [...]},
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
  - At the end of each epoch:   "epoch <i+1>/<EPOCHS> done: loss=<x> f1_macro=<y>"
  - On final completion:        "all done: final f1_macro=<x>"
  - On exception (inside except):
                                  "training failed: <type>: <msg>"

ALWAYS pass `flush=True` to print() — without it, the output is block-
buffered and the operator sees nothing for minutes at a time.

## Hardware memory constraint — IMPORTANT

The operator's machine has limited GPU memory (Apple MPS with ~48 GB
shared). Models with more than ~5 million parameters risk OOM on MPS
when combined with a batch size of 512. Keep your model SMALL:
  - Custom CNNs: 3-4 conv blocks with 32→64→128 channels max.
  - Pretrained backbones: use small variants (EfficientNet-B0,
    MobileNetV3-Small, ResNet-18). Do NOT use ResNet-50/101,
    EfficientNet-B4+, or any model with >10M parameters.
  - If in doubt, prefer fewer channels and fewer layers.

## Common PyTorch pitfalls that have crashed previous runs

### 0. `device.type` is a PROPERTY, not a method — NO parentheses

`torch.device.type` is a plain string attribute:

    if device.type == 'cuda':    # CORRECT
    if device.type == 'mps':     # CORRECT

Writing `device.type()` with parentheses causes:
    TypeError: 'str' object is not callable

NEVER write `device.type()` — always use `device.type` (no parens).

### 0b. `nn.GRU` and `nn.LSTM` return TUPLES, not tensors

`nn.GRU(...)` and `nn.LSTM(...)` return `(output, hidden_state)`,
NOT a single tensor. You MUST unpack them:

    gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
    output, hidden = gru(x)       # CORRECT — unpack the tuple
    x = output                     # use the output tensor

WRONG — calling tensor methods on the tuple:

    x = gru(x).permute(...)       # AttributeError: 'tuple' has no 'permute'
    x = gru(x).transpose(...)     # AttributeError: 'tuple' has no 'transpose'

Always unpack: `output, _ = gru(x)` then operate on `output`.

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
    pass so the lazy layer materializes on the right device.
  - **CRITICAL**: After `.to(device)`, you MUST run a dummy forward
    pass to initialize the lazy parameters BEFORE calling
    `sum(p.numel() for p in model.parameters())`. Without this,
    `p.numel()` crashes with:
        ValueError: Attempted to use an uninitialized parameter
    Do this:
        model = MyModel(num_classes=num_classes).to(device)
        # Initialize lazy modules with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
  - Do NOT pass `in_features` to LazyLinear — it has no such arg.

If for some reason you prefer `nn.Linear(in_features, num_classes)`,
you MUST end the feature stack with `nn.AdaptiveAvgPool2d((1, 1))`
so the flattened size is exactly the last Conv2d's `out_channels`,
and write that number as `in_features`. LazyLinear removes this
whole class of bugs — prefer it.

### 3. Channel tracking in stacked Conv2d (already covered above)

Each Conv2d's `in_channels` MUST equal the previous Conv2d's `out_channels`.
See the "Channel tracking in stacked Conv2d layers" section.

### 4. Pretrained backbones expect 3 channels — use TorchvisionAdapter

Spectrograms have 1 channel. Pretrained torchvision models (EfficientNet,
MobileNetV3, ResNet) expect 3 channels. Do NOT import them directly:

    # WRONG — will crash with "expected 3 channels, got 1"
    import torchvision.models as models
    backbone = models.mobilenet_v3_small(pretrained=True)

Instead, use the registry adapter which handles 1→3 channel expansion
and resize-to-224 automatically:

    from pipelines.models import TorchvisionAdapter
    model = TorchvisionAdapter("mobilenet_v3_small", num_classes=num_classes)

Valid backbone names: "efficientnet_b0", "resnet18", "mobilenet_v3_small".

### 5. Output shape MUST be (batch, num_classes) — ALWAYS flatten

The model's final output must be exactly `(batch, num_classes)`. If your
model outputs `(batch, channels, num_classes)` or `(batch, time, num_classes)`,
`BCEWithLogitsLoss` will crash with:
    ValueError: Target size (torch.Size([B, C])) must be the same as
    input size (torch.Size([B, X, C]))

To fix: ALWAYS end your feature extractor with
`nn.AdaptiveAvgPool2d((1, 1))` followed by `.flatten(1)` before the
classifier head. This collapses spatial dims to `(batch, channels)`.

### 6. NEVER mock or stub the data loader

Do NOT write your own mock/fake version of `load_precomputed_dataset`.
Do NOT write `class MockLoader`, `class MockDataset`, or any fallback
that invents fake data. The real data loader is ALWAYS available in the
sandbox. Just import and call it:

    from pipelines.data_loader import load_precomputed_dataset
    train_loader, val_loader, num_classes = load_precomputed_dataset(
        augmentation=AUGMENTATION,
    )

If you see an ImportError for `pipelines`, your import is wrong — fix
the import, do NOT create a mock.

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
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
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
    # EPOCHS is read from an env var so the orchestrator can override it.
    # Default 4 — early stopping (patience=2) handles convergence.
    EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
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
            # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
            # Without this, p.numel() crashes on uninitialized parameters.
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 128, 313, device=device)
                model(dummy)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"model built: {n_params:,} parameters", flush=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS
            )
            # Per-class pos_weight from the DatasetProfile — critical
            # for the heavy long-tail class imbalance. Capped at 50x.
            pos_weight = compute_pos_weight().to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            n_train_batches = len(train_loader)
            log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

            curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
            PATIENCE = 2
            best_f1 = 0.0
            epochs_no_improve = 0
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

                # --- Metric 1: Macro ROC-AUC ---
                aucs = []
                for c in range(targs.shape[1]):
                    if targs[:, c].sum() > 0:
                        aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
                val_auc = float(np.mean(aucs)) if aucs else 0.0

                # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
                # For each class with positives, take the top-5 predictions by
                # score and compute average precision, then mean across classes.
                aps = []
                for c in range(targs.shape[1]):
                    if targs[:, c].sum() > 0:
                        aps.append(average_precision_score(targs[:, c], probs[:, c]))
                val_cmap5 = float(np.mean(aps)) if aps else 0.0

                # --- Metric 3: Macro F1 with per-class threshold optimization ---
                # Sweep thresholds per class to maximize F1 (critical for imbalanced data)
                best_thresholds = np.full(targs.shape[1], 0.5)
                for c in range(targs.shape[1]):
                    if targs[:, c].sum() == 0:
                        continue
                    best_f1_c = 0.0
                    for thr in np.arange(0.05, 1.0, 0.05):
                        preds_c = (probs[:, c] >= thr).astype(np.float32)
                        f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                        if f1_c > best_f1_c:
                            best_f1_c = f1_c
                            best_thresholds[c] = thr
                preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
                val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

                epoch_loss = float(np.mean(epoch_losses))

                curves["loss"].append(epoch_loss)
                curves["f1_macro"].append(val_f1)
                curves["roc_auc_macro"].append(val_auc)
                curves["cmap_at_5"].append(val_cmap5)
                print(
                    f"epoch {epoch + 1}/{EPOCHS} done: "
                    f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                    f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                    flush=True,
                )
                scheduler.step()

                # Early stopping on F1 (primary metric)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    epochs_no_improve = 0
                    # Save best model weights + thresholds for submission
                    torch.save(model.state_dict(), "best_model.pt")
                    np.save("best_thresholds.npy", best_thresholds)
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(
                        f"early stopping at epoch {epoch + 1} "
                        f"(no improvement for {PATIENCE} epochs)",
                        flush=True,
                    )
                    break

            results = {
                "metrics": {
                    "f1_macro": curves["f1_macro"][-1],
                    "roc_auc_macro": curves["roc_auc_macro"][-1],
                    "cmap_at_5": curves["cmap_at_5"][-1],
                    "loss": curves["loss"][-1],
                },
                "training_curves": curves,
                "duration_seconds": time.time() - start,
            }
            print(
                f"all done: f1={curves['f1_macro'][-1]:.4f} "
                f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
                f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
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
  "architecture": "[cnn_attention] 4-Conv block front-end (32->64->128) -> Temporal Self-Attention -> Global Average Pooling",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 4,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.1
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal ===
# EPOCHS is read from an env var so the orchestrator can override it.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
LR = 1e-3
# Override skeleton default augmentation with proposal values
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# =============================================================================
# CUSTOM MODULE DEFINITIONS (Module Scope)
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Implements a simplified self-attention mechanism over the temporal dimension.
    Input shape expected: (B, C, H, L) where L is the sequence length (time).
    We pool H to 1 first, so input is (B, C, 1, L).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections across the feature dimension (C=embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.tensor(1.0 / np.sqrt(self.head_dim), dtype=torch.float32)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, C, 1, L)
        B, C, _, L = x.size()
        
        # Flatten C and H dimensions together to get (B, C*1, L) for linear layers
        # We reshape to (B, C, L) for easier attention calculation
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, C, L)
        
        # Calculate Q, K, V
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # Scaled Dot-Product Attention: (B, L, C) @ (B, C, L) -> (B, L, L)
        # We transpose L and C to perform dot product across the sequence dimension
        attn_scores = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout and compute context vector
        context = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.output_proj(context)
        
        # Reshape back to (B, C, 1, L) format for the next block/pooling
        return output.view(B, C, 1, L)


class BirdCLEFModel(nn.Module):
    """
    CNN-Attention Model architecture as proposed.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # 1. CNN Front-end (32 -> 64 -> 128 channels)
        # Using padding=1 to preserve dimensions for as long as possible.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Out: 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # In: 32, Out: 64
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # In: 64, Out: 128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 2. Temporal Self-Attention
        self.attention = SelfAttentionBlock(embed_dim=128, num_heads=4, dropout=dropout_rate)
        
        # 3. Global Average Pooling (Collapse Height dimension to 1)
        # This keeps the time dimension (L) intact for the attention mechanism, 
        # which is necessary for temporal modeling.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classifier Head (LazyLinear handles arbitrary feature size)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # x shape: (B, 128, H, L) -> H=128, L=313
        
        # 2. Temporal Self-Attention
        # The attention block expects (B, C, 1, L) after pooling H to 1.
        x_pooled = self.pool(x) # (B, 128, 1, 1) -> ERROR: pool((B, 128, 128, 313)) is (B, 128, 1, 1)
                               # We need to pool H dimension (dim=2) to 1, but keep L dimension (dim=3).
        
        # Manual pooling to keep the time dimension:
        # (B, C, H, L) -> (B, C, 1, L) by averaging over H
        x_attn_input = x.mean(dim=2) 
        
        x_attn = self.attention(x_attn_input)
        
        # 3. Final Pooling and Classifier
        # The output of attention is (B, 128, 1, 313). We now pool H again (which is 1) 
        # and then flatten to (B, 128 * 1) before the linear layer.
        x_final = self.pool(x_attn) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x_flat = x_final.flatten(1) 
        
        # 4. Classification
        return self.head(x_flat)


if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Use the specified augmentation settings
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
        model = BirdCLEFModel(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Use typical input shape (1, 1, 128, 313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
        PATIENCE = 2
        best_f1 = 0.0
        epochs_no_improve = 0
        
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
                    # Compute probabilities: sigmoid(logits)
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 with per-class threshold optimization ---
            best_thresholds = np.full(targs.shape[1], 0.5)
            for c in range(targs.shape[1]):
                if targs[:, c].sum() == 0:
                    continue
                best_f1_c = 0.0
                # Sweep thresholds from 0.05 to 0.95 in steps of 0.05
                for thr in np.arange(0.05, 1.0, 0.05):
                    preds_c = (probs[:, c] >= thr).astype(np.float32)
                    f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                    if f1_c > best_f1_c:
                        best_f1_c = f1_c
                        best_thresholds[c] = thr
            preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["f1_macro"].append(val_f1)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                flush=True,
            )
            scheduler.step()

            # Early stopping on F1 (primary metric)
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                # Save best model weights + thresholds for submission
                torch.save(model.state_dict(), "best_model.pt")
                np.save("best_thresholds.npy", best_thresholds)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

        results = {
            "metrics": {
                "f1_macro": curves["f1_macro"][-1],
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: f1={curves['f1_macro'][-1]:.4f} "
            f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

```

## Output
- **code:** import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pipelines.data_loader import load_precomputed_dataset, compute_pos_weight

# === Device selection (READ from env var — do NOT hardcode) ===
DEVICE_NAME = os.environ.get("BIRDCLEF_DEVICE", "cpu")
device = torch.device(DEVICE_NAME)

# === Hardcode hyperparameters from the proposal ===
# EPOCHS is read from an env var so the orchestrator can override it.
EPOCHS = int(os.environ.get("BIRDCLEF_EPOCHS", "4"))
LR = 1e-3
# Override skeleton default augmentation with proposal values
AUGMENTATION = {"time_shift": False, "noise_injection": False, "mixup": 0.0, "specaugment": True}

# =============================================================================
# CUSTOM MODULE DEFINITIONS (Module Scope)
# =============================================================================

class SelfAttentionBlock(nn.Module):
    """
    Implements a simplified self-attention mechanism over the temporal dimension.
    Input shape expected: (B, C, H, L) where L is the sequence length (time).
    We pool H to 1 first, so input is (B, C, 1, L).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Q, K, V projections across the feature dimension (C=embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.tensor(1.0 / np.sqrt(self.head_dim), dtype=torch.float32)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (B, C, 1, L)
        B, C, _, L = x.size()
        
        # Flatten C and H dimensions together to get (B, C*1, L) for linear layers
        # We reshape to (B, C, L) for easier attention calculation
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, C, L)
        
        # Calculate Q, K, V
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # Scaled Dot-Product Attention: (B, L, C) @ (B, C, L) -> (B, L, L)
        # We transpose L and C to perform dot product across the sequence dimension
        attn_scores = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout and compute context vector
        context = torch.matmul(attn_weights, V)
        
        # Output projection
        output = self.output_proj(context)
        
        # Reshape back to (B, C, 1, L) format for the next block/pooling
        return output.view(B, C, 1, L)


class BirdCLEFModel(nn.Module):
    """
    CNN-Attention Model architecture as proposed.
    Input: (B, 1, 128, 313)
    """
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        
        # 1. CNN Front-end (32 -> 64 -> 128 channels)
        # Using padding=1 to preserve dimensions for as long as possible.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Out: 32
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # In: 32, Out: 64
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # In: 64, Out: 128
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 2. Temporal Self-Attention
        self.attention = SelfAttentionBlock(embed_dim=128, num_heads=4, dropout=dropout_rate)
        
        # 3. Global Average Pooling (Collapse Height dimension to 1)
        # This keeps the time dimension (L) intact for the attention mechanism, 
        # which is necessary for temporal modeling.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classifier Head (LazyLinear handles arbitrary feature size)
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x):
        # 1. CNN Stack
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        
        # x shape: (B, 128, H, L) -> H=128, L=313
        
        # 2. Temporal Self-Attention
        # The attention block expects (B, C, 1, L) after pooling H to 1.
        x_pooled = self.pool(x) # (B, 128, 1, 1) -> ERROR: pool((B, 128, 128, 313)) is (B, 128, 1, 1)
                               # We need to pool H dimension (dim=2) to 1, but keep L dimension (dim=3).
        
        # Manual pooling to keep the time dimension:
        # (B, C, H, L) -> (B, C, 1, L) by averaging over H
        x_attn_input = x.mean(dim=2) 
        
        x_attn = self.attention(x_attn_input)
        
        # 3. Final Pooling and Classifier
        # The output of attention is (B, 128, 1, 313). We now pool H again (which is 1) 
        # and then flatten to (B, 128 * 1) before the linear layer.
        x_final = self.pool(x_attn) # (B, 128, 1, 1)
        
        # Flatten: (B, 128, 1, 1) -> (B, 128)
        x_flat = x_final.flatten(1) 
        
        # 4. Classification
        return self.head(x_flat)


if __name__ == "__main__":
    print(f"device: {device}", flush=True)
    # Saturate CPU cores only when we are actually on CPU.
    if DEVICE_NAME == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    start = time.time()
    results = {}
    try:
        print("loading data...", flush=True)
        # Use the specified augmentation settings
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
        model = BirdCLEFModel(num_classes=num_classes, dropout_rate=0.1)
        model = model.to(device)
        
        # Initialize lazy modules (nn.LazyLinear) with a dummy forward pass.
        with torch.no_grad():
            # Use typical input shape (1, 1, 128, 313)
            dummy = torch.zeros(1, 1, 128, 313, device=device)
            model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model built: {n_params:,} parameters", flush=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
        # Per-class pos_weight from the DatasetProfile — critical
        pos_weight = compute_pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        n_train_batches = len(train_loader)
        log_every = max(1, n_train_batches // 10)  # ~10 progress lines/epoch

        curves = {"loss": [], "f1_macro": [], "roc_auc_macro": [], "cmap_at_5": []}
        PATIENCE = 2
        best_f1 = 0.0
        epochs_no_improve = 0
        
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
                    # Compute probabilities: sigmoid(logits)
                    all_probs.append(torch.sigmoid(model(x)).cpu().numpy())
                    all_targs.append(y.numpy())
            probs = np.concatenate(all_probs, axis=0)
            targs = np.concatenate(all_targs, axis=0)

            # --- Metric 1: Macro ROC-AUC ---
            aucs = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aucs.append(roc_auc_score(targs[:, c], probs[:, c]))
            val_auc = float(np.mean(aucs)) if aucs else 0.0

            # --- Metric 2: cmap@5 (class-mean average precision at k=5) ---
            aps = []
            for c in range(targs.shape[1]):
                if targs[:, c].sum() > 0:
                    aps.append(average_precision_score(targs[:, c], probs[:, c]))
            val_cmap5 = float(np.mean(aps)) if aps else 0.0

            # --- Metric 3: Macro F1 with per-class threshold optimization ---
            best_thresholds = np.full(targs.shape[1], 0.5)
            for c in range(targs.shape[1]):
                if targs[:, c].sum() == 0:
                    continue
                best_f1_c = 0.0
                # Sweep thresholds from 0.05 to 0.95 in steps of 0.05
                for thr in np.arange(0.05, 1.0, 0.05):
                    preds_c = (probs[:, c] >= thr).astype(np.float32)
                    f1_c = float(f1_score(targs[:, c], preds_c, zero_division=0))
                    if f1_c > best_f1_c:
                        best_f1_c = f1_c
                        best_thresholds[c] = thr
            preds_opt = (probs >= best_thresholds[np.newaxis, :]).astype(np.float32)
            val_f1 = float(f1_score(targs, preds_opt, average="macro", zero_division=0))

            epoch_loss = float(np.mean(epoch_losses))

            curves["loss"].append(epoch_loss)
            curves["f1_macro"].append(val_f1)
            curves["roc_auc_macro"].append(val_auc)
            curves["cmap_at_5"].append(val_cmap5)
            print(
                f"epoch {epoch + 1}/{EPOCHS} done: "
                f"loss={epoch_loss:.4f} f1={val_f1:.4f} "
                f"roc_auc={val_auc:.4f} cmap@5={val_cmap5:.4f}",
                flush=True,
            )
            scheduler.step()

            # Early stopping on F1 (primary metric)
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                # Save best model weights + thresholds for submission
                torch.save(model.state_dict(), "best_model.pt")
                np.save("best_thresholds.npy", best_thresholds)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch + 1} "
                    f"(no improvement for {PATIENCE} epochs)",
                    flush=True,
                )
                break

        results = {
            "metrics": {
                "f1_macro": curves["f1_macro"][-1],
                "roc_auc_macro": curves["roc_auc_macro"][-1],
                "cmap_at_5": curves["cmap_at_5"][-1],
                "loss": curves["loss"][-1],
            },
            "training_curves": curves,
            "duration_seconds": time.time() - start,
        }
        print(
            f"all done: f1={curves['f1_macro'][-1]:.4f} "
            f"roc_auc={curves['roc_auc_macro'][-1]:.4f} "
            f"cmap@5={curves['cmap_at_5'][-1]:.4f}",
            flush=True,
        )
    except Exception as exc:
        results = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"training failed: {type(exc).__name__}: {exc}", flush=True)

    with open("results.json", "w") as fh:
        json.dump(results, fh)

