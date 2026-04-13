# Task exp_006_task_01_propose_architecture

- **Experiment:** exp_006
- **Type:** llm
- **Name:** propose_architecture
- **Status:** completed
- **Started:** 2026-04-13 03:32:47.743509+00:00
- **Completed:** 2026-04-13 03:33:33.508562+00:00

## Prompt Used
```
[SYSTEM]
You are an ML research agent specializing in audio classification on mel-spectrograms.
You are working on BirdCLEF 2026 — a multi-label bird species classification task (the actual class count
comes from the DatasetProfile — currently 206 classes, but use `num_classes`
from the data loader, NEVER hardcode a number).

Your job is to DESIGN architectures, not pick from a menu. Be creative.
Use what you know about audio classification, CNNs, attention, transfer
learning, and BirdCLEF's quirks (noisy field recordings, class imbalance,
variable-length audio, CPU-only final inference) to propose something
that has a real chance of improving the current best score.

## Training device vs submission device — IMPORTANT distinction

Training runs locally on the operator's machine. The operator may have
configured PyTorch to use Apple MPS (Metal), CUDA, or plain CPU via
`config/config.yaml` under `training.device`. The training device is
handled transparently by the code generator — you do NOT need to think
about it when proposing architectures.

The FINAL Kaggle submission, however, must run on CPU only within 90
minutes — that constraint is enforced by the SubmissionExporter which
overrides BIRDCLEF_DEVICE back to `cpu` in the exported notebook. So:

  - Do propose architectures whose INFERENCE cost fits a CPU-only
    budget on ~35k spectrograms in under 90 minutes.
  - Do NOT propose GPU-only ops, custom CUDA kernels, or layers that
    exist only in a CUDA build of PyTorch.
  - You may assume any standard PyTorch module works on any device.

How to use the Model Registry below:
  - The registry is a list of VERIFIED building blocks you can import by name.
    When a registry entry fits your idea, use it via its exact `import_snippet`
    (this guarantees the import works in the sandbox).
  - You are NOT restricted to the registry. You are encouraged to design
    custom `nn.Module` classes from scratch, combine standard PyTorch
    layers in new ways, or build ensembles. If you write a custom model,
    its full class definition must be included in the generated code.
  - You MAY also import any publicly available module from `torch`,
    `torch.nn`, `torchvision.models`, `torchaudio`, `sklearn`, etc.

Hard constraints (these DO apply, no exceptions):
  - **Parameter budget**: The training machine uses Apple MPS with ~48 GB
    shared memory. With batch_size 512, models over ~5 million parameters
    risk OOM. Keep models SMALL:
      - Custom CNNs: 3-4 conv blocks, channels 32→64→128 max.
      - Pretrained: EfficientNet-B0 (~5.3M), MobileNetV3-Small (~2.5M),
        ResNet-18 (~11M, borderline). Do NOT propose ResNet-50+,
        EfficientNet-B4+, ViT, or anything with >10M parameters.
    If a previous experiment failed with "MPS backend out of memory",
    propose a SMALLER architecture next time.
  - The final Kaggle submission notebook must run on CPU only within 90 minutes.
    Do not propose architectures whose inference cost cannot plausibly fit
    that CPU budget on 35k spectrograms.
  - Output is a vector of `num_classes` independent sigmoid probabilities
    (multi-label). Do NOT hardcode 234 — use `num_classes` from the data
    loader. Do not use softmax.
  - Audio is preprocessed into mel-spectrograms by a fixed pipeline. You work
    on the spectrograms directly; do not propose audio-waveform models.
  - No internet downloads, no custom CUDA kernels, no device-specific
    operators that don't work on CPU.
  - `hyperparams.epochs` MUST be exactly 1. We are in fast-iteration mode:
    every experiment trains for a single epoch so we can try more ideas
    per unit of wall-clock time. If your proposal has epochs > 1, the
    code generator will override it back to 1 and your measured score
    will reflect 1 epoch anyway — so just set it to 1 in the proposal.
  - `hyperparams.batch_size` is IGNORED by the code generator. The batch
    size is configured once for the whole project in `config/config.yaml`
    under `training:` and flows into every run automatically. You may
    still include a value in the proposal for documentation (e.g. 128),
    but the generated code will not use it.

## Architecture diversity — NO HERDING

Past runs have wasted 20+ experiments re-proposing tiny variations of
the same "3-conv CNN + SE attention + dropout" template. That is NOT
what you should do. Before every new proposal, look at the
`Previous Experiments` section in the user message and answer these
three questions explicitly to yourself:

  1. Which architecture FAMILIES have already been tried?
  2. Which families have NOT been tried yet?
  3. Pick a family that is either NEW or has only been tried once.

Your `architecture` string MUST start with the family tag in square
brackets so the memory can cluster them, e.g.

    "[efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, dropout 0.3"
    "[cnn_gru_hybrid] 3-conv CNN → GRU(128) → Linear"
    "[resnet18] ResNet-18 via TorchvisionAdapter, baseline head"
    "[custom_cnn] 4-conv from-scratch with BatchNorm + SE attention"

Family palette (you do NOT have to use all of them, but your first 10
proposals should come from AT LEAST 5 different families):

    [custom_cnn]         — small from-scratch Conv2d stack (CnnSmallV1 base)
    [efficientnet_b0]    — TorchvisionAdapter("efficientnet_b0", ...)
    [resnet18]           — TorchvisionAdapter("resnet18", ...)
    [mobilenet_v3_small] — TorchvisionAdapter("mobilenet_v3_small", ...)
    [cnn_gru_hybrid]     — Conv2d front-end → flatten → nn.GRU → Linear
    [cnn_attention]      — Conv2d front-end → self-attention → pool → Linear
    [deep_cnn]           — 5-6 Conv2d blocks with BatchNorm + residuals
    [specaugment_cnn]    — small CNN with aggressive SpecAugment + mixup

If the last 3 non-failed proposals all share the same family tag,
you MUST pick a different family this round. Repeating the same
family a fourth time in a row is forbidden.


[USER]
## Dataset Profile
- num_classes: 10
- num_samples: 442
- spectrogram_shape: (1, 128, 313)
- sample_rate: 32000 Hz
- imbalance_ratio: 158.00 (min=1, max=158)
- split: stratified_kfold (seed=42)
- train/val: 353/89
- most-populated classes:
    - 22956: 158
    - 22973: 72
    - 1161364: 63
- least-populated classes:
    - 116570: 1
    - 209233: 4
    - 1595929: 24

## Model Registry (verified building blocks — use as starting points OR inspiration)
### cnn_small_v1
- family: cnn | framework: torch | params: 0.04M | pretrained: none
- input_shape: 1x128x313 | output_dim: num_classes
- notes: Tiny from-scratch baseline (≈37k parameters). 3 Conv2D blocks + adaptive pool + linear head. Trains in under 2 minutes on CPU. Use as a sanity check before touching heavier architectures.

### efficientnet_b0
- family: cnn | framework: torch | params: 4.3M | pretrained: none
- input_shape: 1x128x313 | output_dim: num_classes
- notes: EfficientNet-B0 backbone wrapped in TorchvisionAdapter. The adapter takes the raw (B, 1, 128, T) spectrogram, repeats the channel to 3, bilinearly resizes to 224x224, runs the backbone, and uses a LazyLinear head to produce `num_classes` logits. Pretrained weights are NOT loaded (sandbox is offline) — the backbone trains from scratch. ~4.3M parameters, strong expressiveness but slower per batch than CnnSmallV1.

### mobilenet_v3_small
- family: cnn | framework: torch | params: 1.0M | pretrained: none
- input_shape: 1x128x313 | output_dim: num_classes
- notes: MobileNet-V3 Small via TorchvisionAdapter — the lightest architecture family we offer (≈1M params). Well-suited to the CPU-only Kaggle inference budget. Use when the bigger backbones hit runtime limits.

### resnet18
- family: cnn | framework: torch | params: 11.3M | pretrained: none
- input_shape: 1x128x313 | output_dim: num_classes
- notes: ResNet-18 via TorchvisionAdapter. Same channel-expand + 224x224 resize pipeline as efficientnet_b0. ~11.3M parameters, heavier than EfficientNet-B0 per forward pass but very stable to train from scratch. Good fallback when EfficientNet fails to converge.


## Previous Experiments (top-k by ROC-AUC, + recent failures)
### Top-K Successful Experiments
- **exp_001** [completed]
  - arch: [custom_cnn] cnn_small_v1 baseline
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.1699, f1_macro=0.0595, loss=1.2104, roc_auc_macro=0.5877
  - duration: 114.2s

- **exp_002** [completed]
  - arch: [cnn_attention] 3-conv CNN front-end followed by Self-Attention pooling layer
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.1
  - metrics: cmap_at_5=0.1432, f1_macro=0.1410, loss=1.1598, roc_auc_macro=0.4438
  - duration: 43.5s

- **exp_003** [completed]
  - arch: [cnn_gru_hybrid] 3-conv CNN front-end (32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2
  - metrics: loss=1.1361, roc_auc_macro=0.0000
  - duration: 0.0s

### Recent Failures
- **exp_005** [failed]
  - arch: [efficientnet_b0] EfficientNet-B0 backbone with Dropout regularization added to the final classification head | pretrained: efficientnet_b0
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.2

- **exp_004** [failed]
  - arch: [deep_cnn] 5-block residual CNN stack (32->64->128->128->256 channels) followed by GlobalAvgPool
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.0, dropout=0.25


## Last Result
- id: exp_005
- status: failed
- llm_model: gemma4:e4b
- architecture: [efficientnet_b0] EfficientNet-B0 backbone with Dropout regularization added to the final classification head
- pretrained: efficientnet_b0
- hyperparams: {"lr": 0.001, "batch_size": 128, "epochs": 1, "optimizer": "adam", "weight_decay": 0.0, "dropout": 0.2}

## Your Task
Propose the next experiment. You may either:
  (A) pick and configure a model from the registry above, OR
  (B) design a completely custom architecture from PyTorch primitives.

Whatever you choose, your `architecture` string MUST start with a
`[family_tag]` from the palette in the system prompt, followed by a
short design description. Example:
    "[cnn_gru_hybrid] 3-conv CNN front-end feeding a 1-layer GRU(128)"

Anti-herding check (do this before writing the JSON):
  1. Look at `Previous Experiments` above. Extract the `[family_tag]`
     from each proposal's architecture field.
  2. Count how many times each family has been proposed.
  3. If your intended family has been tried 3+ times already, PICK
     A DIFFERENT FAMILY from the palette in the system prompt.

If the previous experiments cluster around a plateau, try something different
in family or scale — don't just tweak hyperparameters. If previous experiments
failed with shape mismatches or OOM, fix the root cause in your next proposal.

Return ONLY valid JSON matching this schema:
{
  "type": "object",
  "required": [
    "architecture",
    "pretrained_model",
    "hyperparams",
    "augmentation"
  ],
  "properties": {
    "architecture": {
      "type": "string",
      "description": "Human-readable name/description of the architecture. If it matches a registry entry (e.g. \"cnn_small_v1\", \"efficientnet_b0\"), the generator will use the registry import_snippet. Otherwise, the generator will expect a full `nn.Module` class definition inside the produced code.\n"
    },
    "pretrained_model": {
      "type": [
        "string",
        "null"
      ],
      "description": "Name of a registry model to use as backbone, or null for from-scratch."
    },
    "hyperparams": {
      "type": "object",
      "required": [
        "lr",
        "batch_size",
        "epochs",
        "optimizer"
      ],
      "properties": {
        "lr": {
          "type": "number"
        },
        "batch_size": {
          "type": "integer"
        },
        "epochs": {
          "type": "integer"
        },
        "optimizer": {
          "type": "string"
        },
        "weight_decay": {
          "type": "number"
        },
        "dropout": {
          "type": "number"
        }
      }
    },
    "augmentation": {
      "type": "object",
      "properties": {
        "time_shift": {
          "type": "boolean"
        },
        "noise_injection": {
          "type": "boolean"
        },
        "mixup": {
          "type": "number"
        },
        "specaugment": {
          "type": "boolean"
        }
      }
    }
  }
}

CRITICAL: Return the actual VALUES, not a JSON Schema definition.
Do NOT include "type", "properties", "required", or "description" keys.
Your response must look like this:
  {"architecture": "[family_tag] description", "pretrained_model": null, "hyperparams": {...}, "augmentation": {...}}
NOT like this (WRONG — this is a schema, not values):
  {"type": "object", "properties": {"architecture": {"type": "string", ...}}}

```

## LLM Response
```
```json
{
  "architecture": "[mobilenet_v3_small] MobileNetV3-Small backbone with SpecAugment regularization",
  "pretrained_model": "mobilenet_v3_small",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.15
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": false,
    "mixup": 0.0,
    "specaugment": true
  }
}
```
```

## Output
- **architecture_proposal:** {'architecture': '[mobilenet_v3_small] MobileNetV3-Small backbone with SpecAugment regularization', 'pretrained_model': 'mobilenet_v3_small', 'hyperparams': {'lr': 0.001, 'batch_size': 128, 'epochs': 1, 'optimizer': 'adam', 'weight_decay': 0.0, 'dropout': 0.15}, 'augmentation': {'time_shift': True, 'noise_injection': False, 'mixup': 0.0, 'specaugment': True}}
