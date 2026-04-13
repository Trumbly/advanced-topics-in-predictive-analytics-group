# Task exp_020_task_01_propose_architecture

- **Experiment:** exp_020
- **Type:** llm
- **Name:** propose_architecture
- **Status:** completed
- **Started:** 2026-04-11 10:53:27.872435+00:00
- **Completed:** 2026-04-11 10:53:41.428857+00:00

## Prompt Used
```
[SYSTEM]
You are an ML research agent specializing in audio classification on mel-spectrograms.
You are working on BirdCLEF 2026 — a 234-class MULTI-LABEL bird species classification task.

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
  - The final Kaggle submission notebook must run on CPU only within 90 minutes.
    Do not propose architectures whose inference cost cannot plausibly fit
    that CPU budget on 35k spectrograms.
  - Output is a vector of 234 independent sigmoid probabilities (multi-label).
    Your final layer's output_dim MUST be 234. Do not use softmax.
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


[USER]
## Dataset Profile
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

## Model Registry (verified building blocks — use as starting points OR inspiration)
### cnn_small_v1
- family: cnn | framework: torch | params: 0.5M | pretrained: none
- input_shape: 1x128x256 | output_dim: 234
- notes: Fast from-scratch baseline. 3 Conv2D blocks + global pool + linear head. Trains in under 2 minutes on CPU. Use as a sanity check before touching heavier architectures.

### efficientnet_b0
- family: cnn | framework: torch | params: 5.3M | pretrained: imagenet
- input_shape: 3x224x224 | output_dim: 234
- notes: Strong transfer learning baseline. Requires the spectrogram to be resized to 224x224 and stacked to 3 channels. Freeze the backbone for the first few epochs, then fine-tune the last block.

### resnet18
- family: cnn | framework: torch | params: 11.7M | pretrained: imagenet
- input_shape: 3x224x224 | output_dim: 234
- notes: Reliable mid-sized CNN. Heavier than EfficientNet-B0 per forward pass but easier to train stably. Good fallback when EfficientNet overfits on the small labeled subset.


## Previous Experiments (top-k by ROC-AUC, + recent failures)
### Top-K Successful Experiments
- **exp_007** [completed]
  - arch: custom 3-conv CNN with fused residual blocks and SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.0382, roc_auc_macro=0.5613
  - duration: 755.0s

- **exp_008** [completed]
  - arch: cnn_small_v1 | pretrained: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.1028, roc_auc_macro=0.5447
  - duration: 100.4s

- **exp_004** [completed]
  - arch: 4-layer ResNet block with SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1
  - metrics: loss=0.0921, roc_auc_macro=0.5286
  - duration: 782.2s

- **exp_018** [completed]
  - arch: cnn_small_v1 baseline with dropout | pretrained: cnn_small_v1
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2
  - metrics: loss=0.1050, roc_auc_macro=0.5253
  - duration: 0.0s

- **exp_005** [completed]
  - arch: custom 3-conv CNN with dropout
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.2
  - metrics: loss=0.0417, roc_auc_macro=0.5144
  - duration: 776.4s

### Recent Failures
- **exp_019** [failed]
  - arch: 3-conv CNN with fused residual blocks and SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1

- **exp_017** [failed]
  - arch: 3-conv CNN with fused residual blocks and SE attention
  - hyperparams: lr=0.001, batch_size=128, epochs=1, optimizer=adam, weight_decay=0.001, dropout=0.1


## Last Result
- id: exp_019
- status: failed
- llm_model: nemotron-3-nano:4b
- architecture: 3-conv CNN with fused residual blocks and SE attention
- hyperparams: {"lr": 0.001, "batch_size": 128, "epochs": 1, "optimizer": "adam", "weight_decay": 0.001, "dropout": 0.1}

## Your Task
Propose the next experiment. You may either:
  (A) pick and configure a model from the registry above, OR
  (B) design a completely custom architecture from PyTorch primitives.

Whatever you choose, justify it in the `architecture` string: name the
design idea (e.g. "ResNet-style 4-block CNN with SE attention", "EfficientNet-B0
feature extractor + GRU head", "custom 3-conv baseline with dropout").

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

```

## LLM Response
```
{
  "architecture": "Enhanced 3-conv CNN with residual SE attention and dropout",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.001,
    "dropout": 0.2
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.0,
    "specaugment": false
  }
}
```

## Output
- **architecture_proposal:** {'architecture': 'Enhanced 3-conv CNN with residual SE attention and dropout', 'pretrained_model': None, 'hyperparams': {'lr': 0.001, 'batch_size': 128, 'epochs': 1, 'optimizer': 'adam', 'weight_decay': 0.001, 'dropout': 0.2}, 'augmentation': {'time_shift': True, 'noise_injection': True, 'mixup': 0.0, 'specaugment': False}}
