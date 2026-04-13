# Task exp_001_task_01_propose_architecture

- **Experiment:** exp_001
- **Type:** llm
- **Name:** propose_architecture
- **Status:** completed
- **Started:** 2026-04-13 22:57:01.776160+00:00
- **Completed:** 2026-04-13 22:57:16.565996+00:00

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
  - **ALWAYS use pretrained backbones with transfer learning** unless
    every pretrained option has already been tried in past experiments.
    Pretrained ImageNet weights give a massive head-start on spectrogram
    features. EfficientNet-B0 and MobileNetV3-Small via TorchvisionAdapter
    consistently reach 0.90-0.96 ROC-AUC. Custom CNNs from scratch
    rarely exceed 0.75. Your FIRST experiment should ALWAYS be a
    pretrained model (e.g., EfficientNet-B0). Only propose custom CNNs
    after pretrained options have been explored.
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
  - `hyperparams.epochs` should be 7. The agent trains each experiment
    for up to 7 epochs with early stopping (patience=3). Set epochs=7
    in the proposal and let early stopping handle convergence.
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
No previous experiments yet. Your first job is to establish a strong
baseline with a pretrained EfficientNet-B0 backbone via TorchvisionAdapter,
7 epochs, Adam lr=1e-3, light augmentation. Pretrained backbones
consistently reach 0.90+ ROC-AUC — always start with one.

You MUST return ONLY valid JSON, nothing else. No prose, no markdown
fences, no explanation. Your entire response must be parseable as a
JSON object with exactly this shape:

{
  "architecture": "[efficientnet_b0] pretrained EfficientNet-B0 baseline",
  "pretrained_model": "efficientnet_b0",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 7,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.0,
    "specaugment": false
  }
}

You may copy that exact JSON verbatim — it is a valid first experiment.
Later proposals MUST start the architecture string with a different
`[family_tag]` to force diverse exploration.

```

## LLM Response
```
```json
{
  "architecture": "[efficientnet_b0] pretrained EfficientNet-B0 baseline",
  "pretrained_model": "efficientnet_b0",
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 7,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "dropout": 0.1
  },
  "augmentation": {
    "time_shift": true,
    "noise_injection": true,
    "mixup": 0.0,
    "specaugment": false
  }
}
```
```

## Output
- **architecture_proposal:** {'architecture': '[efficientnet_b0] pretrained EfficientNet-B0 baseline', 'pretrained_model': 'efficientnet_b0', 'hyperparams': {'lr': 0.001, 'batch_size': 128, 'epochs': 7, 'optimizer': 'adam', 'weight_decay': 0.0, 'dropout': 0.1}, 'augmentation': {'time_shift': True, 'noise_injection': True, 'mixup': 0.0, 'specaugment': False}}
