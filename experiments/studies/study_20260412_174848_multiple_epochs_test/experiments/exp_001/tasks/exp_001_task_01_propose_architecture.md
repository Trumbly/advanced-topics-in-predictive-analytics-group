# Task exp_001_task_01_propose_architecture

- **Experiment:** exp_001
- **Type:** llm
- **Name:** propose_architecture
- **Status:** completed
- **Started:** 2026-04-12 15:48:48.834278+00:00
- **Completed:** 2026-04-12 15:48:56.330079+00:00

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
No previous experiments yet. Your first job is to verify the pipeline
runs end-to-end with the fastest possible baseline: CnnSmallV1 from the
registry, 2 epochs, batch_size 32, Adam lr=1e-3, light augmentation.
Ambitious architectures come later — this run is a smoke test.

You MUST return ONLY valid JSON, nothing else. No prose, no markdown
fences, no explanation. Your entire response must be parseable as a
JSON object with exactly this shape:

{
  "architecture": "[custom_cnn] cnn_small_v1 baseline",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
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
{
  "architecture": "[custom_cnn] cnn_small_v1 baseline",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 1,
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

## Output
- **architecture_proposal:** {'architecture': '[custom_cnn] cnn_small_v1 baseline', 'pretrained_model': None, 'hyperparams': {'lr': 0.001, 'batch_size': 128, 'epochs': 1, 'optimizer': 'adam', 'weight_decay': 0.0, 'dropout': 0.1}, 'augmentation': {'time_shift': True, 'noise_injection': True, 'mixup': 0.0, 'specaugment': False}}
