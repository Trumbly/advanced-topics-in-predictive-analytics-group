# Task exp_001_task_01_propose_architecture

- **Experiment:** exp_001
- **Type:** llm
- **Name:** propose_architecture
- **Status:** completed
- **Started:** 2026-04-10 14:10:41.468727+00:00
- **Completed:** 2026-04-10 14:10:47.556669+00:00

## Prompt Used
```
[SYSTEM]
You are an ML research agent specializing in audio classification on mel-spectrograms.
You are working on BirdCLEF 2026 — a 234-class MULTI-LABEL bird species classification task.

Your job is to DESIGN architectures, not pick from a menu. Be creative.
Use what you know about audio classification, CNNs, attention, transfer
learning, and BirdCLEF's quirks (noisy field recordings, class imbalance,
variable-length audio, CPU-only inference constraint) to propose something
that has a real chance of improving the current best score.

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
    Do not propose architectures that cannot plausibly fit that budget for
    both training and inference on 35k spectrograms.
  - Output is a vector of 234 independent sigmoid probabilities (multi-label).
    Your final layer's output_dim MUST be 234. Do not use softmax.
  - Audio is preprocessed into mel-spectrograms by a fixed pipeline. You work
    on the spectrograms directly; do not propose audio-waveform models.
  - No GPU-only ops, no internet downloads, no custom CUDA kernels.


[USER]
No previous experiments yet. Your first job is to verify the pipeline
runs end-to-end with the fastest possible baseline: CnnSmallV1 from the
registry, 2 epochs, batch_size 32, Adam lr=1e-3, light augmentation.
Ambitious architectures come later — this run is a smoke test.

You MUST return ONLY valid JSON, nothing else. No prose, no markdown
fences, no explanation. Your entire response must be parseable as a
JSON object with exactly this shape:

{
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 2,
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

```

## LLM Response
```
{
  "architecture": "cnn_small_v1",
  "pretrained_model": null,
  "hyperparams": {
    "lr": 0.001,
    "batch_size": 32,
    "epochs": 2,
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
- **architecture_proposal:** {'architecture': 'cnn_small_v1', 'pretrained_model': None, 'hyperparams': {'lr': 0.001, 'batch_size': 32, 'epochs': 2, 'optimizer': 'adam', 'weight_decay': 0.0, 'dropout': 0.1}, 'augmentation': {'time_shift': True, 'noise_injection': True, 'mixup': 0.0, 'specaugment': False}}
