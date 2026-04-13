# Experiment exp_015

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** running
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 01:30:21.687310+00:00
- **Started:** 2026-04-13 01:30:21.687310+00:00

## Model Config
- architecture: `[deep_cnn] 4-block residual CNN backbone with Squeeze-and-Excitation (SE) attention channel weighting before Global Average Pooling`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_015_task_01_propose_architecture
