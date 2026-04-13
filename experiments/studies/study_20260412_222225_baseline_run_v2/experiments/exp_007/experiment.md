# Experiment exp_007

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 23:20:52.073735+00:00
- **Started:** 2026-04-12 23:20:52.073735+00:00
- **Completed:** 2026-04-12 23:39:48.694687+00:00

## Model Config
- architecture: `[cnn_attention] 3-conv front-end (BatchNorm+ReLU) -> Self-Attention Block (Time-domain context) -> Adaptive Pooling -> Linear Head`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_007_task_01_propose_architecture
- exp_007_task_02_generate_code
