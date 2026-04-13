# Experiment exp_004

- **Study:** study_20260410_133357_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:35:49.284281+00:00
- **Started:** 2026-04-10 13:35:49.284281+00:00
- **Completed:** 2026-04-10 13:36:37.226953+00:00

## Model Config
- architecture: `3-convolutional baseline with global average pooling and dropout`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 30
  - optimizer: adamw
  - weight_decay: 0.0005
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
- exp_004_task_05_capture_metrics
