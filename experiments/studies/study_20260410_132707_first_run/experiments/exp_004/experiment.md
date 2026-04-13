# Experiment exp_004

- **Study:** study_20260410_132707_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:29:28.051685+00:00
- **Started:** 2026-04-10 13:29:28.051685+00:00
- **Completed:** 2026-04-10 13:30:08.778594+00:00

## Model Config
- architecture: `cnn_small_v1`
- pretrained: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 30
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.0
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
