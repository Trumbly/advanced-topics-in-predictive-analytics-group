# Experiment exp_016

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:42:23.712533+00:00
- **Started:** 2026-04-11 10:42:23.712533+00:00
- **Completed:** 2026-04-11 10:44:24.745843+00:00

## Model Config
- architecture: `EfficientNet-B0 full model with linear head`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_016_task_01_propose_architecture
- exp_016_task_02_generate_code
- exp_016_task_03_validate_code
- exp_016_task_04_execute_training
- exp_016_recovery_01_error_recovery
- exp_016_task_05_validate_code
- exp_016_task_06_execute_training
- exp_016_recovery_02_error_recovery
- exp_016_task_07_validate_code
