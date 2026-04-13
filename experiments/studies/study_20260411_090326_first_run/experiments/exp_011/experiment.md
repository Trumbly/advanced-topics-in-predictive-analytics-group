# Experiment exp_011

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:17:44.647659+00:00
- **Started:** 2026-04-11 10:17:44.647659+00:00
- **Completed:** 2026-04-11 10:19:50.446950+00:00

## Model Config
- architecture: `EfficientNet-B0 backbone with linear output head`
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
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_011_task_01_propose_architecture
- exp_011_task_02_generate_code
- exp_011_task_03_validate_code
- exp_011_task_04_execute_training
- exp_011_recovery_01_error_recovery
- exp_011_task_05_validate_code
- exp_011_task_06_execute_training
- exp_011_recovery_02_error_recovery
