# Experiment exp_001

- **Study:** study_20260411_084632_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 08:46:32.776287+00:00
- **Started:** 2026-04-11 08:46:32.776287+00:00
- **Completed:** 2026-04-11 08:48:07.472179+00:00

## Model Config
- architecture: `cnn_small_v1`
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
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_recovery_01_error_recovery
- exp_001_task_05_validate_code
- exp_001_task_06_execute_training
- exp_001_recovery_02_error_recovery
- exp_001_task_07_validate_code
- exp_001_task_08_execute_training
