# Experiment exp_003

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:08:43.697909+00:00
- **Started:** 2026-04-11 09:08:43.697909+00:00
- **Completed:** 2026-04-11 09:11:18.068447+00:00

## Model Config
- architecture: `3-conv SE attention network`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
- exp_003_recovery_01_error_recovery
- exp_003_task_06_validate_code
- exp_003_task_07_execute_training
- exp_003_task_08_capture_metrics
- exp_003_recovery_02_error_recovery
- exp_003_task_09_validate_code
- exp_003_task_10_execute_training
- exp_003_task_11_capture_metrics
