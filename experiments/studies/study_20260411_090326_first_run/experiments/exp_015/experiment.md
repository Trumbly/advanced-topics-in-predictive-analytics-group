# Experiment exp_015

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:39:32.710374+00:00
- **Started:** 2026-04-11 10:39:32.710374+00:00
- **Completed:** 2026-04-11 10:42:23.709744+00:00

## Model Config
- architecture: `custom 3-conv CNN with fused residual blocks, SE attention, dropout, and mixup augmentation`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_015_task_01_propose_architecture
- exp_015_task_02_generate_code
- exp_015_task_03_validate_code
- exp_015_task_04_execute_training
- exp_015_task_05_capture_metrics
- exp_015_recovery_01_error_recovery
- exp_015_task_06_validate_code
- exp_015_task_07_execute_training
- exp_015_task_08_capture_metrics
- exp_015_recovery_02_error_recovery
