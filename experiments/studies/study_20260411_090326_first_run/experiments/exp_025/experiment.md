# Experiment exp_025

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 11:18:18.811810+00:00
- **Started:** 2026-04-11 11:18:18.811810+00:00
- **Completed:** 2026-04-11 11:20:55.275318+00:00

## Model Config
- architecture: `3-conv CNN with fused SE attention and dropout`
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
- exp_025_task_01_propose_architecture
- exp_025_task_02_generate_code
- exp_025_task_03_validate_code
- exp_025_task_04_execute_training
- exp_025_task_05_capture_metrics
- exp_025_recovery_01_error_recovery
- exp_025_task_06_validate_code
- exp_025_task_07_execute_training
- exp_025_task_08_capture_metrics
- exp_025_recovery_02_error_recovery
- exp_025_task_09_validate_code
