# Experiment exp_012

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:19:50.449636+00:00
- **Started:** 2026-04-11 10:19:50.449636+00:00
- **Completed:** 2026-04-11 10:21:46.173704+00:00

## Model Config
- architecture: `custom 3-conv CNN with fused residual blocks, SE attention, and dropout`
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
- exp_012_task_01_propose_architecture
- exp_012_task_02_generate_code
- exp_012_task_03_validate_code
- exp_012_task_04_execute_training
- exp_012_task_05_capture_metrics
- exp_012_recovery_01_error_recovery
