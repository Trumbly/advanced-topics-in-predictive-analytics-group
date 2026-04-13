# Experiment exp_009

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:13:04.005955+00:00
- **Started:** 2026-04-11 10:13:04.005955+00:00
- **Completed:** 2026-04-11 10:15:01.744582+00:00

## Model Config
- architecture: `custom 3-conv CNN with fused residual blocks and SE attention (improved)`
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
- exp_009_task_01_propose_architecture
- exp_009_task_02_generate_code
- exp_009_task_03_validate_code
- exp_009_task_04_execute_training
- exp_009_task_05_capture_metrics
- exp_009_recovery_01_error_recovery
