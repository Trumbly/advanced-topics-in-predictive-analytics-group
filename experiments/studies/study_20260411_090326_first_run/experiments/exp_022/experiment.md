# Experiment exp_022

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 11:12:06.062709+00:00
- **Started:** 2026-04-11 11:12:06.062709+00:00
- **Completed:** 2026-04-11 11:14:07.315350+00:00

## Model Config
- architecture: `3‑conv concatenated CNN with residual SE attention and dropout`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_022_task_01_propose_architecture
- exp_022_task_02_generate_code
- exp_022_task_03_validate_code
- exp_022_task_04_execute_training
- exp_022_task_05_capture_metrics
- exp_022_recovery_01_error_recovery
