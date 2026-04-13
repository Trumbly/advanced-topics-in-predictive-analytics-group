# Experiment exp_017

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:44:24.748444+00:00
- **Started:** 2026-04-11 10:44:24.748444+00:00
- **Completed:** 2026-04-11 10:47:18.769521+00:00

## Model Config
- architecture: `3-conv CNN with fused residual blocks and SE attention`
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
  - mixup: 0
  - specaugment: False

## Tasks
- exp_017_task_01_propose_architecture
- exp_017_task_02_generate_code
- exp_017_task_03_validate_code
- exp_017_recovery_01_error_recovery
- exp_017_task_04_validate_code
- exp_017_task_05_execute_training
- exp_017_recovery_02_error_recovery
