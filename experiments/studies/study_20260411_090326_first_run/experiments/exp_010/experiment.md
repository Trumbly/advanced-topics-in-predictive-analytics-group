# Experiment exp_010

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:15:01.747127+00:00
- **Started:** 2026-04-11 10:15:01.747127+00:00
- **Completed:** 2026-04-11 10:17:44.645034+00:00

## Model Config
- architecture: `custom 3-conv CNN with self-attention and dropout`
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
- exp_010_task_01_propose_architecture
- exp_010_task_02_generate_code
- exp_010_task_03_validate_code
- exp_010_task_04_execute_training
- exp_010_task_05_capture_metrics
- exp_010_recovery_01_error_recovery
- exp_010_task_06_validate_code
- exp_010_task_07_execute_training
- exp_010_recovery_02_error_recovery
- exp_010_task_08_validate_code
