# Experiment exp_003

- **Study:** study_20260411_084202_qwen_retry
- **Status:** failed
- **LLM:** qwen3-coder
- **Created:** 2026-04-11 08:44:15.411564+00:00
- **Started:** 2026-04-11 08:44:15.411564+00:00
- **Completed:** 2026-04-11 08:45:46.896152+00:00

## Model Config
- architecture: `Lightweight 3-Block CNN with Self-Attention and Dynamic Filtering`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

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
