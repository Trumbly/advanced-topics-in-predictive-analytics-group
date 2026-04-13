# Experiment exp_001

- **Study:** study_20260411_084103_qwen_retry
- **Status:** failed
- **LLM:** qwen3-coder
- **Created:** 2026-04-11 08:41:03.172104+00:00
- **Started:** 2026-04-11 08:41:03.172104+00:00
- **Completed:** 2026-04-11 08:41:56.466554+00:00

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
- exp_001_task_05_capture_metrics
- exp_001_recovery_01_error_recovery
- exp_001_task_06_validate_code
- exp_001_task_07_execute_training
- exp_001_task_08_capture_metrics
- exp_001_recovery_02_error_recovery
- exp_001_task_09_validate_code
- exp_001_task_10_execute_training
- exp_001_task_11_capture_metrics
