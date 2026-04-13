# Experiment exp_013

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:21:46.176341+00:00
- **Started:** 2026-04-11 10:21:46.176341+00:00
- **Completed:** 2026-04-11 10:25:51.757961+00:00

## Model Config
- architecture: `EfficientNet-B0 truncated to first two blocks + linear head`
- pretrained: `efficientnet_b0`
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
- exp_013_task_01_propose_architecture
- exp_013_task_02_generate_code
- exp_013_task_03_validate_code
- exp_013_task_04_execute_training
- exp_013_task_05_capture_metrics
- exp_013_recovery_01_error_recovery
- exp_013_task_06_validate_code
- exp_013_task_07_execute_training
- exp_013_task_08_capture_metrics
- exp_013_recovery_02_error_recovery
- exp_013_task_09_validate_code
- exp_013_task_10_execute_training
- exp_013_task_11_capture_metrics
