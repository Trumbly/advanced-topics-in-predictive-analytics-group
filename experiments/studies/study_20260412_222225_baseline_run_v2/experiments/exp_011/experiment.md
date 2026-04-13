# Experiment exp_011

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 00:29:29.715934+00:00
- **Started:** 2026-04-13 00:29:29.715934+00:00
- **Completed:** 2026-04-13 00:51:26.784952+00:00

## Model Config
- architecture: `[deep_cnn] 5-layer residual CNN stack using BatchNorm and residual connections, followed by Global Average Pooling and the final linear head.`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_011_task_01_propose_architecture
- exp_011_task_02_generate_code
- exp_011_task_03_validate_code
- exp_011_task_04_execute_training
- exp_011_task_05_capture_metrics
- exp_011_recovery_01_error_recovery
- exp_011_task_06_validate_code
- exp_011_task_07_execute_training
- exp_011_task_08_capture_metrics
- exp_011_recovery_02_error_recovery
- exp_011_task_09_validate_code
- exp_011_task_10_execute_training
- exp_011_task_11_capture_metrics
- exp_011_recovery_03_error_recovery
- exp_011_task_12_validate_code
- exp_011_task_13_execute_training
- exp_011_task_14_capture_metrics
- exp_011_recovery_04_error_recovery
- exp_011_task_15_validate_code
- exp_011_task_16_execute_training
- exp_011_task_17_capture_metrics
- exp_011_recovery_05_error_recovery
- exp_011_task_18_validate_code
- exp_011_task_19_execute_training
- exp_011_task_20_capture_metrics
