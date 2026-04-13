# Experiment exp_005

- **Study:** study_20260412_103619_ui_study_start_test
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 08:45:21.031569+00:00
- **Started:** 2026-04-12 08:45:21.031569+00:00
- **Completed:** 2026-04-12 08:48:52.059220+00:00

## Model Config
- architecture: `[efficientnet_b0] EfficientNet-B0 backbone followed by a multi-head self-attention pooling layer before the final classification head`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_recovery_01_error_recovery
- exp_005_task_05_validate_code
- exp_005_task_06_execute_training
- exp_005_task_07_capture_metrics
- exp_005_recovery_02_error_recovery
- exp_005_task_08_validate_code
- exp_005_task_09_execute_training
- exp_005_task_10_capture_metrics
- exp_005_recovery_03_error_recovery
- exp_005_task_11_validate_code
- exp_005_task_12_execute_training
- exp_005_task_13_capture_metrics
