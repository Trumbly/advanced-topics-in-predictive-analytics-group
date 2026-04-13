# Experiment exp_008

- **Study:** study_20260412_142736_nemotron_3
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 13:56:17.116510+00:00
- **Started:** 2026-04-12 13:56:17.116510+00:00
- **Completed:** 2026-04-12 14:00:24.769726+00:00

## Model Config
- architecture: `[efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, baseline head`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_008_task_01_propose_architecture
- exp_008_task_02_generate_code
- exp_008_task_03_validate_code
- exp_008_task_04_execute_training
- exp_008_recovery_01_error_recovery
- exp_008_task_05_validate_code
- exp_008_task_06_execute_training
- exp_008_recovery_02_error_recovery
- exp_008_task_07_validate_code
- exp_008_task_08_execute_training
- exp_008_task_09_capture_metrics
- exp_008_recovery_03_error_recovery
- exp_008_task_10_validate_code
- exp_008_task_11_execute_training
- exp_008_task_12_capture_metrics
- exp_008_recovery_04_error_recovery
- exp_008_task_13_validate_code
- exp_008_task_14_execute_training
- exp_008_task_15_capture_metrics
- exp_008_recovery_05_error_recovery
- exp_008_task_16_validate_code
- exp_008_task_17_execute_training
- exp_008_task_18_capture_metrics
