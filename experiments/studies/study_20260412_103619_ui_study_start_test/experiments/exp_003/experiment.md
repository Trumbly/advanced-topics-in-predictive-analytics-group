# Experiment exp_003

- **Study:** study_20260412_103619_ui_study_start_test
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 08:41:46.208330+00:00
- **Started:** 2026-04-12 08:41:46.208330+00:00
- **Completed:** 2026-04-12 08:45:06.116248+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3 backbone with Global Average Pooling (GAP) before the final classification head`
- pretrained: `mobilenet_v3_small`
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
- exp_003_recovery_03_error_recovery
- exp_003_task_12_validate_code
- exp_003_task_13_execute_training
- exp_003_task_14_capture_metrics
