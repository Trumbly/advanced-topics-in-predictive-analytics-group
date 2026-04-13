# Experiment exp_003

- **Study:** study_20260412_093854_ui_study_start_test
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 07:43:26.339023+00:00
- **Started:** 2026-04-12 07:43:26.339023+00:00
- **Completed:** 2026-04-12 07:46:04.098591+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3Small via TorchvisionAdapter, standard classification head`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0005
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.3
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
