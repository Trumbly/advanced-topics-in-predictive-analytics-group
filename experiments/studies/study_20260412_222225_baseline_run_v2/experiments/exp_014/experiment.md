# Experiment exp_014

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 01:18:05.673227+00:00
- **Started:** 2026-04-13 01:18:05.673227+00:00
- **Completed:** 2026-04-13 01:30:21.686306+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3 Small backbone via TorchvisionAdapter, followed by a Global Average Pooling layer and a linear head for multi-label classification.`
- pretrained: `mobilenet_v3_small`
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
- exp_014_task_01_propose_architecture
- exp_014_task_02_generate_code
- exp_014_task_03_validate_code
- exp_014_task_04_execute_training
- exp_014_task_05_capture_metrics
- exp_014_recovery_01_error_recovery
- exp_014_task_06_validate_code
- exp_014_task_07_execute_training
- exp_014_task_08_capture_metrics
- exp_014_recovery_02_error_recovery
- exp_014_task_09_validate_code
- exp_014_task_10_execute_training
- exp_014_task_11_capture_metrics
- exp_014_recovery_03_error_recovery
- exp_014_task_12_validate_code
- exp_014_task_13_execute_training
- exp_014_task_14_capture_metrics
- exp_014_recovery_04_error_recovery
- exp_014_task_15_validate_code
- exp_014_task_16_execute_training
- exp_014_task_17_capture_metrics
- exp_014_recovery_05_error_recovery
- exp_014_task_18_validate_code
- exp_014_task_19_execute_training
- exp_014_task_20_capture_metrics
