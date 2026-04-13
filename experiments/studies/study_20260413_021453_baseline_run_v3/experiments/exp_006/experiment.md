# Experiment exp_006

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 03:32:47.743108+00:00
- **Started:** 2026-04-13 03:32:47.743108+00:00
- **Completed:** 2026-04-13 03:43:48.084924+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3-Small backbone with SpecAugment regularization`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.15
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_006_task_01_propose_architecture
- exp_006_task_02_generate_code
- exp_006_task_03_validate_code
- exp_006_task_04_execute_training
- exp_006_task_05_capture_metrics
- exp_006_recovery_01_error_recovery
- exp_006_task_06_validate_code
- exp_006_task_07_execute_training
- exp_006_task_08_capture_metrics
- exp_006_recovery_02_error_recovery
- exp_006_task_09_validate_code
- exp_006_task_10_execute_training
- exp_006_task_11_capture_metrics
- exp_006_recovery_03_error_recovery
- exp_006_task_12_validate_code
- exp_006_task_13_execute_training
- exp_006_task_14_capture_metrics
- exp_006_recovery_04_error_recovery
- exp_006_task_15_validate_code
- exp_006_task_16_execute_training
- exp_006_task_17_capture_metrics
- exp_006_recovery_05_error_recovery
- exp_006_task_18_validate_code
- exp_006_task_19_execute_training
