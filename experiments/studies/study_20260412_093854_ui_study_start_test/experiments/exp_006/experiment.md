# Experiment exp_006

- **Study:** study_20260412_093854_ui_study_start_test
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 07:51:28.418918+00:00
- **Started:** 2026-04-12 07:51:28.418918+00:00
- **Completed:** 2026-04-12 07:54:20.407850+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3 backbone followed by a custom Squeeze-and-Excitation (SE) feature refinement block and Global Average Pooling`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: True
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
