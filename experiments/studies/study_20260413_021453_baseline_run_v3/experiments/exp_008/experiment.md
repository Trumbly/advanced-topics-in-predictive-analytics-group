# Experiment exp_008

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 04:08:52.787167+00:00
- **Started:** 2026-04-13 04:08:52.787167+00:00
- **Completed:** 2026-04-13 04:21:25.890188+00:00

## Model Config
- architecture: `[efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, standard feature extraction`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.15
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_008_task_01_propose_architecture
- exp_008_task_02_generate_code
- exp_008_task_03_validate_code
- exp_008_task_04_execute_training
- exp_008_recovery_01_error_recovery
- exp_008_task_05_validate_code
- exp_008_task_06_execute_training
- exp_008_task_07_capture_metrics
- exp_008_recovery_02_error_recovery
- exp_008_task_08_validate_code
- exp_008_task_09_execute_training
- exp_008_task_10_capture_metrics
- exp_008_recovery_03_error_recovery
- exp_008_task_11_validate_code
- exp_008_task_12_execute_training
- exp_008_task_13_capture_metrics
- exp_008_recovery_04_error_recovery
- exp_008_task_14_validate_code
- exp_008_task_15_execute_training
- exp_008_task_16_capture_metrics
- exp_008_recovery_05_error_recovery
- exp_008_task_17_validate_code
- exp_008_task_18_execute_training
- exp_008_task_19_capture_metrics
