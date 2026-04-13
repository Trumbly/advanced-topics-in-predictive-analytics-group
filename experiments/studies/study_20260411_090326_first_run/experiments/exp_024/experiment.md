# Experiment exp_024

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 11:16:08.526865+00:00
- **Started:** 2026-04-11 11:16:08.526865+00:00
- **Completed:** 2026-04-11 11:18:18.810229+00:00

## Model Config
- architecture: `3-conv baseline with fused SE attention and dropout`
- pretrained: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_024_task_01_propose_architecture
- exp_024_task_02_generate_code
- exp_024_task_03_validate_code
- exp_024_task_04_execute_training
- exp_024_task_05_capture_metrics
- exp_024_recovery_01_error_recovery
- exp_024_task_06_validate_code
- exp_024_task_07_execute_training
- exp_024_task_08_capture_metrics
- exp_024_recovery_02_error_recovery
- exp_024_task_09_validate_code
- exp_024_task_10_execute_training
- exp_024_task_11_capture_metrics
