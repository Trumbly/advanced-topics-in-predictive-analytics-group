# Experiment exp_004

- **Study:** study_20260412_105608_ui_study_start_test_with_dev_model_for_better_code
- **Status:** failed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 09:45:20.131944+00:00
- **Started:** 2026-04-12 09:45:20.131944+00:00
- **Completed:** 2026-04-12 09:48:27.855000+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end feeding a 2-layer GRU(128) with dropout, followed by a linear head with sigmoid output`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.3
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
- exp_004_task_05_capture_metrics
- exp_004_recovery_01_error_recovery
- exp_004_task_06_validate_code
- exp_004_task_07_execute_training
- exp_004_task_08_capture_metrics
- exp_004_recovery_02_error_recovery
- exp_004_task_09_validate_code
- exp_004_task_10_execute_training
- exp_004_task_11_capture_metrics
- exp_004_recovery_03_error_recovery
- exp_004_task_12_validate_code
- exp_004_task_13_execute_training
- exp_004_task_14_capture_metrics
- exp_004_recovery_04_error_recovery
- exp_004_task_15_validate_code
- exp_004_task_16_execute_training
- exp_004_task_17_capture_metrics
- exp_004_recovery_05_error_recovery
- exp_004_task_18_validate_code
- exp_004_task_19_execute_training
- exp_004_task_20_capture_metrics
