# Experiment exp_004

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 02:56:38.731474+00:00
- **Started:** 2026-04-13 02:56:38.731474+00:00
- **Completed:** 2026-04-13 03:21:31.546209+00:00

## Model Config
- architecture: `[deep_cnn] 5-block residual CNN stack (32->64->128->128->256 channels) followed by GlobalAvgPool`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.25
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_codegen_retry_01
- exp_004_task_04_validate_code
- exp_004_task_05_execute_training
- exp_004_task_06_capture_metrics
- exp_004_recovery_01_error_recovery
- exp_004_task_07_validate_code
- exp_004_codegen_retry_02
- exp_004_task_08_validate_code
- exp_004_codegen_retry_03
- exp_004_task_09_validate_code
- exp_004_task_10_execute_training
- exp_004_task_11_capture_metrics
- exp_004_recovery_02_error_recovery
- exp_004_task_12_validate_code
- exp_004_codegen_retry_04
- exp_004_task_13_validate_code
- exp_004_codegen_retry_05
- exp_004_task_14_validate_code
- exp_004_task_15_execute_training
- exp_004_task_16_capture_metrics
- exp_004_recovery_03_error_recovery
- exp_004_task_17_validate_code
- exp_004_task_18_execute_training
- exp_004_task_19_capture_metrics
- exp_004_recovery_04_error_recovery
- exp_004_task_20_validate_code
- exp_004_task_21_execute_training
- exp_004_task_22_capture_metrics
- exp_004_recovery_05_error_recovery
- exp_004_task_23_validate_code
- exp_004_task_24_execute_training
- exp_004_task_25_capture_metrics
