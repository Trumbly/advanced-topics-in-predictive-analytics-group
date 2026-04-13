# Experiment exp_007

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 03:43:48.086031+00:00
- **Started:** 2026-04-13 03:43:48.086031+00:00
- **Completed:** 2026-04-13 04:08:52.773028+00:00

## Model Config
- architecture: `[deep_cnn] Custom 5-block residual CNN stack (32->64->128->256 channels)`
- hyperparams:
  - lr: 0.0008
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.25
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_007_task_01_propose_architecture
- exp_007_task_02_generate_code
- exp_007_task_03_validate_code
- exp_007_task_04_execute_training
- exp_007_task_05_capture_metrics
- exp_007_recovery_01_error_recovery
- exp_007_task_06_validate_code
- exp_007_task_07_execute_training
- exp_007_task_08_capture_metrics
- exp_007_recovery_02_error_recovery
- exp_007_task_09_validate_code
- exp_007_task_10_execute_training
- exp_007_task_11_capture_metrics
- exp_007_recovery_03_error_recovery
- exp_007_task_12_validate_code
- exp_007_codegen_retry_01
- exp_007_task_13_validate_code
- exp_007_codegen_retry_02
- exp_007_task_14_validate_code
- exp_007_codegen_retry_03
- exp_007_task_15_validate_code
- exp_007_task_16_execute_training
- exp_007_task_17_capture_metrics
- exp_007_recovery_04_error_recovery
- exp_007_task_18_validate_code
- exp_007_codegen_retry_04
- exp_007_task_19_validate_code
- exp_007_task_20_execute_training
- exp_007_task_21_capture_metrics
- exp_007_recovery_05_error_recovery
- exp_007_task_22_validate_code
- exp_007_task_23_execute_training
- exp_007_task_24_capture_metrics
