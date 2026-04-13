# Experiment exp_011

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 04:54:12.419233+00:00
- **Started:** 2026-04-13 04:54:12.419233+00:00
- **Completed:** 2026-04-13 05:15:00.923957+00:00

## Model Config
- architecture: `[specaugment_cnn] Custom 4-block CNN front-end with residual connections, followed by Global Average Pooling head`
- hyperparams:
  - lr: 0.001
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
- exp_011_task_01_propose_architecture
- exp_011_task_02_generate_code
- exp_011_task_03_validate_code
- exp_011_task_04_execute_training
- exp_011_task_05_capture_metrics
- exp_011_recovery_01_error_recovery
- exp_011_task_06_validate_code
- exp_011_task_07_execute_training
- exp_011_task_08_capture_metrics
- exp_011_recovery_02_error_recovery
- exp_011_task_09_validate_code
- exp_011_codegen_retry_01
- exp_011_task_10_validate_code
- exp_011_task_11_execute_training
- exp_011_task_12_capture_metrics
- exp_011_recovery_03_error_recovery
- exp_011_task_13_validate_code
- exp_011_codegen_retry_02
- exp_011_task_14_validate_code
- exp_011_task_15_execute_training
- exp_011_task_16_capture_metrics
- exp_011_recovery_04_error_recovery
- exp_011_task_17_validate_code
- exp_011_task_18_execute_training
- exp_011_task_19_capture_metrics
- exp_011_recovery_05_error_recovery
- exp_011_task_20_validate_code
- exp_011_task_21_execute_training
- exp_011_task_22_capture_metrics
