# Experiment exp_015

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 05:49:48.326264+00:00
- **Started:** 2026-04-13 05:49:48.326264+00:00
- **Completed:** 2026-04-13 06:05:02.203199+00:00

## Model Config
- architecture: `[deep_cnn] 5-block deep residual-style CNN stack (32->64->128 channels) with BatchNorm and a final GlobalAveragePooling head`
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
- exp_015_task_01_propose_architecture
- exp_015_task_02_generate_code
- exp_015_task_03_validate_code
- exp_015_codegen_retry_01
- exp_015_task_04_validate_code
- exp_015_task_05_execute_training
- exp_015_recovery_01_error_recovery
- exp_015_task_06_validate_code
- exp_015_task_07_execute_training
- exp_015_task_08_capture_metrics
- exp_015_recovery_02_error_recovery
- exp_015_task_09_validate_code
- exp_015_task_10_execute_training
- exp_015_task_11_capture_metrics
- exp_015_recovery_03_error_recovery
- exp_015_task_12_validate_code
- exp_015_task_13_execute_training
- exp_015_task_14_capture_metrics
- exp_015_recovery_04_error_recovery
- exp_015_task_15_validate_code
- exp_015_task_16_execute_training
- exp_015_task_17_capture_metrics
- exp_015_recovery_05_error_recovery
- exp_015_task_18_validate_code
- exp_015_task_19_execute_training
- exp_015_task_20_capture_metrics
