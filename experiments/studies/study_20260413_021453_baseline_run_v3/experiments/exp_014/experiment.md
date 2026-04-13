# Experiment exp_014

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 05:35:08.630419+00:00
- **Started:** 2026-04-13 05:35:08.630419+00:00
- **Completed:** 2026-04-13 05:49:48.325001+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] cnn_small_v1 backbone output features sequence fed into 2-layer GRU(128) head`
- pretrained: `cnn_small_v1`
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
  - specaugment: True

## Tasks
- exp_014_task_01_propose_architecture
- exp_014_task_02_generate_code
- exp_014_task_03_validate_code
- exp_014_task_04_execute_training
- exp_014_recovery_01_error_recovery
- exp_014_task_05_validate_code
- exp_014_codegen_retry_01
- exp_014_task_06_validate_code
- exp_014_task_07_execute_training
- exp_014_task_08_capture_metrics
- exp_014_recovery_02_error_recovery
- exp_014_task_09_validate_code
- exp_014_task_10_execute_training
- exp_014_recovery_03_error_recovery
- exp_014_task_11_validate_code
- exp_014_task_12_execute_training
- exp_014_task_13_capture_metrics
- exp_014_recovery_04_error_recovery
- exp_014_task_14_validate_code
- exp_014_task_15_execute_training
- exp_014_recovery_05_error_recovery
- exp_014_task_16_validate_code
- exp_014_task_17_execute_training
