# Experiment exp_003

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 02:40:01.600663+00:00
- **Started:** 2026-04-13 02:40:01.600663+00:00
- **Completed:** 2026-04-13 02:56:38.730011+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end (32->64 channels) -> Global AvgPool over feature channels -> 2-layer GRU(128) head`
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

## Results
- loss: 1.1361
- roc_auc_macro: 0.0000
- duration: 0.0s

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_codegen_retry_01
- exp_003_task_04_validate_code
- exp_003_codegen_retry_02
- exp_003_task_05_validate_code
- exp_003_codegen_retry_03
- exp_003_task_06_validate_code
- exp_003_task_07_execute_training
- exp_003_recovery_01_error_recovery
- exp_003_task_08_validate_code
- exp_003_task_09_execute_training
- exp_003_recovery_02_error_recovery
- exp_003_task_10_validate_code
- exp_003_task_11_execute_training
- exp_003_task_12_capture_metrics
- exp_003_recovery_03_error_recovery
- exp_003_task_13_validate_code
- exp_003_task_14_execute_training
- exp_003_task_15_capture_metrics
- exp_003_task_16_analyze_results
