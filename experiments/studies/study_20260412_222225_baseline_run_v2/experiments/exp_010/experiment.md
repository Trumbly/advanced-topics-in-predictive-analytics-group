# Experiment exp_010

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 23:57:01.081370+00:00
- **Started:** 2026-04-12 23:57:01.081370+00:00
- **Completed:** 2026-04-13 00:29:29.711347+00:00

## Model Config
- architecture: `[cnn_attention] 3-block Conv stack (64 channels) -> Self-Attention over time dimension -> Global Average Pooling -> Linear Head`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Results
- ap_score_macro: 0.1749
- loss: 0.9013
- roc_auc_macro: 0.5699
- duration: 0.0s

## Tasks
- exp_010_task_01_propose_architecture
- exp_010_task_02_generate_code
- exp_010_task_03_validate_code
- exp_010_task_04_execute_training
- exp_010_recovery_01_error_recovery
- exp_010_task_05_validate_code
- exp_010_codegen_retry_01
- exp_010_recovery_02_error_recovery
- exp_010_task_06_validate_code
- exp_010_task_07_execute_training
- exp_010_recovery_03_error_recovery
- exp_010_task_08_validate_code
- exp_010_task_09_execute_training
- exp_010_recovery_04_error_recovery
- exp_010_task_10_validate_code
- exp_010_task_11_execute_training
- exp_010_task_12_capture_metrics
- exp_010_recovery_05_error_recovery
- exp_010_task_13_validate_code
- exp_010_task_14_execute_training
- exp_010_task_15_capture_metrics
- exp_010_task_16_analyze_results
