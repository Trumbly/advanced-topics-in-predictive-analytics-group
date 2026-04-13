# Experiment exp_009

- **Study:** study_20260412_142736_nemotron_3
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 14:00:24.770574+00:00
- **Started:** 2026-04-12 14:00:24.770574+00:00
- **Completed:** 2026-04-12 14:21:17.047327+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end feeding a 1-layer GRU(128) → Linear`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.25
  - specaugment: True

## Results
- cmap_at_5: 0.0125
- f1_macro: 0.0083
- loss: 0.5169
- roc_auc_macro: 0.6677
- duration: 973.9s
- training curves:
  - loss (len=1): first=0.5169, last=0.5169, min=0.5169, max=0.5169
  - roc_auc_macro (len=1): first=0.6677, last=0.6677, min=0.6677, max=0.6677
  - cmap_at_5 (len=1): first=0.0125, last=0.0125, min=0.0125, max=0.0125
  - f1_macro (len=1): first=0.0083, last=0.0083, min=0.0083, max=0.0083

## Tasks
- exp_009_task_01_propose_architecture
- exp_009_task_02_generate_code
- exp_009_task_03_validate_code
- exp_009_task_04_execute_training
- exp_009_task_05_capture_metrics
- exp_009_recovery_01_error_recovery
- exp_009_task_06_validate_code
- exp_009_task_07_execute_training
- exp_009_task_08_capture_metrics
- exp_009_recovery_02_error_recovery
- exp_009_task_09_validate_code
- exp_009_task_10_execute_training
- exp_009_task_11_capture_metrics
- exp_009_recovery_03_error_recovery
- exp_009_task_12_validate_code
- exp_009_task_13_execute_training
- exp_009_task_14_capture_metrics
- exp_009_recovery_04_error_recovery
- exp_009_task_15_validate_code
- exp_009_task_16_execute_training
- exp_009_task_17_capture_metrics
- exp_009_recovery_05_error_recovery
- exp_009_task_18_validate_code
- exp_009_task_19_execute_training
- exp_009_task_20_capture_metrics
- exp_009_task_21_analyze_results
