# Experiment exp_001

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 22:22:25.629297+00:00
- **Started:** 2026-04-12 22:22:25.629297+00:00
- **Completed:** 2026-04-12 22:41:29.872786+00:00

## Model Config
- architecture: `[custom_cnn] cnn_small_v1 baseline`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Results
- cmap_at_5: 0.2184
- f1_macro: 0.1037
- loss: 1.1283
- roc_auc_macro: 0.6676
- duration: 9.0s
- training curves:
  - loss (len=1): first=1.1283, last=1.1283, min=1.1283, max=1.1283
  - roc_auc_macro (len=1): first=0.6676, last=0.6676, min=0.6676, max=0.6676
  - cmap_at_5 (len=1): first=0.2184, last=0.2184, min=0.2184, max=0.2184
  - f1_macro (len=1): first=0.1037, last=0.1037, min=0.1037, max=0.1037

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_codegen_retry_01
- exp_001_task_04_validate_code
- exp_001_task_05_execute_training
- exp_001_task_06_capture_metrics
- exp_001_recovery_01_error_recovery
- exp_001_task_07_validate_code
- exp_001_task_08_execute_training
- exp_001_task_09_capture_metrics
- exp_001_recovery_02_error_recovery
- exp_001_task_10_validate_code
- exp_001_task_11_execute_training
- exp_001_task_12_capture_metrics
- exp_001_recovery_03_error_recovery
- exp_001_task_13_validate_code
- exp_001_codegen_retry_02
- exp_001_task_14_validate_code
- exp_001_task_15_execute_training
- exp_001_task_16_capture_metrics
- exp_001_recovery_04_error_recovery
- exp_001_task_17_validate_code
- exp_001_task_18_execute_training
- exp_001_task_19_capture_metrics
- exp_001_task_20_analyze_results
