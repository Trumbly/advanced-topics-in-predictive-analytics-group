# Experiment exp_002

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 02:19:15.031338+00:00
- **Started:** 2026-04-13 02:19:15.031338+00:00
- **Completed:** 2026-04-13 02:40:01.597113+00:00

## Model Config
- architecture: `[cnn_attention] 3-conv CNN front-end followed by Self-Attention pooling layer`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Results
- cmap_at_5: 0.1432
- f1_macro: 0.1410
- loss: 1.1598
- roc_auc_macro: 0.4438
- duration: 43.5s
- training curves:
  - loss (len=1): first=1.1598, last=1.1598, min=1.1598, max=1.1598
  - roc_auc_macro (len=1): first=0.4438, last=0.4438, min=0.4438, max=0.4438
  - cmap_at_5 (len=1): first=0.1432, last=0.1432, min=0.1432, max=0.1432
  - f1_macro (len=1): first=0.1410, last=0.1410, min=0.1410, max=0.1410

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_codegen_retry_01
- exp_002_task_04_validate_code
- exp_002_codegen_retry_02
- exp_002_task_05_validate_code
- exp_002_codegen_retry_03
- exp_002_task_06_validate_code
- exp_002_codegen_retry_04
- exp_002_task_07_validate_code
- exp_002_codegen_retry_05
- exp_002_task_08_validate_code
- exp_002_recovery_01_error_recovery
- exp_002_task_09_validate_code
- exp_002_recovery_02_error_recovery
- exp_002_task_10_validate_code
- exp_002_recovery_03_error_recovery
- exp_002_task_11_validate_code
- exp_002_task_12_execute_training
- exp_002_task_13_capture_metrics
- exp_002_task_14_analyze_results
