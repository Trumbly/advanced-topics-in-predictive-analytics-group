# Experiment exp_001

- **Study:** study_20260412_135538_baseline_run
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 13:55:42.633555+00:00
- **Started:** 2026-04-12 13:55:42.633555+00:00
- **Completed:** 2026-04-12 14:02:35.569861+00:00

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
- cmap_at_5: 0.2183
- f1_macro: 0.1235
- loss: 1.1567
- roc_auc_macro: 0.6999
- duration: 13.8s
- training curves:
  - loss (len=1): first=1.1567, last=1.1567, min=1.1567, max=1.1567
  - roc_auc_macro (len=1): first=0.6999, last=0.6999, min=0.6999, max=0.6999
  - cmap_at_5 (len=1): first=0.2183, last=0.2183, min=0.2183, max=0.2183
  - f1_macro (len=1): first=0.1235, last=0.1235, min=0.1235, max=0.1235

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_recovery_01_error_recovery
- exp_001_task_06_validate_code
- exp_001_task_07_execute_training
- exp_001_task_08_capture_metrics
- exp_001_task_09_analyze_results
