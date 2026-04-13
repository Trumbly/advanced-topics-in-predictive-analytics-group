# Experiment exp_001

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 02:14:53.660091+00:00
- **Started:** 2026-04-13 02:14:53.660091+00:00
- **Completed:** 2026-04-13 02:19:15.026474+00:00

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
- cmap_at_5: 0.1699
- f1_macro: 0.0595
- loss: 1.2104
- roc_auc_macro: 0.5877
- duration: 114.2s
- training curves:
  - loss (len=1): first=1.2104, last=1.2104, min=1.2104, max=1.2104
  - roc_auc_macro (len=1): first=0.5877, last=0.5877, min=0.5877, max=0.5877
  - cmap_at_5 (len=1): first=0.1699, last=0.1699, min=0.1699, max=0.1699
  - f1_macro (len=1): first=0.0595, last=0.0595, min=0.0595, max=0.0595

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
