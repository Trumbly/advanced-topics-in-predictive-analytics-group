# Experiment exp_001

- **Study:** study_20260412_142736_nemotron_3
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 12:27:37.487230+00:00
- **Started:** 2026-04-12 12:27:37.487230+00:00
- **Completed:** 2026-04-12 12:30:02.960776+00:00

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
- cmap_at_5: 0.0147
- f1_macro: 0.0123
- loss: 0.5325
- roc_auc_macro: 0.6538
- duration: 0.0s
- training curves:
  - loss (len=1): first=0.5325, last=0.5325, min=0.5325, max=0.5325
  - roc_auc_macro (len=1): first=0.6538, last=0.6538, min=0.6538, max=0.6538
  - cmap_at_5 (len=1): first=0.0147, last=0.0147, min=0.0147, max=0.0147
  - f1_macro (len=1): first=0.0123, last=0.0123, min=0.0123, max=0.0123

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
