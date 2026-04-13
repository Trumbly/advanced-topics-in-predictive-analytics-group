# Experiment exp_004

- **Study:** study_20260412_174848_multiple_epochs_test
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 18:59:33.850314+00:00
- **Started:** 2026-04-12 18:59:33.850314+00:00
- **Completed:** 2026-04-12 19:36:43.339243+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end feeding a 2-layer GRU(256) with dropout, followed by a linear head with sigmoid output`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.4
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.5
  - specaugment: True

## Results
- cmap_at_5: 0.0717
- f1_macro: 0.0718
- loss: 0.3517
- roc_auc_macro: 0.8639
- duration: 2109.6s
- training curves:
  - loss (len=5): first=0.5168, last=0.3517, min=0.3517, max=0.5168
  - roc_auc_macro (len=5): first=0.6755, last=0.8639, min=0.6755, max=0.8768
  - cmap_at_5 (len=5): first=0.0143, last=0.0717, min=0.0143, max=0.0869
  - f1_macro (len=5): first=0.0109, last=0.0718, min=0.0109, max=0.0770

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
- exp_004_task_05_capture_metrics
- exp_004_recovery_01_error_recovery
- exp_004_task_06_validate_code
- exp_004_task_07_execute_training
- exp_004_task_08_capture_metrics
- exp_004_task_09_analyze_results
