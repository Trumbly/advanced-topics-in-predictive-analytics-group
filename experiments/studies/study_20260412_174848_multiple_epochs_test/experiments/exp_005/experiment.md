# Experiment exp_005

- **Study:** study_20260412_174848_multiple_epochs_test
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 19:36:43.341862+00:00
- **Started:** 2026-04-12 19:36:43.341862+00:00
- **Completed:** 2026-04-12 21:37:43.317116+00:00

## Model Config
- architecture: `[custom_cnn] 4-conv from-scratch with BatchNorm, SE attention, and multi-scale pooling`
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
  - mixup: 0.2
  - specaugment: True

## Results
- cmap_at_5: 0.6746
- f1_macro: 0.4353
- loss: 0.1244
- roc_auc_macro: 0.9806
- duration: 7149.9s
- training curves:
  - loss (len=5): first=0.3613, last=0.1244, min=0.1244, max=0.3613
  - roc_auc_macro (len=5): first=0.9250, last=0.9806, min=0.9250, max=0.9806
  - cmap_at_5 (len=5): first=0.2307, last=0.6746, min=0.2307, max=0.6746
  - f1_macro (len=5): first=0.1428, last=0.4353, min=0.1428, max=0.4353

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_task_05_capture_metrics
- exp_005_recovery_01_error_recovery
- exp_005_task_06_validate_code
- exp_005_task_07_execute_training
- exp_005_task_08_capture_metrics
- exp_005_task_09_analyze_results
