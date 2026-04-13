# Experiment exp_005

- **Study:** study_20260412_105608_ui_study_start_test_with_dev_model_for_better_code
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 09:48:27.855671+00:00
- **Started:** 2026-04-12 09:48:27.855671+00:00
- **Completed:** 2026-04-12 10:13:42.771782+00:00

## Model Config
- architecture: `[custom_cnn] 4-conv from-scratch with BatchNorm, SE attention, and adaptive pooling`
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
- loss: 0.3776
- roc_auc_macro: 0.9110
- duration: 1439.3s
- training curves:
  - loss (len=1): first=0.3776, last=0.3776, min=0.3776, max=0.3776
  - roc_auc_macro (len=1): first=0.9110, last=0.9110, min=0.9110, max=0.9110

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
