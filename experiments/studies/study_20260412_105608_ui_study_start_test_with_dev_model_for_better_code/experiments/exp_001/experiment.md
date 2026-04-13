# Experiment exp_001

- **Study:** study_20260412_105608_ui_study_start_test_with_dev_model_for_better_code
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 08:56:08.968735+00:00
- **Started:** 2026-04-12 08:56:08.968735+00:00
- **Completed:** 2026-04-12 09:07:23.687467+00:00

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
- loss: 0.4795
- roc_auc_macro: 0.7957
- duration: 604.3s
- training curves:
  - loss (len=1): first=0.4795, last=0.4795, min=0.4795, max=0.4795
  - roc_auc_macro (len=1): first=0.7957, last=0.7957, min=0.7957, max=0.7957

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
