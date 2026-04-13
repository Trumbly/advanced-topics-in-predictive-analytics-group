# Experiment exp_001

- **Study:** study_20260412_093854_ui_study_start_test
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 07:38:55.367664+00:00
- **Started:** 2026-04-12 07:38:55.367664+00:00
- **Completed:** 2026-04-12 07:43:06.788450+00:00

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
- loss: 0.5097
- roc_auc_macro: 0.7144
- duration: 158.5s
- training curves:
  - loss (len=1): first=0.5097, last=0.5097, min=0.5097, max=0.5097
  - roc_auc_macro (len=1): first=0.7144, last=0.7144, min=0.7144, max=0.7144

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
