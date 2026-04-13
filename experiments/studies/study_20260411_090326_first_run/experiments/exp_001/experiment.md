# Experiment exp_001

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:03:26.670130+00:00
- **Started:** 2026-04-11 09:03:26.670130+00:00
- **Completed:** 2026-04-11 09:06:32.974509+00:00

## Model Config
- architecture: `cnn_small_v1`
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
- loss: 0.1055
- roc_auc_macro: 0.5081
- duration: 128.4s
- training curves:
  - loss (len=1): first=0.1055, last=0.1055, min=0.1055, max=0.1055
  - roc_auc_macro (len=1): first=0.5081, last=0.5081, min=0.5081, max=0.5081

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
