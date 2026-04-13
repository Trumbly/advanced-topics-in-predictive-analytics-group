# Experiment exp_018

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:47:18.772397+00:00
- **Started:** 2026-04-11 10:47:18.772397+00:00
- **Completed:** 2026-04-11 10:50:22.943412+00:00

## Model Config
- architecture: `cnn_small_v1 baseline with dropout`
- pretrained: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0
  - specaugment: False

## Results
- loss: 0.1050
- roc_auc_macro: 0.5253
- duration: 0.0s
- training curves:
  - loss (len=1): first=0.1050, last=0.1050, min=0.1050, max=0.1050
  - roc_auc_macro (len=1): first=0.5253, last=0.5253, min=0.5253, max=0.5253

## Tasks
- exp_018_task_01_propose_architecture
- exp_018_task_02_generate_code
- exp_018_task_03_validate_code
- exp_018_task_04_execute_training
- exp_018_recovery_01_error_recovery
- exp_018_task_05_validate_code
- exp_018_task_06_execute_training
- exp_018_task_07_capture_metrics
- exp_018_task_08_analyze_results
