# Experiment exp_008

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:09:42.993914+00:00
- **Started:** 2026-04-11 10:09:42.993914+00:00
- **Completed:** 2026-04-11 10:13:04.003154+00:00

## Model Config
- architecture: `cnn_small_v1`
- pretrained: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Results
- loss: 0.1028
- roc_auc_macro: 0.5447
- duration: 100.4s
- training curves:
  - loss (len=1): first=0.1028, last=0.1028, min=0.1028, max=0.1028
  - roc_auc_macro (len=1): first=0.5447, last=0.5447, min=0.5447, max=0.5447

## Tasks
- exp_008_task_01_propose_architecture
- exp_008_task_02_generate_code
- exp_008_task_03_validate_code
- exp_008_task_04_execute_training
- exp_008_task_05_capture_metrics
- exp_008_recovery_01_error_recovery
- exp_008_task_06_validate_code
- exp_008_task_07_execute_training
- exp_008_task_08_capture_metrics
- exp_008_task_09_analyze_results
