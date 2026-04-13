# Experiment exp_014

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:25:51.765701+00:00
- **Started:** 2026-04-11 10:25:51.765701+00:00
- **Completed:** 2026-04-11 10:39:32.706242+00:00

## Model Config
- architecture: `3-conv CNN with fused residual blocks, SE attention, and mixup augmentation`
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
  - mixup: 0.2
  - specaugment: False

## Results
- loss: 0.0428
- roc_auc_macro: 0.4938
- duration: 755.5s
- training curves:
  - loss (len=1): first=0.0428, last=0.0428, min=0.0428, max=0.0428
  - roc_auc_macro (len=1): first=0.4938, last=0.4938, min=0.4938, max=0.4938

## Tasks
- exp_014_task_01_propose_architecture
- exp_014_task_02_generate_code
- exp_014_task_03_validate_code
- exp_014_task_04_execute_training
- exp_014_task_05_capture_metrics
- exp_014_task_06_analyze_results
