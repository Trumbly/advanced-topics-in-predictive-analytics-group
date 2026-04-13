# Experiment exp_020

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 10:53:27.871872+00:00
- **Started:** 2026-04-11 10:53:27.871872+00:00
- **Completed:** 2026-04-11 11:10:09.371112+00:00

## Model Config
- architecture: `Enhanced 3-conv CNN with residual SE attention and dropout`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Results
- loss: 0.0559
- roc_auc_macro: 0.5264
- duration: 944.5s
- training curves:
  - loss (len=1): first=0.0559, last=0.0559, min=0.0559, max=0.0559
  - roc_auc_macro (len=1): first=0.5264, last=0.5264, min=0.5264, max=0.5264

## Tasks
- exp_020_task_01_propose_architecture
- exp_020_task_02_generate_code
- exp_020_task_03_validate_code
- exp_020_task_04_execute_training
- exp_020_task_05_capture_metrics
- exp_020_task_06_analyze_results
