# Experiment exp_006

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:39:01.737624+00:00
- **Started:** 2026-04-11 09:39:01.737624+00:00
- **Completed:** 2026-04-11 09:55:59.417033+00:00

## Model Config
- architecture: `custom 3-conv CNN with SE attention and dropout`
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
  - mixup: 0.2
  - specaugment: False

## Results
- loss: 0.0847
- roc_auc_macro: 0.4882
- duration: 964.8s
- training curves:
  - loss (len=1): first=0.0847, last=0.0847, min=0.0847, max=0.0847
  - roc_auc_macro (len=1): first=0.4882, last=0.4882, min=0.4882, max=0.4882

## Tasks
- exp_006_task_01_propose_architecture
- exp_006_task_02_generate_code
- exp_006_task_03_validate_code
- exp_006_task_04_execute_training
- exp_006_task_05_capture_metrics
- exp_006_task_06_analyze_results
