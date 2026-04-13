# Experiment exp_005

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:25:13.995990+00:00
- **Started:** 2026-04-11 09:25:13.995990+00:00
- **Completed:** 2026-04-11 09:39:01.734565+00:00

## Model Config
- architecture: `custom 3-conv CNN with dropout`
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
  - mixup: 0.2
  - specaugment: False

## Results
- loss: 0.0417
- roc_auc_macro: 0.5144
- duration: 776.4s
- training curves:
  - loss (len=1): first=0.0417, last=0.0417, min=0.0417, max=0.0417
  - roc_auc_macro (len=1): first=0.5144, last=0.5144, min=0.5144, max=0.5144

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_task_05_capture_metrics
- exp_005_task_06_analyze_results
