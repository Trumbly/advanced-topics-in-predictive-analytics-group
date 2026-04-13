# Experiment exp_010

- **Study:** study_20260410_163024_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 19:03:00.971217+00:00
- **Started:** 2026-04-10 19:03:00.971217+00:00
- **Completed:** 2026-04-10 21:02:45.002184+00:00

## Model Config
- architecture: `custom 4-block CNN with global average pooling and dropout`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 5
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0
  - specaugment: False

## Results
- loss: 0.0848
- roc_auc_macro: 0.9334
- duration: 7128.4s
- training curves:
  - loss (len=5): first=0.0928, last=0.0848, min=0.0848, max=0.0928
  - roc_auc_macro (len=5): first=0.8600, last=0.9334, min=0.8600, max=0.9334

## Tasks
- exp_010_task_01_propose_architecture
- exp_010_task_02_generate_code
- exp_010_task_03_validate_code
- exp_010_task_04_execute_training
- exp_010_task_05_capture_metrics
- exp_010_task_06_analyze_results
