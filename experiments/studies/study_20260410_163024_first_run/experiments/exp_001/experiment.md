# Experiment exp_001

- **Study:** study_20260410_163024_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 16:30:24.661820+00:00
- **Started:** 2026-04-10 16:30:24.661820+00:00
- **Completed:** 2026-04-10 17:13:19.444486+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 2
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Results
- loss: 0.0257
- roc_auc_macro: 0.8126
- duration: 2544.8s
- training curves:
  - loss (len=2): first=0.0325, last=0.0257, min=0.0257, max=0.0325
  - roc_auc_macro (len=2): first=0.6936, last=0.8126, min=0.6936, max=0.8126

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
