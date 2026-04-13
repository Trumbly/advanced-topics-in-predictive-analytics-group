# Experiment exp_007

- **Study:** study_20260411_090326_first_run
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:55:59.420193+00:00
- **Started:** 2026-04-11 09:55:59.420193+00:00
- **Completed:** 2026-04-11 10:09:42.990682+00:00

## Model Config
- architecture: `custom 3-conv CNN with fused residual blocks and SE attention`
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
  - mixup: 0.3
  - specaugment: False

## Results
- loss: 0.0382
- roc_auc_macro: 0.5613
- duration: 755.0s
- training curves:
  - loss (len=1): first=0.0382, last=0.0382, min=0.0382, max=0.0382
  - roc_auc_macro (len=1): first=0.5613, last=0.5613, min=0.5613, max=0.5613

## Tasks
- exp_007_task_01_propose_architecture
- exp_007_task_02_generate_code
- exp_007_task_03_validate_code
- exp_007_task_04_execute_training
- exp_007_task_05_capture_metrics
- exp_007_task_06_analyze_results
