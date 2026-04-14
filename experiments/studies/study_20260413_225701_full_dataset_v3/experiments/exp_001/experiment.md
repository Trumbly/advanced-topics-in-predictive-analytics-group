# Experiment exp_001

- **Study:** study_20260413_225701_full_dataset_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 22:57:01.775681+00:00
- **Started:** 2026-04-13 22:57:01.775681+00:00
- **Completed:** 2026-04-14 02:35:39.108058+00:00

## Model Config
- architecture: `[efficientnet_b0] pretrained EfficientNet-B0 baseline`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 7
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Results
- cmap_at_5: 0.8058
- f1_macro: 0.5248
- loss: 0.0460
- roc_auc_macro: 0.9901
- duration: 12960.9s
- training curves:
  - loss (len=7): first=0.2921, last=0.0460, min=0.0460, max=0.2921
  - roc_auc_macro (len=7): first=0.9580, last=0.9901, min=0.9580, max=0.9901
  - cmap_at_5 (len=7): first=0.3585, last=0.8058, min=0.3585, max=0.8058
  - f1_macro (len=7): first=0.1905, last=0.5248, min=0.1905, max=0.5248

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
