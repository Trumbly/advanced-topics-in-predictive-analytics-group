# Experiment exp_002

- **Study:** study_20260413_203901_full_dataset_v1
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 20:50:12.184357+00:00
- **Started:** 2026-04-13 20:50:12.184357+00:00
- **Completed:** 2026-04-13 21:23:53.225661+00:00

## Model Config
- architecture: `[efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, baseline head`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Results
- cmap_at_5: 0.3697
- f1_macro: 0.1800
- loss: 0.2901
- roc_auc_macro: 0.9590
- duration: 1851.4s
- training curves:
  - loss (len=1): first=0.2901, last=0.2901, min=0.2901, max=0.2901
  - roc_auc_macro (len=1): first=0.9590, last=0.9590, min=0.9590, max=0.9590
  - cmap_at_5 (len=1): first=0.3697, last=0.3697, min=0.3697, max=0.3697
  - f1_macro (len=1): first=0.1800, last=0.1800, min=0.1800, max=0.1800

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
- exp_002_task_06_analyze_results
