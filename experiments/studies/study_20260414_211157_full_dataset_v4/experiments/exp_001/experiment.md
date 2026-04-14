# Experiment exp_001

- **Study:** study_20260414_211157_full_dataset_v4
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-14 21:11:58.400140+00:00
- **Started:** 2026-04-14 21:11:58.400140+00:00
- **Completed:** 2026-04-14 22:44:46.678766+00:00

## Model Config
- architecture: `[efficientnet_b0] pretrained EfficientNet-B0 baseline`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 4
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Results
- cmap_at_5: 0.5929
- f1_macro: 0.5858
- loss: 0.1414
- roc_auc_macro: 0.9825
- duration: 5413.9s
- training curves:
  - loss (len=3): first=0.3188, last=0.1414, min=0.1414, max=0.3188
  - f1_macro (len=3): first=0.3371, last=0.5858, min=0.3371, max=0.5858
  - roc_auc_macro (len=3): first=0.9511, last=0.9825, min=0.9511, max=0.9825
  - cmap_at_5 (len=3): first=0.2968, last=0.5929, min=0.2968, max=0.5929

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
