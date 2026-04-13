# Experiment exp_007

- **Study:** study_20260412_142736_nemotron_3
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 13:47:13.063936+00:00
- **Started:** 2026-04-12 13:47:13.063936+00:00
- **Completed:** 2026-04-12 13:56:17.114342+00:00

## Model Config
- architecture: `[resnet18] ResNet-18 via TorchvisionAdapter, baseline head`
- pretrained: `resnet18`
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
  - mixup: 0.2
  - specaugment: False

## Results
- cmap_at_5: 0.3195
- f1_macro: 0.2256
- loss: 0.2853
- roc_auc_macro: 0.9508
- duration: 408.9s
- training curves:
  - loss (len=1): first=0.2853, last=0.2853, min=0.2853, max=0.2853
  - roc_auc_macro (len=1): first=0.9508, last=0.9508, min=0.9508, max=0.9508
  - cmap_at_5 (len=1): first=0.3195, last=0.3195, min=0.3195, max=0.3195
  - f1_macro (len=1): first=0.2256, last=0.2256, min=0.2256, max=0.2256

## Tasks
- exp_007_task_01_propose_architecture
- exp_007_task_02_generate_code
- exp_007_task_03_validate_code
- exp_007_task_04_execute_training
- exp_007_task_05_capture_metrics
- exp_007_recovery_01_error_recovery
- exp_007_task_06_validate_code
- exp_007_task_07_execute_training
- exp_007_task_08_capture_metrics
- exp_007_recovery_02_error_recovery
- exp_007_task_09_validate_code
- exp_007_task_10_execute_training
- exp_007_task_11_capture_metrics
- exp_007_task_12_analyze_results
