# Experiment exp_010

- **Study:** study_20260412_142736_nemotron_3
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 14:21:17.049804+00:00
- **Started:** 2026-04-12 14:21:17.049804+00:00
- **Completed:** 2026-04-12 14:26:05.118868+00:00

## Model Config
- architecture: `[mobilenet_v3_small] Lightweight MobileNet-V3 Small backbone with bottleneck attention`
- pretrained: `mobilenet_v3_small`
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
- cmap_at_5: 0.0124
- f1_macro: 0.0036
- loss: 0.3270
- roc_auc_macro: 0.6929
- duration: 154.0s
- training curves:
  - loss (len=1): first=0.3270, last=0.3270, min=0.3270, max=0.3270
  - roc_auc_macro (len=1): first=0.6929, last=0.6929, min=0.6929, max=0.6929
  - cmap_at_5 (len=1): first=0.0124, last=0.0124, min=0.0124, max=0.0124
  - f1_macro (len=1): first=0.0036, last=0.0036, min=0.0036, max=0.0036

## Tasks
- exp_010_task_01_propose_architecture
- exp_010_task_02_generate_code
- exp_010_task_03_validate_code
- exp_010_task_04_execute_training
- exp_010_recovery_01_error_recovery
- exp_010_task_05_validate_code
- exp_010_task_06_execute_training
- exp_010_task_07_capture_metrics
- exp_010_recovery_02_error_recovery
- exp_010_task_08_validate_code
- exp_010_task_09_execute_training
- exp_010_task_10_capture_metrics
- exp_010_task_11_analyze_results
