# Experiment exp_005

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 23:13:34.470083+00:00
- **Started:** 2026-04-12 23:13:34.470083+00:00
- **Completed:** 2026-04-12 23:20:10.522532+00:00

## Model Config
- architecture: `[efficientnet_b0] EfficientNet-B0 backbone adapted for spectrograms, followed by a Global Average Pooling layer for robust feature summarization before the final linear head.`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.15
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.5
  - specaugment: True

## Results
- cmap_at_5: 0.1619
- f1_macro: 0.0862
- loss: 8.4285
- roc_auc_macro: 0.4542
- duration: 6.2s
- training curves:
  - loss (len=1): first=8.4285, last=8.4285, min=8.4285, max=8.4285
  - roc_auc_macro (len=1): first=0.4542, last=0.4542, min=0.4542, max=0.4542
  - cmap_at_5 (len=1): first=0.1619, last=0.1619, min=0.1619, max=0.1619
  - f1_macro (len=1): first=0.0862, last=0.0862, min=0.0862, max=0.0862

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_task_05_capture_metrics
- exp_005_recovery_01_error_recovery
- exp_005_task_06_validate_code
- exp_005_task_07_execute_training
- exp_005_task_08_capture_metrics
- exp_005_task_09_analyze_results
