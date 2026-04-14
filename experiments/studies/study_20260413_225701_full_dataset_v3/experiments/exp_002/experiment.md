# Experiment exp_002

- **Study:** study_20260413_225701_full_dataset_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-14 02:35:39.112455+00:00
- **Started:** 2026-04-14 02:35:39.112455+00:00
- **Completed:** 2026-04-14 03:22:38.390493+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNetV3-Small via TorchvisionAdapter, baseline head`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 7
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Results
- cmap_at_5: 0.6152
- f1_macro: 0.3232
- loss: 0.1231
- roc_auc_macro: 0.9831
- duration: 2640.6s
- training curves:
  - loss (len=7): first=0.3461, last=0.1231, min=0.1231, max=0.3461
  - roc_auc_macro (len=7): first=0.9347, last=0.9831, min=0.9347, max=0.9831
  - cmap_at_5 (len=7): first=0.2395, last=0.6152, min=0.2395, max=0.6152
  - f1_macro (len=7): first=0.1546, last=0.3232, min=0.1546, max=0.3232

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
- exp_002_task_06_analyze_results
