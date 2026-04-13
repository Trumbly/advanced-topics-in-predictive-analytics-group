# Experiment exp_013

- **Study:** study_20260413_021453_baseline_run_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 05:29:34.759971+00:00
- **Started:** 2026-04-13 05:29:34.759971+00:00
- **Completed:** 2026-04-13 05:35:08.628330+00:00

## Model Config
- architecture: `[efficientnet_b0] EfficientNet-B0 via TorchvisionAdapter, full backbone training from scratch`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Results
- cmap_at_5: 0.2453
- f1_macro: 0.0881
- loss: 1.1573
- roc_auc_macro: 0.6789
- duration: 7.4s
- training curves:
  - loss (len=1): first=1.1573, last=1.1573, min=1.1573, max=1.1573
  - roc_auc_macro (len=1): first=0.6789, last=0.6789, min=0.6789, max=0.6789
  - cmap_at_5 (len=1): first=0.2453, last=0.2453, min=0.2453, max=0.2453
  - f1_macro (len=1): first=0.0881, last=0.0881, min=0.0881, max=0.0881

## Tasks
- exp_013_task_01_propose_architecture
- exp_013_task_02_generate_code
- exp_013_task_03_validate_code
- exp_013_codegen_retry_01
- exp_013_task_04_validate_code
- exp_013_task_05_execute_training
- exp_013_task_06_capture_metrics
- exp_013_task_07_analyze_results
