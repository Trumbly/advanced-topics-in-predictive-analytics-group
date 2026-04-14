# Experiment exp_002

- **Study:** study_20260414_211157_full_dataset_v4
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-14 22:44:46.681399+00:00
- **Started:** 2026-04-14 22:44:46.681399+00:00
- **Completed:** 2026-04-14 23:08:17.883233+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNet-V3 Small via TorchvisionAdapter, baseline head`
- pretrained: `mobilenet_v3_small`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 4
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: True

## Results
- cmap_at_5: 0.5508
- f1_macro: 0.5568
- loss: 0.1570
- roc_auc_macro: 0.9796
- duration: 1209.9s
- training curves:
  - loss (len=3): first=0.3042, last=0.1570, min=0.1570, max=0.3042
  - f1_macro (len=3): first=0.3384, last=0.5568, min=0.3384, max=0.5568
  - roc_auc_macro (len=3): first=0.9499, last=0.9796, min=0.9499, max=0.9796
  - cmap_at_5 (len=3): first=0.2892, last=0.5508, min=0.2892, max=0.5508

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
- exp_002_task_06_analyze_results
