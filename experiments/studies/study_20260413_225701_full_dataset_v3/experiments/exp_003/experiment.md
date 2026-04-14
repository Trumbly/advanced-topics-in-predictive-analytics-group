# Experiment exp_003

- **Study:** study_20260413_225701_full_dataset_v3
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-14 03:22:38.391715+00:00
- **Started:** 2026-04-14 03:22:38.391715+00:00
- **Completed:** 2026-04-14 04:22:07.682164+00:00

## Model Config
- architecture: `[cnn_attention] 3-conv CNN feature extractor feeding into a self-attention block, followed by Global Average Pooling`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 7
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Results
- cmap_at_5: 0.1860
- f1_macro: 0.1406
- loss: 0.3059
- roc_auc_macro: 0.9176
- duration: 3358.4s
- training curves:
  - loss (len=7): first=0.4718, last=0.3059, min=0.3059, max=0.4718
  - roc_auc_macro (len=7): first=0.7980, last=0.9176, min=0.7980, max=0.9176
  - cmap_at_5 (len=7): first=0.0418, last=0.1860, min=0.0418, max=0.1860
  - f1_macro (len=7): first=0.0400, last=0.1406, min=0.0400, max=0.1406

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
- exp_003_task_06_analyze_results
