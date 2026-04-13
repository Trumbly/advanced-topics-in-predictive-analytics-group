# Experiment exp_001

- **Study:** study_20260413_203901_full_dataset_v1
- **Status:** completed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 20:39:02.298192+00:00
- **Started:** 2026-04-13 20:39:02.298192+00:00
- **Completed:** 2026-04-13 20:50:12.183243+00:00

## Model Config
- architecture: `[custom_cnn] cnn_small_v1 baseline`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

## Results
- cmap_at_5: 0.0601
- f1_macro: 0.0627
- loss: 0.4512
- roc_auc_macro: 0.8298
- duration: 500.2s
- training curves:
  - loss (len=1): first=0.4512, last=0.4512, min=0.4512, max=0.4512
  - roc_auc_macro (len=1): first=0.8298, last=0.8298, min=0.8298, max=0.8298
  - cmap_at_5 (len=1): first=0.0601, last=0.0601, min=0.0601, max=0.0601
  - f1_macro (len=1): first=0.0627, last=0.0627, min=0.0627, max=0.0627

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
