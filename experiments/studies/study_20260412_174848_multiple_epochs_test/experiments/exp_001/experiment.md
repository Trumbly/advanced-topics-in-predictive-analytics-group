# Experiment exp_001

- **Study:** study_20260412_174848_multiple_epochs_test
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 15:48:48.833928+00:00
- **Started:** 2026-04-12 15:48:48.833928+00:00
- **Completed:** 2026-04-12 16:32:38.314126+00:00

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
- cmap_at_5: 0.1941
- f1_macro: 0.1522
- loss: 0.3019
- roc_auc_macro: 0.9168
- duration: 2576.4s
- training curves:
  - loss (len=5): first=0.4805, last=0.3019, min=0.3019, max=0.4805
  - roc_auc_macro (len=5): first=0.7910, last=0.9168, min=0.7910, max=0.9168
  - cmap_at_5 (len=5): first=0.0466, last=0.1941, min=0.0466, max=0.1941
  - f1_macro (len=5): first=0.0468, last=0.1522, min=0.0468, max=0.1522

## Tasks
- exp_001_task_01_propose_architecture
- exp_001_task_02_generate_code
- exp_001_task_03_validate_code
- exp_001_task_04_execute_training
- exp_001_task_05_capture_metrics
- exp_001_task_06_analyze_results
