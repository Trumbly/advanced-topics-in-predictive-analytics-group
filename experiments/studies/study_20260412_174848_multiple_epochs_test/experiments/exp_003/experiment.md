# Experiment exp_003

- **Study:** study_20260412_174848_multiple_epochs_test
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 17:31:08.537979+00:00
- **Started:** 2026-04-12 17:31:08.537979+00:00
- **Completed:** 2026-04-12 18:59:33.847845+00:00

## Model Config
- architecture: `[deep_cnn] 5-conv deep CNN with residual connections, BatchNorm, and SE attention blocks, followed by adaptive pooling and sigmoid output`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.3
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.5
  - specaugment: True

## Results
- cmap_at_5: 0.6258
- f1_macro: 0.4325
- loss: 0.1284
- roc_auc_macro: 0.9797
- duration: 5247.8s
- training curves:
  - loss (len=5): first=0.3378, last=0.1284, min=0.1284, max=0.3378
  - roc_auc_macro (len=5): first=0.9318, last=0.9797, min=0.9318, max=0.9797
  - cmap_at_5 (len=5): first=0.2783, last=0.6258, min=0.2783, max=0.6258
  - f1_macro (len=5): first=0.1913, last=0.4325, min=0.1913, max=0.4325

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
- exp_003_task_06_analyze_results
