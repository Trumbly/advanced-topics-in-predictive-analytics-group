# Experiment exp_002

- **Study:** study_20260412_174848_multiple_epochs_test
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 16:32:38.315936+00:00
- **Started:** 2026-04-12 16:32:38.315936+00:00
- **Completed:** 2026-04-12 17:31:08.535597+00:00

## Model Config
- architecture: `[cnn_attention] 4-conv CNN with self-attention and SE blocks, followed by adaptive pooling and sigmoid output`
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
  - mixup: 0.2
  - specaugment: True

## Results
- cmap_at_5: 0.4624
- f1_macro: 0.3105
- loss: 0.1836
- roc_auc_macro: 0.9664
- duration: 3452.1s
- training curves:
  - loss (len=5): first=0.3914, last=0.1836, min=0.1836, max=0.3914
  - roc_auc_macro (len=5): first=0.8971, last=0.9664, min=0.8971, max=0.9664
  - cmap_at_5 (len=5): first=0.1489, last=0.4624, min=0.1489, max=0.4624
  - f1_macro (len=5): first=0.1167, last=0.3105, min=0.1167, max=0.3105

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
- exp_002_task_06_analyze_results
