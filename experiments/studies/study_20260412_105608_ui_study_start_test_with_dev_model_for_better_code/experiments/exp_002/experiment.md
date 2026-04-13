# Experiment exp_002

- **Study:** study_20260412_105608_ui_study_start_test_with_dev_model_for_better_code
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 09:07:23.689068+00:00
- **Started:** 2026-04-12 09:07:23.689068+00:00
- **Completed:** 2026-04-12 09:20:06.506776+00:00

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
- loss: 0.4039
- roc_auc_macro: 0.9066
- duration: 682.0s
- training curves:
  - loss (len=1): first=0.4039, last=0.4039, min=0.4039, max=0.4039
  - roc_auc_macro (len=1): first=0.9066, last=0.9066, min=0.9066, max=0.9066

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
- exp_002_recovery_01_error_recovery
- exp_002_task_06_validate_code
- exp_002_task_07_execute_training
- exp_002_task_08_capture_metrics
- exp_002_task_09_analyze_results
