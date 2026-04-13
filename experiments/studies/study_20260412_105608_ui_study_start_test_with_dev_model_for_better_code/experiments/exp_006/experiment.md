# Experiment exp_006

- **Study:** study_20260412_105608_ui_study_start_test_with_dev_model_for_better_code
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 10:13:42.773626+00:00
- **Started:** 2026-04-12 10:13:42.773626+00:00
- **Completed:** 2026-04-12 10:27:42.287853+00:00

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
- loss: 0.4081
- roc_auc_macro: 0.9073
- duration: 748.6s
- training curves:
  - loss (len=1): first=0.4081, last=0.4081, min=0.4081, max=0.4081
  - roc_auc_macro (len=1): first=0.9073, last=0.9073, min=0.9073, max=0.9073

## Tasks
- exp_006_task_01_propose_architecture
- exp_006_task_02_generate_code
- exp_006_task_03_validate_code
- exp_006_task_04_execute_training
- exp_006_task_05_capture_metrics
- exp_006_recovery_01_error_recovery
- exp_006_task_06_validate_code
- exp_006_task_07_execute_training
- exp_006_task_08_capture_metrics
- exp_006_task_09_analyze_results
