# Experiment exp_003

- **Study:** study_20260412_105608_ui_study_start_test_with_dev_model_for_better_code
- **Status:** completed
- **LLM:** qwen3-coder:latest
- **Created:** 2026-04-12 09:20:06.508620+00:00
- **Started:** 2026-04-12 09:20:06.508620+00:00
- **Completed:** 2026-04-12 09:45:20.131118+00:00

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
- loss: 0.3476
- roc_auc_macro: 0.9385
- duration: 1436.3s
- training curves:
  - loss (len=1): first=0.3476, last=0.3476, min=0.3476, max=0.3476
  - roc_auc_macro (len=1): first=0.9385, last=0.9385, min=0.9385, max=0.9385

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
- exp_003_recovery_01_error_recovery
- exp_003_task_06_validate_code
- exp_003_task_07_execute_training
- exp_003_task_08_capture_metrics
- exp_003_task_09_analyze_results
