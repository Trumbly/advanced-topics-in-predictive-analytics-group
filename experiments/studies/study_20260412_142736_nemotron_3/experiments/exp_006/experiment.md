# Experiment exp_006

- **Study:** study_20260412_142736_nemotron_3
- **Status:** completed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 12:50:41.994389+00:00
- **Started:** 2026-04-12 12:50:41.994389+00:00
- **Completed:** 2026-04-12 13:47:13.061860+00:00

## Model Config
- architecture: `[cnn_attention] 4-conv CNN + self-attention → Linear`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Results
- cmap_at_5: 0.0198
- f1_macro: 0.0255
- loss: 0.5050
- roc_auc_macro: 0.6837
- duration: 3271.9s
- training curves:
  - loss (len=1): first=0.5050, last=0.5050, min=0.5050, max=0.5050
  - roc_auc_macro (len=1): first=0.6837, last=0.6837, min=0.6837, max=0.6837
  - cmap_at_5 (len=1): first=0.0198, last=0.0198, min=0.0198, max=0.0198
  - f1_macro (len=1): first=0.0255, last=0.0255, min=0.0255, max=0.0255

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
