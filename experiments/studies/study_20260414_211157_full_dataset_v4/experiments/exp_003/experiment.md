# Experiment exp_003

- **Study:** study_20260414_211157_full_dataset_v4
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-14 23:08:17.885817+00:00
- **Started:** 2026-04-14 23:08:17.885817+00:00
- **Completed:** 2026-04-14 23:25:26.173934+00:00

## Model Config
- architecture: `[cnn_attention] 4-Conv block front-end (32->64->128) -> Temporal Self-Attention -> Global Average Pooling`
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
  - mixup: 0.0
  - specaugment: True

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
- exp_003_recovery_02_error_recovery
- exp_003_task_09_validate_code
- exp_003_task_10_execute_training
- exp_003_task_11_capture_metrics
- exp_003_recovery_03_error_recovery
- exp_003_task_12_validate_code
- exp_003_task_13_execute_training
- exp_003_task_14_capture_metrics
- exp_003_recovery_04_error_recovery
- exp_003_task_15_validate_code
- exp_003_task_16_execute_training
- exp_003_task_17_capture_metrics
- exp_003_recovery_05_error_recovery
- exp_003_task_18_validate_code
- exp_003_task_19_execute_training
- exp_003_task_20_capture_metrics
