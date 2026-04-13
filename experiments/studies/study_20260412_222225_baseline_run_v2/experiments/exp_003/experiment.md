# Experiment exp_003

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 22:42:05.771315+00:00
- **Started:** 2026-04-12 22:42:05.771315+00:00
- **Completed:** 2026-04-12 22:54:41.723286+00:00

## Model Config
- architecture: `[cnn_attention] 3-stage approach: 1. Small 3-conv CNN front-end (extract local features) -> 2. Self-Attention Pooling Block (aggregate global context) -> 3. Linear head for multi-label output`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.3
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
