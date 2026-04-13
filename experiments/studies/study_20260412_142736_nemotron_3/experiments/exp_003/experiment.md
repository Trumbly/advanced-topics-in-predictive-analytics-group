# Experiment exp_003

- **Study:** study_20260412_142736_nemotron_3
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 12:34:45.581047+00:00
- **Started:** 2026-04-12 12:34:45.581047+00:00
- **Completed:** 2026-04-12 12:39:45.768004+00:00

## Model Config
- architecture: `[cnn_attention] 2-conv CNN front-end with self-attention over channels`
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
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
- exp_003_recovery_01_error_recovery
- exp_003_task_06_validate_code
- exp_003_codegen_retry_01
- exp_003_task_07_validate_code
- exp_003_task_08_execute_training
- exp_003_recovery_02_error_recovery
- exp_003_task_09_validate_code
- exp_003_task_10_execute_training
- exp_003_recovery_03_error_recovery
- exp_003_task_11_validate_code
- exp_003_task_12_execute_training
- exp_003_recovery_04_error_recovery
- exp_003_task_13_validate_code
- exp_003_task_14_execute_training
- exp_003_recovery_05_error_recovery
- exp_003_task_15_validate_code
- exp_003_task_16_execute_training
