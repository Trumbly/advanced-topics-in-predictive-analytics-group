# Experiment exp_013

- **Study:** study_20260412_222225_baseline_run_v2
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-13 00:52:26.853675+00:00
- **Started:** 2026-04-13 00:52:26.853675+00:00
- **Completed:** 2026-04-13 01:18:05.671785+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] Lightweight 3-conv stack (64 channels) -> Flatten -> 1-layer GRU(128) -> Global Average Pooling -> Linear Head`
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
- exp_013_task_01_propose_architecture
- exp_013_task_02_generate_code
- exp_013_task_03_validate_code
- exp_013_task_04_execute_training
- exp_013_task_05_capture_metrics
- exp_013_recovery_01_error_recovery
- exp_013_task_06_validate_code
- exp_013_task_07_execute_training
- exp_013_task_08_capture_metrics
- exp_013_recovery_02_error_recovery
- exp_013_task_09_validate_code
- exp_013_task_10_execute_training
- exp_013_task_11_capture_metrics
- exp_013_recovery_03_error_recovery
- exp_013_task_12_validate_code
- exp_013_task_13_execute_training
- exp_013_task_14_capture_metrics
- exp_013_recovery_04_error_recovery
- exp_013_task_15_validate_code
- exp_013_task_16_execute_training
- exp_013_task_17_capture_metrics
- exp_013_recovery_05_error_recovery
- exp_013_task_18_validate_code
- exp_013_task_19_execute_training
- exp_013_task_20_capture_metrics
