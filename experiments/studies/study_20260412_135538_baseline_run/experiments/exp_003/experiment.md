# Experiment exp_003

- **Study:** study_20260412_135538_baseline_run
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 14:03:16.802067+00:00
- **Started:** 2026-04-12 14:03:16.802067+00:00
- **Completed:** 2026-04-12 14:20:18.253558+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end (Conv(32) -> BatchNorm -> ReLU) followed by Temporal Pooling, feeding into a single-layer GRU(128) with a final linear projection`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.0
  - specaugment: False

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
