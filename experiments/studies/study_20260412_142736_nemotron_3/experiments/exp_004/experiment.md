# Experiment exp_004

- **Study:** study_20260412_142736_nemotron_3
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 12:39:45.768679+00:00
- **Started:** 2026-04-12 12:39:45.768679+00:00
- **Completed:** 2026-04-12 12:45:24.130775+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 3-conv CNN front-end feeding 1-layer GRU(128)`
- pretrained: `null`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.05
  - specaugment: False

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_codegen_retry_01
- exp_004_task_04_validate_code
- exp_004_task_05_execute_training
- exp_004_recovery_01_error_recovery
- exp_004_task_06_validate_code
- exp_004_task_07_execute_training
- exp_004_recovery_02_error_recovery
- exp_004_task_08_validate_code
- exp_004_codegen_retry_02
- exp_004_task_09_validate_code
- exp_004_task_10_execute_training
- exp_004_recovery_03_error_recovery
- exp_004_task_11_validate_code
- exp_004_task_12_execute_training
- exp_004_recovery_04_error_recovery
- exp_004_task_13_validate_code
- exp_004_task_14_execute_training
- exp_004_recovery_05_error_recovery
- exp_004_task_15_validate_code
- exp_004_task_16_execute_training
