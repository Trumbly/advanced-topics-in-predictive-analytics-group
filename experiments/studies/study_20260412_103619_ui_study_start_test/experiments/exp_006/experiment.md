# Experiment exp_006

- **Study:** study_20260412_103619_ui_study_start_test
- **Status:** running
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 08:48:52.059903+00:00
- **Started:** 2026-04-12 08:48:52.059903+00:00

## Model Config
- architecture: `[cnn_gru_hybrid] 2-layer CNN front-end (64 -> 128 channels) followed by a 1-layer GRU(256) for sequence modeling`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.25
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_006_task_01_propose_architecture
- exp_006_task_02_generate_code
- exp_006_task_03_validate_code
- exp_006_codegen_retry_01
- exp_006_task_04_validate_code
- exp_006_codegen_retry_02
- exp_006_task_05_validate_code
- exp_006_codegen_retry_03
- exp_006_task_06_validate_code
- exp_006_recovery_01_error_recovery
- exp_006_task_07_validate_code
- exp_006_recovery_02_error_recovery
- exp_006_task_08_validate_code
- exp_006_task_09_execute_training
