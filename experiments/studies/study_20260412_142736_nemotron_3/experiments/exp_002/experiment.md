# Experiment exp_002

- **Study:** study_20260412_142736_nemotron_3
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 12:30:02.962241+00:00
- **Started:** 2026-04-12 12:30:02.962241+00:00
- **Completed:** 2026-04-12 12:34:45.580292+00:00

## Model Config
- architecture: `[mobilenet_v3_small] lightweight Conv2d backbone (1M params) with LazyLinear head`
- pretrained: `mobilenet_v3_small`
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
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_codegen_retry_01
- exp_002_task_04_validate_code
- exp_002_task_05_execute_training
- exp_002_recovery_01_error_recovery
- exp_002_task_06_validate_code
- exp_002_task_07_execute_training
- exp_002_recovery_02_error_recovery
- exp_002_task_08_validate_code
- exp_002_task_09_execute_training
- exp_002_recovery_03_error_recovery
- exp_002_task_10_validate_code
- exp_002_task_11_execute_training
- exp_002_recovery_04_error_recovery
- exp_002_task_12_validate_code
- exp_002_task_13_execute_training
- exp_002_recovery_05_error_recovery
- exp_002_task_14_validate_code
- exp_002_task_15_execute_training
