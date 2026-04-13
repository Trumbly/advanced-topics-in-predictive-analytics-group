# Experiment exp_005

- **Study:** study_20260412_142736_nemotron_3
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-12 12:45:24.131560+00:00
- **Started:** 2026-04-12 12:45:24.131560+00:00
- **Completed:** 2026-04-12 12:50:41.993228+00:00

## Model Config
- architecture: `[mobilenet_v3_small] MobileNet-V3 Small via TorchvisionAdapter (lightweight backbone)`
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
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_codegen_retry_01
- exp_005_task_04_validate_code
- exp_005_task_05_execute_training
- exp_005_recovery_01_error_recovery
- exp_005_task_06_validate_code
- exp_005_task_07_execute_training
- exp_005_recovery_02_error_recovery
- exp_005_task_08_validate_code
- exp_005_task_09_execute_training
- exp_005_recovery_03_error_recovery
- exp_005_task_10_validate_code
- exp_005_task_11_execute_training
- exp_005_recovery_04_error_recovery
- exp_005_task_12_validate_code
- exp_005_task_13_execute_training
- exp_005_recovery_05_error_recovery
- exp_005_task_14_validate_code
- exp_005_task_15_execute_training
