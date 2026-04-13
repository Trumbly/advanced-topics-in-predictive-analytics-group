# Experiment exp_004

- **Study:** study_20260412_093854_ui_study_start_test
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 07:46:04.099731+00:00
- **Started:** 2026-04-12 07:46:04.099731+00:00
- **Completed:** 2026-04-12 07:48:44.335750+00:00

## Model Config
- architecture: `[cnn_attention] EfficientNet-B0 backbone feeding into Self-Attention (Time-Domain) Encoder block`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.0008
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.25
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.5
  - specaugment: True

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_recovery_01_error_recovery
- exp_004_task_04_validate_code
- exp_004_recovery_02_error_recovery
- exp_004_task_05_validate_code
- exp_004_task_06_execute_training
