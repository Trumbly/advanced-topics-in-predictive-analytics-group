# Experiment exp_021

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 11:10:09.374049+00:00
- **Started:** 2026-04-11 11:10:09.374049+00:00
- **Completed:** 2026-04-11 11:12:06.061453+00:00

## Model Config
- architecture: `ResNet-style 4-block CNN with fused SE attention and adaptive dropout`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.4
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0
  - specaugment: False

## Tasks
- exp_021_task_01_propose_architecture
- exp_021_task_02_generate_code
- exp_021_task_03_validate_code
- exp_021_task_04_execute_training
- exp_021_task_05_capture_metrics
- exp_021_recovery_01_error_recovery
- exp_021_task_06_validate_code
- exp_021_task_07_execute_training
- exp_021_recovery_02_error_recovery
- exp_021_task_08_validate_code
- exp_021_task_09_execute_training
