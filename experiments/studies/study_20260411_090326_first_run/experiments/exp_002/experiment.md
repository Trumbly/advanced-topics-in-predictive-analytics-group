# Experiment exp_002

- **Study:** study_20260411_090326_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-11 09:06:32.975400+00:00
- **Started:** 2026-04-11 09:06:32.975400+00:00
- **Completed:** 2026-04-11 09:08:43.695330+00:00

## Model Config
- architecture: `custom 3-conv SE baseline`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
- exp_002_recovery_01_error_recovery
