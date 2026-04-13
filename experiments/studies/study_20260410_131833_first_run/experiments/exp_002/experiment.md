# Experiment exp_002

- **Study:** study_20260410_131833_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:19:02.165219+00:00
- **Started:** 2026-04-10 13:19:02.165219+00:00
- **Completed:** 2026-04-10 13:19:37.454327+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 100
  - optimizer: SGD
  - weight_decay: 0.001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
