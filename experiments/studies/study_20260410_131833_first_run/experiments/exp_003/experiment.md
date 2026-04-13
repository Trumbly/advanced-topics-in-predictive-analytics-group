# Experiment exp_003

- **Study:** study_20260410_131833_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:19:37.454955+00:00
- **Started:** 2026-04-10 13:19:37.454955+00:00
- **Completed:** 2026-04-10 13:20:09.980211+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.01
  - batch_size: 64
  - epochs: 150
  - optimizer: Adam
  - weight_decay: 0.0
  - dropout: 0.0
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
