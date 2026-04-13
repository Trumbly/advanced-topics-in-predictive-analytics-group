# Experiment exp_004

- **Study:** study_20260410_131833_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:20:09.981004+00:00
- **Started:** 2026-04-10 13:20:09.981004+00:00
- **Completed:** 2026-04-10 13:20:45.600052+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 30
  - optimizer: Adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.2
  - specaugment: True

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
