# Experiment exp_004

- **Study:** study_20260410_131041_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:12:11.325818+00:00
- **Started:** 2026-04-10 13:12:11.325818+00:00
- **Completed:** 2026-04-10 13:12:38.165939+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.0001
  - batch_size: 256
  - epochs: 80
  - optimizer: adam
  - weight_decay: 0.0002
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: True

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
