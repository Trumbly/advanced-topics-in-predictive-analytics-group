# Experiment exp_005

- **Study:** study_20260410_131041_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:12:38.167369+00:00
- **Started:** 2026-04-10 13:12:38.167369+00:00
- **Completed:** 2026-04-10 13:13:18.768959+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 5e-05
  - batch_size: 256
  - epochs: 60
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
