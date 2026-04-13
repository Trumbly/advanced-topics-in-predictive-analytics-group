# Experiment exp_003

- **Study:** study_20260410_135338_first_run
- **Status:** timeout
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:55:45.429687+00:00
- **Started:** 2026-04-10 13:55:45.429687+00:00
- **Completed:** 2026-04-10 14:06:38.612932+00:00

## Model Config
- architecture: `cnn_small_v1 with dropout and spectral augmentation`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 50
  - optimizer: Adam
  - weight_decay: 1e-05
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: True

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
