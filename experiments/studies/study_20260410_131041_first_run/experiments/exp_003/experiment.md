# Experiment exp_003

- **Study:** study_20260410_131041_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:11:37.351281+00:00
- **Started:** 2026-04-10 13:11:37.351281+00:00
- **Completed:** 2026-04-10 13:12:11.324798+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 40
  - optimizer: adam
  - weight_decay: 0.0005
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.1
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
