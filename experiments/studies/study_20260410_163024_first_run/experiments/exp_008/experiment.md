# Experiment exp_008

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 19:01:18.233939+00:00
- **Started:** 2026-04-10 19:01:18.233939+00:00
- **Completed:** 2026-04-10 19:01:48.990365+00:00

## Model Config
- architecture: `cnn_small_v1 with specAug augmentation`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 5
  - optimizer: adam
  - weight_decay: 0
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0
  - specaugment: True

## Tasks
- exp_008_task_01_propose_architecture
- exp_008_task_02_generate_code
- exp_008_task_03_validate_code
- exp_008_task_04_execute_training
