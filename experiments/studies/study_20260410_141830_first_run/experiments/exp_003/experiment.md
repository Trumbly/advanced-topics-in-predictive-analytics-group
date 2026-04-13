# Experiment exp_003

- **Study:** study_20260410_141830_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 14:20:45.197133+00:00
- **Started:** 2026-04-10 14:20:45.197133+00:00
- **Completed:** 2026-04-10 14:21:41.362539+00:00

## Model Config
- architecture: `3-conv CNN with SE attention and global pooling`
- hyperparams:
  - lr: 0.0005
  - batch_size: 64
  - epochs: 30
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.4
  - specaugment: True

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
