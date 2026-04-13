# Experiment exp_002

- **Study:** study_20260410_141830_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 14:19:00.408561+00:00
- **Started:** 2026-04-10 14:19:00.408561+00:00
- **Completed:** 2026-04-10 14:20:45.195538+00:00

## Model Config
- architecture: `3-conv CNN with channel-wise SE attention and global pooling`
- hyperparams:
  - lr: 0.0005
  - batch_size: 64
  - epochs: 10
  - optimizer: adam
  - weight_decay: 0.001
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: True

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
