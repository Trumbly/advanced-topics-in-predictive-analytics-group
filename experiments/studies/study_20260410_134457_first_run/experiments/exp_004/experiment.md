# Experiment exp_004

- **Study:** study_20260410_134457_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:46:51.444127+00:00
- **Started:** 2026-04-10 13:46:51.444127+00:00
- **Completed:** 2026-04-10 13:47:44.369501+00:00

## Model Config
- architecture: `lightweight 2-conv CNN with per-channel SE attention and global average pooling`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 40
  - optimizer: Adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
- exp_004_task_05_capture_metrics
