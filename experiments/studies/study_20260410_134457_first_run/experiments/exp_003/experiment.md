# Experiment exp_003

- **Study:** study_20260410_134457_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:46:06.190193+00:00
- **Started:** 2026-04-10 13:46:06.190193+00:00
- **Completed:** 2026-04-10 13:46:51.443486+00:00

## Model Config
- architecture: `4-block lightweight CNN with SE attention and global average pooling`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 50
  - optimizer: Adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
