# Experiment exp_002

- **Study:** study_20260410_133357_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:34:20.308136+00:00
- **Started:** 2026-04-10 13:34:20.308136+00:00
- **Completed:** 2026-04-10 13:35:09.874046+00:00

## Model Config
- architecture: `3-conv2d backbone with global average pooling, dropout, and linear head`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 30
  - optimizer: adamw
  - weight_decay: 0.0005
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
