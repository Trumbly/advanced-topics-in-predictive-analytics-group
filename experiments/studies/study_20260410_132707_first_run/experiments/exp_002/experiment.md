# Experiment exp_002

- **Study:** study_20260410_132707_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:27:50.965211+00:00
- **Started:** 2026-04-10 13:27:50.965211+00:00
- **Completed:** 2026-04-10 13:28:32.102186+00:00

## Model Config
- architecture: `efficientnet_b0`
- pretrained: `efficientnet_b0`
- hyperparams:
  - lr: 0.0001
  - batch_size: 32
  - epochs: 30
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
