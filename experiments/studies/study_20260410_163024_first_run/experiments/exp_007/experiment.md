# Experiment exp_007

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 19:00:21.283356+00:00
- **Started:** 2026-04-10 19:00:21.283356+00:00
- **Completed:** 2026-04-10 19:01:18.232906+00:00

## Model Config
- architecture: `ResNet-20 style 3-block CNN with SE attention`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 5
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_007_task_01_propose_architecture
- exp_007_task_02_generate_code
- exp_007_task_03_validate_code
- exp_007_task_04_execute_training
- exp_007_task_05_capture_metrics
