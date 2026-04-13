# Experiment exp_005

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 17:16:40.947539+00:00
- **Started:** 2026-04-10 17:16:40.947539+00:00
- **Completed:** 2026-04-10 17:17:57.681866+00:00

## Model Config
- architecture: `custom 3‑conv residual CNN with SE attention and attention dropout`
- hyperparams:
  - lr: 0.0005
  - batch_size: 32
  - epochs: 3
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
- exp_005_task_03_validate_code
- exp_005_task_04_execute_training
- exp_005_task_05_capture_metrics
