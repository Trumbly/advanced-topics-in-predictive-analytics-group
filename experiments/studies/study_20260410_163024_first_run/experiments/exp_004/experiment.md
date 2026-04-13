# Experiment exp_004

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 17:15:34.621355+00:00
- **Started:** 2026-04-10 17:15:34.621355+00:00
- **Completed:** 2026-04-10 17:16:40.946869+00:00

## Model Config
- architecture: `custom 3-conv baseline with spatial dropout and 1‑channel feature extraction`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 3
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_004_task_01_propose_architecture
- exp_004_task_02_generate_code
- exp_004_task_03_validate_code
- exp_004_task_04_execute_training
- exp_004_task_05_capture_metrics
