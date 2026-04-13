# Experiment exp_009

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 19:01:48.991430+00:00
- **Started:** 2026-04-10 19:01:48.991430+00:00
- **Completed:** 2026-04-10 19:03:00.970258+00:00

## Model Config
- architecture: `Custom 3-conv baseline with global average pooling and dropout`
- hyperparams:
  - lr: 0.0005
  - batch_size: 32
  - epochs: 8
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: False

## Tasks
- exp_009_task_01_propose_architecture
- exp_009_task_02_generate_code
- exp_009_task_03_validate_code
- exp_009_task_04_execute_training
- exp_009_task_05_capture_metrics
