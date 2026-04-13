# Experiment exp_003

- **Study:** study_20260410_135001_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:51:13.270446+00:00
- **Started:** 2026-04-10 13:51:13.270446+00:00
- **Completed:** 2026-04-10 13:52:06.100226+00:00

## Model Config
- architecture: `custom 3-conv baseline with 1x1 reduction + 3x3 stride, global avg pool, linear head`
- hyperparams:
  - lr: 0.0005
  - batch_size: 128
  - epochs: 30
  - optimizer: Adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
- exp_003_task_05_capture_metrics
