# Experiment exp_002

- **Study:** study_20260410_130707_first_run
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-10 13:07:34.182804+00:00
- **Started:** 2026-04-10 13:07:34.182804+00:00
- **Completed:** 2026-04-10 13:08:17.230021+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 50
  - optimizer: AdamW
  - weight_decay: 0.01
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
