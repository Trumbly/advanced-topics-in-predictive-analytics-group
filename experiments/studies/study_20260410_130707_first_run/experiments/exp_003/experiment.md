# Experiment exp_003

- **Study:** study_20260410_130707_first_run
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-10 13:08:17.231037+00:00
- **Started:** 2026-04-10 13:08:17.231037+00:00
- **Completed:** 2026-04-10 13:08:58.073424+00:00

## Model Config
- architecture: `cnn_small_v1`
- hyperparams:
  - lr: 0.0005
  - batch_size: 32
  - epochs: 75
  - optimizer: AdamW
  - weight_decay: 0.02
  - dropout: 0.15
- augmentation:
  - time_shift: False
  - noise_injection: False
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
