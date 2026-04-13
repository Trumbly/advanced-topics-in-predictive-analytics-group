# Experiment exp_002

- **Study:** study_20260410_131041_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 13:11:10.162124+00:00
- **Started:** 2026-04-10 13:11:10.162124+00:00
- **Completed:** 2026-04-10 13:11:37.350371+00:00

## Model Config
- architecture: `resnet18`
- pretrained: `resnet18`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 40
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: True

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
