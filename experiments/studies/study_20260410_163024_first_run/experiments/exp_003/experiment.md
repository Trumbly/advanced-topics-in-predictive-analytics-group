# Experiment exp_003

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 17:13:56.723489+00:00
- **Started:** 2026-04-10 17:13:56.723489+00:00
- **Completed:** 2026-04-10 17:15:34.620468+00:00

## Model Config
- architecture: `ResNet-style 3-block CNN with SE attention per block for 1-channel spectrogram`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 3
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.2
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.05
  - specaugment: False

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
- exp_003_task_04_execute_training
