# Experiment exp_002

- **Study:** study_20260410_163024_first_run
- **Status:** failed
- **LLM:** nemotron-3-nano:4b
- **Created:** 2026-04-10 17:13:19.446626+00:00
- **Started:** 2026-04-10 17:13:19.446626+00:00
- **Completed:** 2026-04-10 17:13:56.722713+00:00

## Model Config
- architecture: `3-conv CNN with SE attention per block for 1-channel spectrogram, 234 output`
- hyperparams:
  - lr: 0.001
  - batch_size: 32
  - epochs: 3
  - optimizer: adam
  - weight_decay: 0.0001
  - dropout: 0.1
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.2
  - specaugment: False

## Tasks
- exp_002_task_01_propose_architecture
- exp_002_task_02_generate_code
- exp_002_task_03_validate_code
- exp_002_task_04_execute_training
- exp_002_task_05_capture_metrics
