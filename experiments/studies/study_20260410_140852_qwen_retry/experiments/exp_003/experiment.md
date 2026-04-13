# Experiment exp_003

- **Study:** study_20260410_140852_qwen_retry
- **Status:** failed
- **LLM:** qwen3-coder
- **Created:** 2026-04-10 14:09:46.712834+00:00
- **Started:** 2026-04-10 14:09:46.712834+00:00
- **Completed:** 2026-04-10 14:10:17.343121+00:00

## Model Config
- architecture: `Custom 3-block CNN with Squeeze-and-Excitation (SE) attention + adaptive pooling`
- hyperparams:
  - lr: 0.001
  - batch_size: 64
  - epochs: 25
  - optimizer: adam
  - weight_decay: 1e-05
  - dropout: 0.3
- augmentation:
  - time_shift: True
  - noise_injection: True
  - mixup: 0.2
  - specaugment: True

## Tasks
- exp_003_task_01_propose_architecture
- exp_003_task_02_generate_code
- exp_003_task_03_validate_code
