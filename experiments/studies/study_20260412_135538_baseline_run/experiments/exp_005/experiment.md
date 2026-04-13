# Experiment exp_005

- **Study:** study_20260412_135538_baseline_run
- **Status:** failed
- **LLM:** gemma4:e4b
- **Created:** 2026-04-12 14:20:56.440284+00:00
- **Started:** 2026-04-12 14:20:56.440284+00:00
- **Completed:** 2026-04-12 14:39:45.811882+00:00

## Model Config
- architecture: `[cnn_attention] Deep residual CNN front-end (5 blocks) feeding into a Multi-Head Self-Attention layer, followed by Adaptive Pooling`
- hyperparams:
  - lr: 0.001
  - batch_size: 128
  - epochs: 1
  - optimizer: adam
  - weight_decay: 0.0
  - dropout: 0.2
- augmentation:
  - time_shift: False
  - noise_injection: True
  - mixup: 0.0
  - specaugment: True

## Tasks
- exp_005_task_01_propose_architecture
- exp_005_task_02_generate_code
