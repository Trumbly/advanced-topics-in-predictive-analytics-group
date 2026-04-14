# Study: full_dataset_v3

- **ID:** study_20260413_225701_full_dataset_v3
- **Status:** completed
- **Mode:** autonomous
- **Hypothesis:** 7 epochs max, early stopping patience=3, batch_size=128, pretrained EfficientNet-B0 first then other pretrained moels for audio and images, cosine LR
- **Created:** 2026-04-13 22:57:01.060077+00:00
- **Updated:** 2026-04-14 07:44:49.989122+00:00

## Compute Budget
- max_experiments: 10
- max_wallclock_minutes: 480
- max_experiment_seconds: 14400
- max_epochs_per_run: 7

## Configuration
- pipeline: `config/pipelines/default_pipeline.yaml`
- dataset profile: `data/processed/dataset_profile.json`
- model registry: `registry/models.yaml`

## Progress
- experiments run: 5
- best experiment: exp_001
- best score: 0.9901357359437392
- submissions: 0
