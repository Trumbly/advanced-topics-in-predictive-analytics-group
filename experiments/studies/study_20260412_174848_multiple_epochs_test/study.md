# Study: Multiple epochs test

- **ID:** study_20260412_174848_multiple_epochs_test
- **Status:** completed
- **Mode:** autonomous
- **Hypothesis:** Check performance of multiple epochs, again with a larger qwen model
- **Created:** 2026-04-12 15:48:48.425122+00:00
- **Updated:** 2026-04-12 21:37:43.320758+00:00

## Compute Budget
- max_experiments: 10
- max_wallclock_minutes: 240
- max_experiment_seconds: 7200
- max_epochs_per_run: 5

## Configuration
- pipeline: `config/pipelines/default_pipeline.yaml`
- dataset profile: `data/processed/dataset_profile.json`
- model registry: `registry/models.yaml`

## Progress
- experiments run: 5
- best experiment: exp_005
- best score: 0.9806411851870984
- submissions: 0
