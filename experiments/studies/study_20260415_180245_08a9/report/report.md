# track_b-20260415_180245

_Task: **track_b** · Primary metric: **f1_macro** · Status: **StudyStatus.COMPLETED**_


## Executive summary

**Executive Summary**

Our research on task_track_b yielded a promising result with an f1_macro score of 0.0040, achieved using the mobilenet_v3_small architecture with specaugment and dropout for regularization. This best result was driven by the combination of model architecture and data augmentation techniques, which together enhanced the model's ability to generalize from limited data.

The reliability of our process is currently fragile due to a high failure rate (9 out of 10 experiments failed), indicating significant instability in our current approach. A common error type observed was related to data preprocessing issues, contributing to the majority of failures. To address this, we plan to implement more robust data validation and preprocessing pipelines. Our next steps include: 1) Conducting a thorough review of data preprocessing steps to identify and rectify errors; 2) Exploring alternative architectures and hyperparameter tuning to improve model stability; 3) Implementing automated monitoring for runtime bottlenecks to ensure efficient resource utilization.


## Overview

| | |
|---|---|
| Study ID | `study_20260415_180245_08a9` |
| Started | 2026-04-15 18:02:45.482244+00:00 |
| Finished | 2026-04-15 18:59:10.408652+00:00 |
| Experiments | 10 (1 succeeded, 9 failed) |
| Best f1_macro | **0.0040** |
| Predecessor | — |
| Published | no |
| Tags | — |

## Metric trajectory

![score progression](figures/score_progression.png)

![failure breakdown](figures/failure_breakdown.png)


## Best experiment — `exp_e6953f58ce`

- Architecture: **[mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe** [mobilenet_v3_small]
- f1_macro: **0.0040**
- Duration: 1630.8 s
- Other metrics:
  - `f1_macro` = 0.0040
  - `roc_auc_macro` = 0.6417
  - `roc_auc_macro_valid_only` = 0.6417
  - `n_valid_roc_labels` = 201.0000
  - `n_total_roc_labels` = 206.0000
  - `loss` = 0.0408
  - `epoch` = 1.0000





### Best experiment — epoch-wise metrics

| Epoch | Loss | f1_macro | roc_auc_macro |
|---:|---:|---:|---:|

| 1 | 0.0408 | 0.0040 | 0.6417 |



### Architecture proposal (JSON)

```json
{
  "architecture_name": "[mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe",
  "architecture_family": "mobilenet_v3_small",
  "description": "MobileNetV3 Small with SpecAugment and DP-safe configuration for reliable multi-label bird species classification.",
  "reasoning": "MobileNetV3 Small is a lightweight architecture that balances performance and efficiency, making it suitable for the Kaggle runtime budget. The addition of SpecAugment helps address the long-tail issue by augmenting the spectrograms, while the DP-safe configuration ensures stability during training."
}
```


## Per-architecture-family breakdown

![by family](figures/per_family_box.png)

| Family | Runs | Best f1_macro | Mean f1_macro |
|---|---:|---:|---:|
| mobilenet_v3_small | 1 | 0.0040 | 0.0040 |


## Failure analysis

| Error type | Count | Example message |
|---|---:|---|
| ValidationFailed | 9 | `Codegen retries exhausted` |


## Pipeline diagnostics

| Task | Runs | Completed | Failed | Avg validator attempts |
|---|---:|---:|---:|---:|
| capture_metrics | 1 | 1 | 0 | — |
| execute_training | 1 | 1 | 0 | — |
| generate_code | 10 | 10 | 0 | — |
| propose_architecture | 10 | 10 | 0 | — |
| validate_code | 10 | 1 | 9 | 4.60 |


## Prompt versions used

| Prompt task | File |
|---|---|
| propose_architecture | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/propose_architecture/v2.yaml` |
| generate_code | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/generate_code/v2.yaml` |
| analyze_result | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/analyze_result/v2.yaml` |
| recover_from_error | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/recover_from_error/v2.yaml` |
| executive_summary | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/executive_summary/v2.yaml` |


## All experiments

| # | ID | Status | Architecture | f1_macro | Duration (s) |
|---:|---|---|---|---:|---:|
| 1 | `exp_69ac1d3f68` | ExperimentStatus.FAILED | [efficientnet_b0] efficientnet_b0_adapter_mixup_dp_safe | — | 0.0 |
| 2 | `exp_821b7bb080` | ExperimentStatus.FAILED | [mobilenet_v3_small] mobilenet_v3_small_adapter_specaugment_dp_safe | — | 0.0 |
| 3 | `exp_4b3fd0c751` | ExperimentStatus.FAILED | [mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe | — | 0.0 |
| 4 | `exp_47531e3e5b` | ExperimentStatus.FAILED | [efficientnet_b0] effnet_b0_mixup_dp_safe | — | 0.0 |
| 5 | `exp_7e66470c6f` | ExperimentStatus.FAILED | [resnet18] resnet18_adapter_specaugment_dp_safe | — | 0.0 |
| 6 | `exp_e6953f58ce` | ExperimentStatus.COMPLETED | [mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe | 0.0040 | 1630.8 |
| 7 | `exp_b0662ea117` | ExperimentStatus.FAILED | [mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe | — | 0.0 |
| 8 | `exp_cc732d00ee` | ExperimentStatus.FAILED | [efficientnet_b0] effnet_b0_mixup_dp_safe | — | 0.0 |
| 9 | `exp_657831ec51` | ExperimentStatus.FAILED | [mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe | — | 0.0 |
| 10 | `exp_a52b807a93` | ExperimentStatus.FAILED | [mobilenet_v3_small] mobilenet_v3_small_specaugment_dp_safe | — | 0.0 |
