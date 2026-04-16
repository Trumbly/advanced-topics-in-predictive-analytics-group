# Test opus with kaggle training

_Task: **track_b** · Primary metric: **f1_macro** · Status: **StudyStatus.ABORTED**_


## Executive summary

**Executive Summary**

No model achieved a successful run on the Track B task, with the best macro F1 score standing at 0.0000 across the single experiment attempted. The sole experiment failed during execution, meaning no architecture or hyperparameter configuration produced valid predictions. The most surprising finding was that even a baseline attempt could not complete successfully, suggesting fundamental issues with data preprocessing, input formatting, or environment configuration rather than model design choices. As a concrete next step, we recommend running a minimal diagnostic experiment—such as fitting a simple logistic regression on a small data subset—to isolate whether the failure stems from data loading, label encoding, or computational resource constraints before scaling to more complex architectures.


## Overview

| | |
|---|---|
| Study ID | `study_20260415_082507_915c` |
| Started | 2026-04-15 08:25:07.684042+00:00 |
| Finished | 2026-04-15 08:28:50.251446+00:00 |
| Experiments | 1 (0 succeeded, 1 failed) |
| Best f1_macro | **—** |
| Predecessor | — |
| Published | no |
| Tags | opus4.6, kaggle, gpu |

## Metric trajectory

![score progression](figures/score_progression.png)

![failure breakdown](figures/failure_breakdown.png)



## Per-architecture-family breakdown



| Family | Runs | Best f1_macro | Mean f1_macro |
|---|---:|---:|---:|


## Failure analysis

| Error type | Count | Example message |
|---|---:|---|
| Aborted | 1 | `User requested abort` |


## Prompt versions used

| Prompt task | File |
|---|---|
| propose_architecture | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/propose_architecture/v1.yaml` |
| generate_code | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/generate_code/v1.yaml` |
| analyze_result | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/analyze_result/v1.yaml` |
| recover_from_error | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/recover_from_error/v1.yaml` |
| executive_summary | `/Users/max/Documents/Master/Courses/Advanced Predictive Analytics/Group Work/advanced-topics-in-predictive-analytics-group/config/prompts/executive_summary/v1.yaml` |


## All experiments

| # | ID | Status | Architecture | f1_macro | Duration (s) |
|---:|---|---|---|---:|---:|
| 1 | `exp_d9bdde11a1` | ExperimentStatus.ABORTED | [efficientnet_b0] efficientnet_b0_mixup_focal | — | 13.2 |
