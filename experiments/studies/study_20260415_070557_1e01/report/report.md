# Test opus with local training now with gpu

_Task: **track_b** · Primary metric: **f1_macro** · Status: **StudyStatus.ABORTED**_


## Executive summary

**Executive Summary**

No model achieved a successful result on the Track B task, with the best macro F1 score remaining at 0.0000 across the single experiment attempted, which failed during execution. The most surprising finding was that even a single baseline experiment could not be brought to completion, suggesting fundamental issues with either the data pipeline, environment configuration, or task setup rather than mere model underperformance. As a concrete next step, we recommend running a minimal diagnostic experiment—such as fitting a simple logistic regression on a small data subset with verbose logging—to isolate whether the failure stems from data loading, preprocessing, or model training, before attempting more complex architectures.


## Overview

| | |
|---|---|
| Study ID | `study_20260415_070557_1e01` |
| Started | 2026-04-15 07:05:57.558398+00:00 |
| Finished | 2026-04-15 07:25:00.083512+00:00 |
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
| 1 | `exp_384954ad83` | ExperimentStatus.ABORTED | [efficientnet_b0] efficientnet_b0_mixup_focal | — | 1114.2 |
