# Task exp_010_task_12_capture_metrics

- **Experiment:** exp_010
- **Type:** predefined
- **Name:** capture_metrics
- **Status:** failed
- **Started:** 2026-04-13 00:26:57.364147+00:00
- **Completed:** 2026-04-13 00:26:57.371741+00:00

## Output
- **results_json_path:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_222225_baseline_run_v2/exp_010/results.json
- **raw_results:** {'model_architecture': 'cnn_attention_model', 'hyperparameters': {'lr': 0.001, 'epochs': 1, 'batch_size': '512', 'device': 'mps'}, 'metrics': {'roc_auc_macro': 0.3377540678886641, 'ap_score_macro': 0.09261339635437048}}

## Error
- **type:** MetricsParseError
- **message:** /Users/dqureshi/advanced-topics-in-predictive-analytics-group/sandbox/study_20260412_222225_baseline_run_v2/exp_010/results.json: missing required metrics ['loss']. Got: ['ap_score_macro', 'roc_auc_macro']
