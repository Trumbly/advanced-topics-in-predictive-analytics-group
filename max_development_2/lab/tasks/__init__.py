"""Task adapters.

Each adapter bridges the task-agnostic agent loop to a specific track:
how data is loaded, what metric is primary, how to build a submission.
See ``base.TaskAdapter`` for the contract and ``registry.get_task_adapter``
for the lookup.
"""
