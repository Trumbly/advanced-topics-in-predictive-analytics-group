"""Predefined task handlers for the agent pipeline.

Each handler module exposes a single `run(task, experiment, context)`
function. The orchestrator dispatches to them by the dotted path in a
pipeline YAML:

    steps:
      - task_name: execute_training
        task_type: predefined
        handler: agent.handlers.execute_training

The handler mutates the given `Task` (status, output, error) and
returns it.
"""

from agent.handlers import (
    capture_metrics,
    execute_training,
    validate_code,
)

__all__ = ["capture_metrics", "execute_training", "validate_code"]
