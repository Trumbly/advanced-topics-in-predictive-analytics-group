"""Prompt engineering module.

Three pieces, split for testability:

  * ``engine`` — template loading and slot filling.
  * ``registry`` — versioned, immutable prompt files with a mutable
    ``_registry.yaml`` that tracks which version is active.
  * ``scoring`` — aggregates experiment scores by (task, version) so
    we can A/B test prompt wording against real runs.
"""
