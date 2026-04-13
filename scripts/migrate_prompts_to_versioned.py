#!/usr/bin/env python3
"""One-time migration: move flat prompt YAML files into versioned directories.

Before:  config/prompts/generate_code.yaml
After:   config/prompts/generate_code/v1.yaml + config/prompts/_registry.yaml

Safe to run multiple times — already-migrated tasks are skipped.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Add repo root to path so we can import agent.*
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agent.prompt_registry import PromptRegistryManager


def main() -> None:
    prompts_dir = repo_root / "config" / "prompts"
    print(f"Migrating prompts in {prompts_dir}")
    registry = PromptRegistryManager.migrate_flat_to_versioned(prompts_dir)
    print(f"Done. {len(registry.tasks)} tasks registered:")
    for task_name, entry in sorted(registry.tasks.items()):
        print(f"  {task_name}: default={entry.default_version}, versions={sorted(entry.versions.keys())}")


if __name__ == "__main__":
    main()
