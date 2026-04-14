"""Build the dataset profile for a task.

Thin wrapper: instantiates the task adapter and calls its ``build_profile``.
The adapter is responsible for all task-specific IO.

Usage::

    python scripts/build_profile.py --task track_a
    python scripts/build_profile.py --task track_b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.tasks.registry import get_task_adapter  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", help="task name (defaults to config/config.yaml default_task)")
    args = p.parse_args()

    adapter = get_task_adapter(task_name=args.task)
    profile = adapter.build_profile()
    processed = adapter.settings.abspath(
        adapter.task_cfg.get("data", {}).get("processed_dir", "data/processed")
    )
    processed.mkdir(parents=True, exist_ok=True)
    out = processed / "dataset_profile.json"
    out.write_text(profile.model_dump_json(indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
