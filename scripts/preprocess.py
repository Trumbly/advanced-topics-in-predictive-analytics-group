"""Task-dispatched preprocessing (used by Track B).

The adapter owns the real preprocessing. This script is a thin CLI
wrapper — you can subclass :class:`TaskAdapter` to override
``build_profile`` with full preprocessing (spectrogram rendering, etc.)
and invoke here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.tasks.registry import get_task_adapter  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    args = p.parse_args()
    adapter = get_task_adapter(task_name=args.task)
    profile = adapter.build_profile()
    processed = adapter.settings.abspath(
        adapter.task_cfg.get("data", {}).get("processed_dir", "data/processed")
    )
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "dataset_profile.json").write_text(profile.model_dump_json(indent=2))
    print(f"[{args.task}] wrote dataset_profile.json to {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
