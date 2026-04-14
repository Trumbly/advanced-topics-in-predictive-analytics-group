"""Selective study packaging.

Bundles a filtered subset of ``experiments/studies/*`` into either a
tarball (default) or a git commit on a dedicated branch.

Filter dimensions (AND'd together):
  --ids ID1,ID2            Explicit study IDs
  --tags foo,bar           Any-match on tag list
  --task track_a           Only studies for a given task
  --best N                 Top-N by best_score (global; applied after other filters)
  --published-only         Only studies with publish=true in study.json

At least one filter must be specified — naked ``upload_studies`` refuses
to bundle everything.

Usage::

    python scripts/upload_studies.py --published-only
    python scripts/upload_studies.py --tags baseline --best 3
    python scripts/upload_studies.py --ids study_20260101_120000_abcd --format git
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab.config import load_settings  # noqa: E402


def _load_summary(study_dir: Path) -> dict | None:
    p = study_dir / "study.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def select_studies(
    experiments_dir: Path,
    *,
    ids: list[str] | None = None,
    tags: list[str] | None = None,
    task: str | None = None,
    published_only: bool = False,
    best: int | None = None,
) -> list[Path]:
    candidates: list[tuple[Path, dict]] = []
    for d in sorted(experiments_dir.iterdir()):
        data = _load_summary(d)
        if not data:
            continue
        if ids and data.get("id") not in ids:
            continue
        if task and data.get("task_name") != task:
            continue
        if tags:
            s_tags = set(data.get("tags", []) or [])
            if not (s_tags & set(tags)):
                continue
        if published_only and not data.get("publish"):
            continue
        candidates.append((d, data))

    if best is not None:
        candidates.sort(
            key=lambda kv: (kv[1].get("best_score") or float("-inf")),
            reverse=True,
        )
        candidates = candidates[:best]

    return [d for d, _ in candidates]


def bundle_tarball(studies: list[Path], out_dir: Path, prompts_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"studies_bundle_{ts}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        for d in studies:
            tar.add(d, arcname=f"studies/{d.name}")
        if prompts_dir.exists():
            tar.add(prompts_dir, arcname="prompts")
    return out_path


def bundle_git_commit(studies: list[Path], branch: str, experiments_dir: Path) -> str:
    repo_root = experiments_dir.parent.parent  # experiments/studies -> experiments -> repo
    rel_paths = [str(d.relative_to(repo_root)) for d in studies]
    msg = (
        f"upload: {len(studies)} studies "
        f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\n\n"
        + "\n".join(f"- {p}" for p in rel_paths)
    )
    env = os.environ.copy()
    subprocess.run(["git", "checkout", "-B", branch], cwd=repo_root, check=True, env=env)
    subprocess.run(["git", "add", "--"] + rel_paths, cwd=repo_root, check=True, env=env)
    subprocess.run(["git", "commit", "-m", msg], cwd=repo_root, check=True, env=env)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, env=env
    ).decode().strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", help="comma-separated study IDs")
    p.add_argument("--tags", help="comma-separated tag filter (any-match)")
    p.add_argument("--task", help="only studies for this task")
    p.add_argument("--best", type=int, help="top-N by best_score (applied after other filters)")
    p.add_argument("--published-only", action="store_true")
    p.add_argument("--format", choices=["tar", "git"], default="tar")
    p.add_argument("--out-dir", default="reports/bundles",
                   help="output directory for tarballs (format=tar only)")
    p.add_argument("--branch", default="studies-upload",
                   help="branch name for format=git")
    p.add_argument("--dry-run", action="store_true",
                   help="list selected studies but don't bundle")
    args = p.parse_args()

    settings = load_settings()
    experiments_dir = settings.abspath(settings.paths.experiments)
    prompts_dir = settings.abspath(settings.paths.prompts_dir)

    ids = [s.strip() for s in (args.ids or "").split(",") if s.strip()]
    tags = [s.strip() for s in (args.tags or "").split(",") if s.strip()]

    if not (ids or tags or args.task or args.best or args.published_only):
        print("Refusing to bundle everything. Pass at least one filter.", file=sys.stderr)
        return 2

    selected = select_studies(
        experiments_dir,
        ids=ids or None,
        tags=tags or None,
        task=args.task,
        published_only=args.published_only,
        best=args.best,
    )
    if not selected:
        print("No studies matched the filters.", file=sys.stderr)
        return 1

    for d in selected:
        print(f"  selected  {d.name}")

    if args.dry_run:
        print(f"\n(dry-run: would bundle {len(selected)} study directories)")
        return 0

    if args.format == "tar":
        out = bundle_tarball(selected, settings.abspath(args.out_dir), prompts_dir)
        print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    else:
        sha = bundle_git_commit(selected, args.branch, experiments_dir)
        print(f"\ncommitted {sha[:12]} on branch {args.branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
