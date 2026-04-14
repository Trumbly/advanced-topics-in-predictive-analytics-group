"""Command-line interface.

Run ``python -m lab --help`` (or ``lab --help`` after ``pip install -e .``)
to see all subcommands.

Uses ``argparse`` rather than Typer/Click to keep the dependency graph
minimal — the CLI has no third-party CLI-framework dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab import __version__
from lab.config import load_settings
from lab.core import telemetry
from lab.core.models import Study
from lab.reporting.generator import generate_report
from lab.submission.builder import build_submission_for_study
from lab.tasks.registry import get_task_adapter, list_available_tasks
from lab.ui import loaders


def _cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings(task=args.task)
    telemetry.configure(settings)
    log = telemetry.get("cli")

    predecessor = None
    if args.resume:
        exp_dir = settings.abspath(settings.paths.experiments) / args.resume
        if not (exp_dir / "study.json").exists():
            log.error("No predecessor study %s", args.resume)
            return 2
        predecessor = Study.load(exp_dir)

    adapter = get_task_adapter(task_name=settings.task_config.get("name"), settings=settings)
    log.info("Task %s | primary metric %s", adapter.name, adapter.primary_metric)

    # Import here so the CLI loads fast when the user only runs non-orchestrator
    # commands on a Python without torch installed.
    from lab.core.orchestrator import Orchestrator

    orch = Orchestrator(settings=settings, adapter=adapter, predecessor=predecessor)
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    study = orch.run_study(name=args.name or "", tags=tags)

    log.info("Done. Study=%s best=%s", study.id, study.best_score)
    if args.report:
        generate_report(study, settings)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    settings = load_settings(task=args.task)
    experiments_dir = settings.abspath(settings.paths.experiments)
    for s in loaders.iter_studies(experiments_dir):
        best = f"{s.best_score:.4f}" if s.best_score is not None else "—"
        tags = ", ".join(s.tags) if s.tags else ""
        pub = "PUB" if s.publish else "   "
        print(f"{pub}  {s.id}  [{s.status:9s}]  {s.task_name:8s}  "
              f"n={s.experiments_count:2d}  best={best:>8s}  tags={tags}")
    return 0


def _cmd_tasks(_args: argparse.Namespace) -> int:
    settings = load_settings()
    for t in list_available_tasks(settings):
        print(f"{t['name']:12s}  {t['kind']:30s}  {t['description'][:80]}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    settings = load_settings(task=args.task)
    study_dir = settings.abspath(settings.paths.experiments) / args.study_id
    if not (study_dir / "study.json").exists():
        print(f"No study at {study_dir}", file=sys.stderr)
        return 2
    study = Study.load(study_dir)
    path = generate_report(study, settings, write_exec_summary=not args.no_llm)
    print(path)
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    settings = load_settings(task=args.task)
    out = build_submission_for_study(args.study_id, settings=settings)
    print(out)
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        return 2
    host = args.host or settings.ui.host
    port = args.port or settings.ui.port
    uvicorn.run(
        "lab.ui.app:create_app",
        host=host, port=port, factory=True, reload=args.reload,
    )
    return 0


def _cmd_config(_args: argparse.Namespace) -> int:
    settings = load_settings()
    print(json.dumps(settings.model_dump(mode="json"), indent=2, default=str))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a Python file against the validator rules (useful for CI)."""
    from lab.core.validator import validate
    code = Path(args.file).read_text()
    result = validate(code)
    if result.ok:
        print("OK")
        return 0
    print(f"{result.error_type}: {result.message}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab", description=f"lab v{__version__} — LLM research agent")
    p.add_argument("--task", help="task name (default: from config/config.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", help="run an experiment study")
    sp.add_argument("--name", help="study name (auto-generated if omitted)")
    sp.add_argument("--tags", help="comma-separated tags")
    sp.add_argument("--resume", metavar="STUDY_ID",
                    help="continue from predecessor study (seeds memory)")
    sp.add_argument("--report", action="store_true",
                    help="also render the report when the study finishes")
    sp.set_defaults(fn=_cmd_run)

    sp = sub.add_parser("list", help="list studies")
    sp.set_defaults(fn=_cmd_list)

    sp = sub.add_parser("tasks", help="list available tasks")
    sp.set_defaults(fn=_cmd_tasks)

    sp = sub.add_parser("report", help="render a study report")
    sp.add_argument("study_id")
    sp.add_argument("--no-llm", action="store_true",
                    help="skip LLM executive summary (use deterministic fallback)")
    sp.set_defaults(fn=_cmd_report)

    sp = sub.add_parser("submit", help="build submission artifact for the best experiment")
    sp.add_argument("study_id")
    sp.set_defaults(fn=_cmd_submit)

    sp = sub.add_parser("ui", help="serve the web dashboard")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(fn=_cmd_ui)

    sp = sub.add_parser("config", help="print the merged configuration as JSON")
    sp.set_defaults(fn=_cmd_config)

    sp = sub.add_parser("validate", help="validate a Python file against validator rules")
    sp.add_argument("file")
    sp.set_defaults(fn=_cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
