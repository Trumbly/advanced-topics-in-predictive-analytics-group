"""argparse CLI: `python -m lab <subcommand>`.

Subcommands:
    run        -- launch a new study
    report     -- render a study report
    submit     -- build the Kaggle submission notebook
    prompts    -- list / activate / new (prompt versions)
    ui         -- start the FastAPI dashboard
    preprocess -- one-off mel-spectrogram cache build
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    return _DISPATCH[args.cmd](args)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lab", description="Autonomous research agent")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="launch a new study")
    run.add_argument("--task", default="track_b")
    run.add_argument("--predecessor")
    run.add_argument("--use-best-prompts", action="store_true")
    run.add_argument("--agent-memory", action="store_true")
    run.add_argument(
        "--personality", choices=["exploratory", "conservative"], default=None
    )
    run.add_argument("--max-experiments", type=int)
    run.add_argument("--max-wallclock-min", type=int)
    run.add_argument(
        "--llm-model",
        default=None,
        help="override the LLM model (Ollama tag) for this study only.",
    )

    report = sub.add_parser("report", help="render a study report")
    report.add_argument("study_id")

    submit = sub.add_parser("submit", help="build the Kaggle submission notebook")
    submit.add_argument("study_id")

    prompts = sub.add_parser("prompts", help="prompt registry actions")
    prompts.add_argument("action", choices=["list", "activate", "new"])
    prompts.add_argument("task", nargs="?")
    prompts.add_argument("version", nargs="?")
    prompts.add_argument("--system-file")
    prompts.add_argument("--user-file")

    ui = sub.add_parser("ui", help="start the FastAPI dashboard")
    ui.add_argument("--host", default=None)
    ui.add_argument("--port", type=int, default=None)

    preprocess = sub.add_parser("preprocess", help="one-off mel-spectrogram cache")
    preprocess.add_argument("--task", default="track_b")
    preprocess.add_argument(
        "--synthetic",
        action="store_true",
        help="opt-in synthetic shards for smoke testing only",
    )
    preprocess.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help=(
            "cap per-class sample count and write eager train.pt+val.pt "
            "(omit for the default lazy index covering every sample)"
        ),
    )
    preprocess.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite train.pt + val.pt even if present",
    )
    preprocess.add_argument(
        "--train-audio",
        action="store_true",
        help=(
            "build mel-spectrograms for every clip under "
            "data/raw/train_audio/<class>/<sid>.ogg. Use this to recreate "
            "the per-clip cache from scratch when the off-repo build_profile "
            "output is missing."
        ),
    )
    preprocess.add_argument(
        "--soundscapes",
        action="store_true",
        help=(
            "build mel-spectrograms for the labelled soundscape windows "
            "(reads data/raw/train_soundscapes_labels.csv + "
            "data/raw/train_soundscapes/*.ogg, writes to "
            "data/processed/spectrograms/<file>_w<idx>.npy). Required for "
            "training on the 28 soundscape-only target species."
        ),
    )
    preprocess.add_argument(
        "--soundscapes-all-windows",
        action="store_true",
        help=(
            "with --soundscapes: also build mels for unlabelled windows "
            "(every 5s window of every soundscape file, not just the 739 "
            "labelled ones). Useful for SSL/pseudo-labelling experiments."
        ),
    )
    preprocess.add_argument(
        "--unify-labels",
        action="store_true",
        help=(
            "rebuild data/processed/labels.csv from train.csv + "
            "train_soundscapes_labels.csv with the canonical 234-class set "
            "ordering from sample_submission.csv"
        ),
    )

    sub.add_parser("benchmark", help="cross-study (family, arch) leaderboard")

    return p


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    settings = load_settings(args.task)
    settings = _apply_run_overrides(settings, args)
    if args.use_best_prompts:
        settings = _apply_use_best_prompts(settings)

    from lab.core.executor import LocalExecutor
    from lab.core.experiment import RunContext
    from lab.core.judge import Judge
    from lab.core.lifecycle import StudyRunner
    from lab.core.llm import LLMClient
    from lab.core.memory import Memory
    from lab.core.models import Study
    from lab.core.recovery import Recovery
    from lab.core.validator import Validator
    from lab.prompts.engine import PromptEngine
    from lab.prompts.registry import PromptRegistry
    from lab.tasks import get_task_adapter
    from lab.tasks.eda import run_eda

    real_processed = Path(settings.task.processed_data_dir)
    if not (
        (real_processed / "train.pt").exists()
        or (real_processed / "train_index.json").exists()
    ):
        print(
            "ERROR: no processed dataset at "
            f"{real_processed}. Build it with `lab preprocess` (lazy "
            "index, covers all samples) or `lab preprocess "
            "--samples-per-class N` (eager subsample) or "
            "`lab preprocess --synthetic` (smoke test only — score will "
            "be random).",
            file=sys.stderr,
        )
        return 2

    client = LLMClient(settings.llm)
    registry = PromptRegistry(Path(settings.paths.prompts_dir))
    engine = PromptEngine(registry)
    memory = Memory(
        top_k=settings.context.memory_top_k,
        recent_failures=settings.context.recent_failures,
        path=Path(settings.paths.experiments_dir) / "memory.json",
    )
    adapter = get_task_adapter(settings)
    eda = run_eda(adapter, settings)

    ctx = RunContext(
        settings=settings,
        adapter=adapter,
        client=client,
        engine=engine,
        memory=memory,
        validator=Validator(settings),
        executor=LocalExecutor(settings),
        recovery=Recovery(client, engine, settings),
        judge=Judge(client, engine),
        eda_summary=eda.markdown,
    )

    predecessor: Study | None = None
    if args.predecessor:
        predecessor = Study.load(Path(settings.paths.experiments_dir), args.predecessor)

    study = StudyRunner(ctx).run(predecessor=predecessor)
    print(f"study {study.id} -> {study.status}")
    if study.best_score is not None:
        print(f"best: {study.best_experiment_id} ({study.best_score:.4f})")
    return 0


def cmd_report(args) -> int:
    from lab.core.models import Study
    from lab.reporting.generator import generate_report

    settings = load_settings()
    study = Study.load(Path(settings.paths.experiments_dir), args.study_id)
    md = generate_report(study, settings)
    print(f"report written: {md}")
    return 0


def cmd_submit(args) -> int:
    from lab.core.models import Study
    from lab.submission.builder import build_submission_for_study

    settings = load_settings()
    study = Study.load(Path(settings.paths.experiments_dir), args.study_id)
    nb = build_submission_for_study(study, settings)
    print(f"submission written: {nb}")
    return 0


def cmd_prompts(args) -> int:
    from lab.prompts.registry import PromptRegistry

    settings = load_settings()
    registry = PromptRegistry(Path(settings.paths.prompts_dir))

    if args.action == "list":
        for task in (
            "propose_architecture",
            "generate_code",
            "recover_from_error",
            "analyze_result",
            "judge_experiment",
            "judge_study",
        ):
            try:
                active = registry.active_version(task)
                versions = registry.list_versions(task)
                print(f"{task}: active={active} versions={versions}")
            except KeyError:
                print(f"{task}: (no active version)")
        return 0

    if args.action == "activate":
        if not args.task or not args.version:
            print("activate requires <task> <version>", file=sys.stderr)
            return 2
        registry.set_active(args.task, args.version)
        print(f"activated {args.task}={args.version}")
        return 0

    if args.action == "new":
        if not (args.task and args.system_file and args.user_file):
            print("new requires <task> --system-file PATH --user-file PATH", file=sys.stderr)
            return 2
        sys_text = Path(args.system_file).read_text(encoding="utf-8")
        usr_text = Path(args.user_file).read_text(encoding="utf-8")
        v = registry.save_new_version(args.task, sys_text, usr_text)
        print(f"saved {args.task}/{v}")
        return 0

    return 2


def cmd_ui(args) -> int:
    settings = load_settings()
    host = args.host or settings.ui.host
    port = args.port or settings.ui.port
    try:
        import uvicorn  # type: ignore[import-not-found]
        from lab.ui.app import create_app
    except ImportError as exc:
        print(f"UI deps missing: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(create_app(settings), host=host, port=port)
    return 0


def cmd_preprocess(args) -> int:
    settings = load_settings(args.task)

    if args.synthetic:
        from lab.tasks.dataset import write_synthetic_shards

        target = write_synthetic_shards(settings, overwrite=True)
        print(f"synthetic shards written to {target}")
        return 0

    if getattr(args, "train_audio", False):
        return _run_train_audio_mels(
            settings,
            overwrite=getattr(args, "overwrite", False),
        )

    if getattr(args, "soundscapes", False):
        return _run_soundscape_mels(
            settings,
            include_unlabelled=getattr(args, "soundscapes_all_windows", False),
            overwrite=getattr(args, "overwrite", False),
        )

    if getattr(args, "unify_labels", False):
        return _run_unify_labels(settings)

    from lab.tasks.real_preprocess import build_real_shards

    try:
        train_path, val_path = build_real_shards(
            settings,
            samples_per_class=getattr(args, "samples_per_class", None),
            overwrite=getattr(args, "overwrite", False),
        )
    except FileNotFoundError as exc:
        print(f"preprocess failed: {exc}", file=sys.stderr)
        return 2
    print(f"real shards written: {train_path}  +  {val_path}")
    return 0


def _run_train_audio_mels(settings, *, overwrite: bool) -> int:
    """Recursive per-clip mel build over data/raw/train_audio/<class>/<sid>."""
    from lab.tasks.audio_mels import build_train_audio_mels

    processed_dir = Path(settings.task.processed_data_dir)
    spectrograms_dir = processed_dir.parent / "spectrograms"
    raw_dir = processed_dir.parent.parent / "raw"
    if not raw_dir.exists():
        print(f"raw dir missing: {raw_dir}", file=sys.stderr)
        return 2
    try:
        counts = build_train_audio_mels(
            raw_dir, spectrograms_dir, overwrite=overwrite
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"train_audio mel build failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"train_audio mels: built={counts['built']}, "
        f"skipped_existing={counts['skipped_existing']}, "
        f"skipped_short={counts['skipped_short']}, "
        f"files_processed={counts['files_processed']}"
    )
    return 0


def _run_soundscape_mels(settings, *, include_unlabelled: bool, overwrite: bool) -> int:
    """Build the missing soundscape mels (default: only the labelled ones)."""
    from lab.tasks.audio_mels import (
        build_soundscape_mels,
        soundscape_sids_for_labels,
    )

    processed_dir = Path(settings.task.processed_data_dir)
    spectrograms_dir = processed_dir.parent / "spectrograms"
    raw_dir = processed_dir.parent.parent / "raw"
    if not raw_dir.exists():
        print(f"raw dir missing: {raw_dir}", file=sys.stderr)
        return 2

    target_sids = None
    if not include_unlabelled:
        labels_csv = raw_dir / "train_soundscapes_labels.csv"
        if not labels_csv.exists():
            print(
                f"missing {labels_csv}; pass --soundscapes-all-windows to "
                "build mels for every window regardless",
                file=sys.stderr,
            )
            return 2
        target_sids = soundscape_sids_for_labels(labels_csv)
        print(f"targeting {len(target_sids)} labelled soundscape windows")

    try:
        counts = build_soundscape_mels(
            raw_dir,
            spectrograms_dir,
            target_sids=target_sids,
            overwrite=overwrite,
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"soundscape mel build failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"soundscape mels: built={counts['built']}, "
        f"skipped_existing={counts['skipped_existing']}, "
        f"skipped_unrequested={counts['skipped_unrequested']}, "
        f"skipped_short={counts['skipped_short']}, "
        f"files_processed={counts['files_processed']}"
    )
    return 0


def _run_unify_labels(settings) -> int:
    """Rebuild labels.csv with the canonical 234-class union."""
    from lab.tasks.soundscape_preprocess import build_unified_labels

    processed_dir = Path(settings.task.processed_data_dir)
    raw_dir = processed_dir.parent.parent / "raw"
    out_path = processed_dir.parent / "labels.csv"
    try:
        merged = build_unified_labels(raw_dir=raw_dir, out_path=out_path)
    except FileNotFoundError as exc:
        print(f"unify-labels failed: {exc}", file=sys.stderr)
        return 2
    n_classes = len({c for v in merged.values() for c in v})
    print(
        f"unified labels.csv -> {out_path}: {len(merged)} samples, "
        f"{n_classes} distinct classes"
    )
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _apply_run_overrides(settings, args):
    cb = settings.compute_budget
    if args.max_experiments is not None:
        cb = cb.model_copy(update={"max_experiments": args.max_experiments})
    if args.max_wallclock_min is not None:
        cb = cb.model_copy(update={"max_wallclock_minutes": args.max_wallclock_min})

    agent = settings.agent
    if args.personality is not None:
        agent = agent.model_copy(update={"personality": args.personality})
    if args.agent_memory:
        agent = agent.model_copy(update={"memory_enabled": True})

    llm = settings.llm
    chosen_model = getattr(args, "llm_model", None)
    if chosen_model:
        llm = llm.model_copy(update={"model": chosen_model})

    return settings.model_copy(
        update={"compute_budget": cb, "agent": agent, "llm": llm}
    )


def _apply_use_best_prompts(settings):
    from lab.prompts.registry import PromptRegistry
    from lab.prompts.scoring import get_best_version

    registry = PromptRegistry(Path(settings.paths.prompts_dir))
    for task in (
        "propose_architecture",
        "generate_code",
        "recover_from_error",
        "analyze_result",
        "judge_experiment",
        "judge_study",
    ):
        best = get_best_version(
            settings.task_name, Path(settings.paths.experiments_dir), min_runs=3
        )
        if best is not None and best in registry.list_versions(task):
            registry.set_active(task, best)
    return settings


def cmd_benchmark(_args) -> int:
    from lab.core.benchmark import benchmark

    settings = load_settings()
    rows = benchmark(Path(settings.paths.experiments_dir))
    if not rows:
        print("(no scored experiments yet)")
        return 0
    print(f"{'family':<28} {'architecture':<28} {'runs':>5} {'mean':>8} {'best':>8}")
    for r in rows:
        print(
            f"{r.family:<28} {r.architecture_name:<28} {r.runs:>5} "
            f"{r.mean_score:>8.4f} {r.best_score:>8.4f}"
        )
    return 0


_DISPATCH = {
    "run": cmd_run,
    "report": cmd_report,
    "submit": cmd_submit,
    "prompts": cmd_prompts,
    "ui": cmd_ui,
    "preprocess": cmd_preprocess,
    "benchmark": cmd_benchmark,
}


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
