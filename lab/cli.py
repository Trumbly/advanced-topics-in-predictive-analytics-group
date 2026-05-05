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
    preprocess.add_argument("--synthetic", action="store_true")

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
    out = Path(settings.task.processed_data_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        import torch

        n_train, n_val = 200, 50
        for split, n in (("train", n_train), ("val", n_val)):
            x = torch.randn(n, *settings.task.input_tensor_shape)
            y = torch.zeros(n, settings.task.expected_num_classes)
            idx = torch.randint(0, settings.task.expected_num_classes, (n,))
            y[torch.arange(n), idx] = 1.0
            torch.save({"x": x, "y": y}, out / f"{split}.pt")
        print(f"synthetic shards written to {out}")
        return 0

    print(
        "non-synthetic preprocessing not implemented in CLI; use scripts/preprocess.py "
        "for the real BirdCLEF mel-spectrogram pipeline.",
        file=sys.stderr,
    )
    return 1


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

    return settings.model_copy(update={"compute_budget": cb, "agent": agent})


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
