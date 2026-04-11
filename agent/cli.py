"""Click-based CLI for the BirdCLEF autonomous research agent.

Commands
--------
    agent start       — start a new Study from a name + pipeline
    agent resume      — resume an existing Study (re-load state from disk)
    agent status      — show overall state of a Study (or the latest)
    agent show-best   — print the best experiment from a Study
    agent list        — list all Studies on disk
    agent submit      — export the best experiment as a Kaggle notebook

Entry point:
    python -m agent.main start --study demo --pipeline default_pipeline.yaml
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

from agent.context_handler import ContextHandler
from agent.executor import CodeExecutor
from agent.llm_client import LLMClient
from agent.logger import ExperimentLogger
from agent.memory import ExperimentMemory
from agent.models import (
    ComputeBudget,
    DatasetProfile,
    GlobalConfig,
    Study,
    StudyMode,
    StudyStatus,
)
from agent.orchestrator import Orchestrator, StopReason
from agent.prompt_engine import PromptEngine
from registry import ModelRegistry


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_global_config(path: Path = Path("config/config.yaml")) -> GlobalConfig:
    data = yaml.safe_load(Path(path).read_text())
    return GlobalConfig.model_validate(data)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _setup_logging(level: str = "INFO") -> None:
    """Root logger config.

    The agent.orchestrator and agent.executor loggers get a minimal
    "%(message)s" format so their progress lines stay readable in the
    terminal. All other loggers keep the full timestamped format for
    debugging. Chatty third-party loggers (httpx, urllib3) are bumped
    up to WARNING to keep the output clean.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet down noisy third-party loggers (they ping every LLM call)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Replace the handler on the orchestrator and executor loggers with a
    # clean formatter. We don't propagate to root so the default handler
    # does not double-log.
    for name in ("agent.orchestrator", "agent.executor"):
        log = logging.getLogger(name)
        log.propagate = False
        log.setLevel(level)
        if log.handlers:
            log.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)


# ---------------------------------------------------------------------------
# Orchestrator construction
# ---------------------------------------------------------------------------


def _resolve_device(configured: str) -> str:
    """Convert a configured device setting ("auto"/"cpu"/"mps"/"cuda") into
    a concrete device string. "auto" prefers MPS on Apple Silicon, then
    CUDA on Linux/Windows, finally CPU. Unknown or unavailable devices
    silently fall back to CPU with a warning logged to the CLI user."""
    configured = (configured or "auto").lower()
    if configured == "cpu":
        return "cpu"
    if configured == "mps":
        if _mps_available():
            return "mps"
        click.echo(
            "⚠ training.device=mps but MPS is not available — falling back to cpu",
            err=True,
        )
        return "cpu"
    if configured == "cuda":
        if _cuda_available():
            return "cuda"
        click.echo(
            "⚠ training.device=cuda but CUDA is not available — falling back to cpu",
            err=True,
        )
        return "cpu"
    # "auto" (or anything else): try each backend in preference order.
    if _mps_available():
        return "mps"
    if _cuda_available():
        return "cuda"
    return "cpu"


def _mps_available() -> bool:
    try:
        import torch

        return bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
    except Exception:  # noqa: BLE001
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _training_env_from_config(gc: GlobalConfig) -> dict[str, str]:
    """Convert the `training:` section of config.yaml into BIRDCLEF_* env
    vars for the sandbox subprocess. Only emit a var when the value is
    actually set — None means "let the data loader pick its default".

    The `device` key is always emitted as a concrete value (cpu/mps/cuda);
    `auto` is resolved here via `_resolve_device`.
    """
    env: dict[str, str] = {
        "BIRDCLEF_DEVICE": _resolve_device(gc.training.device),
        "BIRDCLEF_BATCH_SIZE": str(gc.training.batch_size),
        # Smoke-phase default. The orchestrator bumps this to
        # `compute_budget.promoted_epochs` during the promotion phase.
        "BIRDCLEF_EPOCHS": "1",
    }
    if gc.training.num_workers is not None:
        env["BIRDCLEF_NUM_WORKERS"] = str(gc.training.num_workers)
    env["BIRDCLEF_PERSISTENT_WORKERS"] = (
        "true" if gc.training.persistent_workers else "false"
    )
    env["BIRDCLEF_PREFETCH_FACTOR"] = str(gc.training.prefetch_factor)
    return env


def _build_orchestrator(
    study: Study,
    global_config: GlobalConfig,
) -> Orchestrator:
    """Wire up all dependencies and return a ready-to-run Orchestrator."""
    if not study.dataset_profile_path.exists():
        raise click.ClickException(
            f"Dataset profile not found at {study.dataset_profile_path}. "
            f"Run `python scripts/build_profile.py` first."
        )

    dataset_profile = DatasetProfile.from_json_file(study.dataset_profile_path)
    registry = ModelRegistry(study.model_registry_path)

    llm_client = LLMClient(
        base_url=global_config.llm.base_url,
        model=global_config.llm.default_model,
        temperature=global_config.llm.temperature,
        max_tokens=global_config.llm.max_tokens,
        timeout_seconds=global_config.llm.timeout_seconds,
        retry_attempts=global_config.llm.retry_attempts,
        retry_backoff_seconds=global_config.llm.retry_backoff_seconds,
    )

    prompt_engine = PromptEngine()

    study_dir = global_config.paths.experiments / study.study_id
    memory = ExperimentMemory(study_dir=study_dir)
    experiment_logger = ExperimentLogger(study_dir=study_dir)
    executor = CodeExecutor(
        sandbox_root=global_config.paths.sandbox / study.study_id,
        timeout_seconds=study.compute_budget.max_experiment_seconds,
        training_env=_training_env_from_config(global_config),
    )

    context_handler = ContextHandler(
        prompt_engine=prompt_engine,
        llm_client=llm_client,
        dataset_profile=dataset_profile,
        registry=registry,
        memory=memory,
        max_prompt_tokens=global_config.context.max_prompt_tokens,
        initial_memory_top_k=global_config.context.memory_top_k,
    )

    return Orchestrator(
        study=study,
        llm_client=llm_client,
        prompt_engine=prompt_engine,
        context_handler=context_handler,
        memory=memory,
        experiment_logger=experiment_logger,
        executor=executor,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("config/config.yaml"),
    show_default=True,
    help="Path to the global config YAML",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    show_default=True,
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path, log_level: str) -> None:
    """BirdCLEF autonomous research agent."""
    _setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["global_config"] = load_global_config(config_path)


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--study",
    "study_name",
    required=True,
    help="Human-readable study name (will be slugged into the study_id)",
)
@click.option(
    "--hypothesis",
    default="Explore architectures for BirdCLEF 2026.",
    help="One-line description of what this study tests",
)
@click.option(
    "--pipeline",
    "pipeline_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("config/pipelines/default_pipeline.yaml"),
    show_default=True,
)
@click.option(
    "--model",
    "model_override",
    default=None,
    help="Override the default LLM model (e.g. gemma4:e4b, qwen3:9b)",
)
@click.option(
    "--max-experiments",
    type=int,
    default=None,
    help="Override compute_budget.max_experiments",
)
@click.option(
    "--mode",
    type=click.Choice(["autonomous", "interactive"]),
    default="autonomous",
    show_default=True,
)
@click.pass_context
def start(
    ctx: click.Context,
    study_name: str,
    hypothesis: str,
    pipeline_path: Path,
    model_override: str | None,
    max_experiments: int | None,
    mode: str,
) -> None:
    """Start a new Study and run the agent loop to completion."""
    gc: GlobalConfig = ctx.obj["global_config"]

    if model_override:
        gc.llm.default_model = model_override

    now = _now()
    study_id = f"study_{now.strftime('%Y%m%d_%H%M%S')}_{_slug(study_name)}"

    budget = ComputeBudget(
        max_experiments=max_experiments or gc.compute_budget.max_experiments,
        max_wallclock_minutes=gc.compute_budget.max_wallclock_minutes,
        max_experiment_seconds=gc.compute_budget.max_experiment_seconds,
        max_epochs_per_run=gc.compute_budget.max_epochs_per_run,
    )

    study = Study(
        study_id=study_id,
        name=study_name,
        hypothesis=hypothesis,
        mode=StudyMode(mode),
        compute_budget=budget,
        pipeline_config_path=pipeline_path,
        dataset_profile_path=gc.paths.dataset_profile,
        model_registry_path=gc.paths.model_registry,
        status=StudyStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )

    orchestrator = _build_orchestrator(study, gc)
    click.echo(f"Starting {study_id}")
    click.echo(f"  pipeline: {pipeline_path}")
    click.echo(f"  model:    {gc.llm.default_model}")
    click.echo(f"  budget:   {budget.max_experiments} experiments, "
               f"{budget.max_wallclock_minutes} min wallclock")
    click.echo()

    study, stop_reason = orchestrator.run()

    click.echo()
    click.echo(f"Stopped: {stop_reason}")
    click.echo(f"Experiments run: {len(study.experiment_ids)}")
    if study.best_experiment_id:
        click.echo(
            f"Best: {study.best_experiment_id} "
            f"(score={study.best_score:.4f})"
        )
    else:
        click.echo("No successful experiments.")


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("study_id")
@click.pass_context
def resume(ctx: click.Context, study_id: str) -> None:
    """Resume an existing Study from disk (continue its remaining budget)."""
    gc: GlobalConfig = ctx.obj["global_config"]
    study_path = gc.paths.experiments / study_id / "study.json"
    if not study_path.exists():
        raise click.ClickException(f"Study not found: {study_path}")

    study = Study.from_json_file(study_path)
    study.status = StudyStatus.ACTIVE

    orchestrator = _build_orchestrator(study, gc)
    click.echo(f"Resuming {study_id}")
    click.echo(f"  experiments so far: {len(study.experiment_ids)}")
    click.echo(f"  best score:         {study.best_score}")
    click.echo()

    study, stop_reason = orchestrator.run()
    click.echo(f"Stopped: {stop_reason}")


# ---------------------------------------------------------------------------
# status / list / show-best
# ---------------------------------------------------------------------------


@cli.command("list")
@click.pass_context
def list_studies(ctx: click.Context) -> None:
    """List all studies on disk."""
    gc: GlobalConfig = ctx.obj["global_config"]
    root = gc.paths.experiments
    if not root.exists():
        click.echo("(no studies yet)")
        return

    studies = sorted(p for p in root.iterdir() if p.is_dir())
    if not studies:
        click.echo("(no studies yet)")
        return

    click.echo(f"{'STUDY ID':<50} {'STATUS':<12} {'EXPERIMENTS':>12} {'BEST SCORE':>12}")
    click.echo("-" * 90)
    for study_dir in studies:
        study_json = study_dir / "study.json"
        if not study_json.exists():
            continue
        try:
            study = Study.from_json_file(study_json)
        except Exception:
            continue
        best = f"{study.best_score:.4f}" if study.best_score is not None else "-"
        click.echo(
            f"{study.study_id:<50} {study.status:<12} "
            f"{len(study.experiment_ids):>12} {best:>12}"
        )


@cli.command()
@click.argument("study_id", required=False)
@click.pass_context
def status(ctx: click.Context, study_id: str | None) -> None:
    """Show the status of a Study (or the most recent one)."""
    gc: GlobalConfig = ctx.obj["global_config"]
    study = _resolve_study(gc, study_id)
    click.echo(f"Study:       {study.study_id}")
    click.echo(f"Name:        {study.name}")
    click.echo(f"Status:      {study.status}")
    click.echo(f"Mode:        {study.mode}")
    click.echo(f"Created:     {study.created_at}")
    click.echo(f"Updated:     {study.updated_at}")
    click.echo(f"Experiments: {len(study.experiment_ids)}")
    click.echo(f"Best exp:    {study.best_experiment_id or '(none)'}")
    click.echo(
        f"Best score:  "
        f"{study.best_score if study.best_score is not None else '(none)'}"
    )


@cli.command("show-best")
@click.argument("study_id", required=False)
@click.pass_context
def show_best(ctx: click.Context, study_id: str | None) -> None:
    """Pretty-print the best experiment of a Study."""
    gc: GlobalConfig = ctx.obj["global_config"]
    study = _resolve_study(gc, study_id)
    if not study.best_experiment_id:
        click.echo("No successful experiments yet.")
        return

    exp_json = (
        gc.paths.experiments
        / study.study_id
        / "experiments"
        / study.best_experiment_id
        / "experiment.json"
    )
    if not exp_json.exists():
        click.echo(f"Best experiment file not found: {exp_json}")
        return

    data = json.loads(exp_json.read_text())
    click.echo(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# report — (re)generate an LLM-authored Markdown report for a study
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("study_id", required=False)
@click.pass_context
def report(ctx: click.Context, study_id: str | None) -> None:
    """Generate a Markdown report (with training + failure charts) for a
    completed study. Re-runnable — overwrites the existing report."""
    gc: GlobalConfig = ctx.obj["global_config"]
    study = _resolve_study(gc, study_id)

    if not study.experiment_ids:
        raise click.ClickException(
            f"Study {study.study_id} has no experiments — nothing to report on."
        )

    study_dir = gc.paths.experiments / study.study_id

    # Rebuild the dependencies (LLMClient, PromptEngine, ExperimentMemory)
    # from the global config so the report command works standalone.
    from agent.llm_client import LLMClient
    from agent.memory import ExperimentMemory
    from agent.prompt_engine import PromptEngine
    from agent.report import ReportGenerator

    llm_client = LLMClient(
        base_url=gc.llm.base_url,
        model=gc.llm.default_model,
        temperature=gc.llm.temperature,
        max_tokens=gc.llm.max_tokens,
        timeout_seconds=gc.llm.timeout_seconds,
        retry_attempts=gc.llm.retry_attempts,
        retry_backoff_seconds=gc.llm.retry_backoff_seconds,
    )
    memory = ExperimentMemory(study_dir=study_dir)
    prompt_engine = PromptEngine()

    generator = ReportGenerator(
        study=study,
        memory=memory,
        llm_client=llm_client,
        prompt_engine=prompt_engine,
        study_dir=study_dir,
    )

    click.echo(f"Generating report for {study.study_id}...")
    report_path = generator.generate()
    click.echo(f"  wrote: {report_path}")
    click.echo(f"  figures: {report_path.parent / 'figures'}")


# ---------------------------------------------------------------------------
# submit (placeholder until Phase 5 lands)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("study_id", required=False)
@click.pass_context
def submit(ctx: click.Context, study_id: str | None) -> None:
    """Export the best experiment as a Kaggle submission notebook."""
    gc: GlobalConfig = ctx.obj["global_config"]
    study = _resolve_study(gc, study_id)

    try:
        from agent.submission import SubmissionExporter  # type: ignore[import-not-found]
    except ImportError:
        raise click.ClickException(
            "SubmissionExporter is not yet implemented (Phase 5). "
            "The best experiment is at experiments/studies/"
            f"{study.study_id}/experiments/{study.best_experiment_id}/"
        )

    exporter = SubmissionExporter()
    exp_dir = (
        gc.paths.experiments
        / study.study_id
        / "experiments"
        / (study.best_experiment_id or "")
    )
    click.echo(f"Exporting best experiment from {exp_dir}")
    output_path = gc.paths.experiments / study.study_id / "submissions" / "submission.ipynb"
    exporter.export(study=study, output_path=output_path)
    click.echo(f"Wrote submission notebook to {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_study(gc: GlobalConfig, study_id: str | None) -> Study:
    """Return the Study with the given ID, or the most recent one."""
    if study_id:
        path = gc.paths.experiments / study_id / "study.json"
        if not path.exists():
            raise click.ClickException(f"Study not found: {path}")
        return Study.from_json_file(path)

    root = gc.paths.experiments
    if not root.exists():
        raise click.ClickException("No studies exist yet.")
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "study.json").exists()),
        key=lambda p: (p / "study.json").stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise click.ClickException("No studies exist yet.")
    return Study.from_json_file(candidates[0] / "study.json")


def _slug(text: str) -> str:
    """Slugify a name into something safe for a directory."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    return "".join(out).strip("_") or "study"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
