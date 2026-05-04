"""I-06 acceptance: end-to-end study with stubbed LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lab.config import load_settings
from lab.core import telemetry
from lab.core.executor import LocalExecutor
from lab.core.experiment import RunContext
from lab.core.judge import Judge
from lab.core.lifecycle import StudyRunner
from lab.core.llm import LLMClient
from lab.core.memory import Memory
from lab.core.recovery import Recovery
from lab.core.validator import Validator
from lab.prompts.engine import PromptEngine
from lab.prompts.registry import PromptRegistry
from lab.tasks import get_task_adapter

REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 4
INPUT_SHAPE = (1, 8, 8)


_PROPOSAL_JSON = json.dumps(
    {
        "architecture_name": "TinyLinear",
        "family": "cnn_scratch",
        "lr": 1e-3,
        "lr_schedule": "constant",
        "epochs": 1,
    }
)
_BUILD_BLOCK = """\
# --- AGENT_BUILD_MODEL_START ---
def build_model(num_classes: int) -> nn.Module:
    in_chan = INPUT_SHAPE[0]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_chan * INPUT_SHAPE[1] * INPUT_SHAPE[2], num_classes),
    )
# --- AGENT_BUILD_MODEL_END ---
"""
_VERDICT_KEEP = json.dumps(
    {"verdict": "keep", "score": 0.6, "rationale": "ok"}
)
_VERDICT_ABORT = json.dumps(
    {"verdict": "abort_study", "score": 0.5, "rationale": "stop"}
)


class _ScriptedPoster:
    """Cycles through a queue of scripted LLM replies, one per HTTP call."""

    def __init__(self, payloads: list[str]):
        self.calls = []
        self.payloads = list(payloads)

    def post(self, url, body, headers):
        self.calls.append(body)
        if not self.payloads:
            raise RuntimeError("scripted LLM exhausted")
        text = self.payloads.pop(0)
        return 200, json.dumps(
            {"choices": [{"message": {"content": text}}]}
        ).encode("utf-8")


def _make_synthetic_shard(processed_dir: Path, n_train: int = 60, n_val: int = 20):
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    def make(n):
        x = torch.randn(n, *INPUT_SHAPE)
        y = torch.zeros(n, NUM_CLASSES)
        idx = torch.randint(0, NUM_CLASSES, (n,))
        y[torch.arange(n), idx] = 1.0
        return {"x": x, "y": y}

    torch.save(make(n_train), processed_dir / "train.pt")
    torch.save(make(n_val), processed_dir / "val.pt")


@pytest.fixture
def isolated_settings(tmp_path):
    s = load_settings("track_b", repo_root=REPO_ROOT)
    new_paths = s.paths.model_copy(
        update={
            "experiments_dir": str(tmp_path / "studies"),
            "sandbox": str(tmp_path / "sandbox"),
        }
    )
    new_task = s.task.model_copy(
        update={
            "expected_num_classes": NUM_CLASSES,
            "input_tensor_shape": list(INPUT_SHAPE),
            "processed_data_dir": str(tmp_path / "mels"),
        }
    )
    new_budget = s.compute_budget.model_copy(
        update={
            "max_experiments": 2,
            "max_wallclock_minutes": 5,
            "max_experiment_seconds": 60,
            "max_epochs_per_run": 1,
            "max_codegen_retries": 1,
            "max_recovery_attempts": 1,
        }
    )
    s = s.model_copy(update={"paths": new_paths, "task": new_task, "compute_budget": new_budget})
    _make_synthetic_shard(Path(new_task.processed_data_dir))
    telemetry._reset_for_tests()
    yield s
    telemetry._reset_for_tests()


def _build_ctx(settings, scripted: list[str]) -> tuple[RunContext, _ScriptedPoster]:
    poster = _ScriptedPoster(scripted)
    cfg = settings.llm.model_copy(update={"retry_attempts": 0, "retry_backoff_seconds": 0.0})
    client = LLMClient(cfg, http=poster, sleep=lambda _: None)
    registry = PromptRegistry(REPO_ROOT / "config" / "prompts")
    engine = PromptEngine(registry)
    memory = Memory(top_k=3, recent_failures=3, path=Path(settings.paths.experiments_dir) / "memory.json")
    validator = Validator(settings)
    executor = LocalExecutor(settings)
    recovery = Recovery(client, engine, settings)
    judge = Judge(client, engine)
    adapter = get_task_adapter(settings)
    return (
        RunContext(
            settings=settings,
            adapter=adapter,
            client=client,
            engine=engine,
            memory=memory,
            validator=validator,
            executor=executor,
            recovery=recovery,
            judge=judge,
            eda_summary="(no EDA)",
        ),
        poster,
    )


def test_study_runs_two_experiments_end_to_end(isolated_settings):
    """One propose call + one generate call + judge per experiment, twice, plus a study judge."""
    scripted = []
    for _ in range(2):
        scripted += [_PROPOSAL_JSON, _BUILD_BLOCK, _VERDICT_KEEP]
    scripted.append(_VERDICT_KEEP)  # study-level verdict
    ctx, poster = _build_ctx(isolated_settings, scripted)

    runner = StudyRunner(ctx)
    study = runner.run()

    assert study.status == "COMPLETED"
    assert len(study.experiments) == 2
    for exp in study.experiments:
        assert exp.status == "JUDGED"
        assert exp.primary_score is not None

    # Study JSON exists on disk
    saved = Path(isolated_settings.paths.experiments_dir) / study.id / "study.json"
    assert saved.exists()


def test_abort_study_verdict_halts_loop(isolated_settings):
    """A judge verdict of abort_study after exp 1 stops the loop early."""
    scripted = [
        _PROPOSAL_JSON,
        _BUILD_BLOCK,
        _VERDICT_ABORT,  # judges experiment 1 with abort_study
        _VERDICT_KEEP,  # study-level verdict
    ]
    ctx, _ = _build_ctx(isolated_settings, scripted)

    # Bump experiments cap so abort_study is the actual stopper.
    cb = isolated_settings.compute_budget.model_copy(update={"max_experiments": 5})
    ctx.settings = isolated_settings.model_copy(update={"compute_budget": cb})

    study = StudyRunner(ctx).run()
    assert len(study.experiments) == 1
    assert study.experiments[0].verdict.verdict == "abort_study"
