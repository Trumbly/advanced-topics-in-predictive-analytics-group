"""Unit tests for `agent.context_handler.ContextHandler`."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from agent.context_handler import ContextHandler, EmptyMemory
from agent.llm_client import LLMClient
from agent.models import (
    ClassStats,
    DatasetProfile,
    Experiment,
    ExperimentStatus,
    ModelConfig,
    PromptTemplate,
    TrainingResults,
)
from agent.prompt_engine import PromptEngine


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubMemory:
    """Controllable memory stub for tests."""

    def __init__(
        self,
        markdown: str = "",
        empty: bool = False,
        last: Experiment | None = None,
    ) -> None:
        self._markdown = markdown
        self._empty = empty
        self._last = last
        self.last_top_k: int | None = None

    def to_markdown(self, top_k: int = 5) -> str:
        self.last_top_k = top_k
        # Make the memory shrink as top_k decreases, so the trimming
        # test can verify the token-budget loop actually converges.
        return "\n".join(f"- exp_{i:03d}: ..." for i in range(top_k))

    def is_empty(self) -> bool:
        return self._empty

    def last(self) -> Experiment | None:
        return self._last


class StubRegistry:
    def to_markdown(self) -> str:
        return "### tiny_model\n- params: 0.1M\n"


class StubBackend:
    def create(self, **_: Any) -> Any:
        return {"choices": [{"message": {"content": "ok"}}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_profile() -> DatasetProfile:
    return DatasetProfile(
        num_classes=3,
        num_samples=100,
        spectrogram_shape=(1, 64, 128),
        sample_rate=32_000,
        class_stats=[
            ClassStats(class_id="sp_a", sample_count=50, avg_duration_seconds=5.0),
            ClassStats(class_id="sp_b", sample_count=30, avg_duration_seconds=5.0),
            ClassStats(class_id="sp_c", sample_count=20, avg_duration_seconds=5.0),
        ],
        imbalance_ratio=2.5,
        min_class_samples=20,
        max_class_samples=50,
        split_strategy="fixed_split",
        split_seed=42,
        train_indices=list(range(80)),
        val_indices=list(range(80, 100)),
    )


@pytest.fixture
def llm_client() -> LLMClient:
    return LLMClient(backend=StubBackend(), retry_backoff_seconds=0.0)


@pytest.fixture
def propose_template_path() -> Path:
    return Path("config/prompts/propose_architecture.yaml")


@pytest.fixture
def prompt_engine() -> PromptEngine:
    return PromptEngine()


# ---------------------------------------------------------------------------
# Cold-start (empty memory) behavior
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_empty_memory_triggers_fallback(
        self,
        prompt_engine: PromptEngine,
        llm_client: LLMClient,
        dataset_profile: DatasetProfile,
        propose_template_path: Path,
    ) -> None:
        handler = ContextHandler(
            prompt_engine=prompt_engine,
            llm_client=llm_client,
            dataset_profile=dataset_profile,
            registry=StubRegistry(),
            memory=EmptyMemory(),
        )
        template = prompt_engine.load(propose_template_path)
        system, user = handler.build(template)
        assert "No previous experiments yet" in user


# ---------------------------------------------------------------------------
# Slot filling with real template
# ---------------------------------------------------------------------------


class TestSlotFilling:
    def test_fills_all_slots_for_propose_architecture(
        self,
        prompt_engine: PromptEngine,
        llm_client: LLMClient,
        dataset_profile: DatasetProfile,
        propose_template_path: Path,
    ) -> None:
        memory = StubMemory(markdown="- exp_001: ok", empty=False)
        handler = ContextHandler(
            prompt_engine=prompt_engine,
            llm_client=llm_client,
            dataset_profile=dataset_profile,
            registry=StubRegistry(),
            memory=memory,
        )
        template = prompt_engine.load(propose_template_path)
        system, user = handler.build(template)

        # Dataset profile slot
        assert "num_classes: 3" in user
        assert "imbalance_ratio: 2.50" in user

        # Registry slot
        assert "tiny_model" in user

        # Memory slot — memory.to_markdown returns structured content
        assert "exp_" in user

        # Response schema slot
        assert "architecture" in user

    def test_extra_slots_merge(
        self,
        prompt_engine: PromptEngine,
        llm_client: LLMClient,
        dataset_profile: DatasetProfile,
        tmp_path: Path,
    ) -> None:
        # Make a template that uses extra slots
        import yaml
        data = {
            "name": "custom",
            "description": "test",
            "system": "sys",
            "template": "proposal: {architecture_proposal}\nprofile: {dataset_profile}",
            "slots": ["architecture_proposal", "dataset_profile"],
        }
        path = tmp_path / "custom.yaml"
        path.write_text(yaml.safe_dump(data))

        handler = ContextHandler(
            prompt_engine=prompt_engine,
            llm_client=llm_client,
            dataset_profile=dataset_profile,
            registry=StubRegistry(),
            memory=StubMemory(empty=False),
        )
        template = prompt_engine.load(path)
        _, user = handler.build(
            template,
            extra_slots={"architecture_proposal": "a tiny CNN"},
        )
        assert "a tiny CNN" in user
        assert "num_classes: 3" in user


# ---------------------------------------------------------------------------
# Token budget trimming
# ---------------------------------------------------------------------------


class TestTokenBudgetTrimming:
    def test_trims_memory_when_over_budget(
        self,
        prompt_engine: PromptEngine,
        llm_client: LLMClient,
        dataset_profile: DatasetProfile,
        propose_template_path: Path,
    ) -> None:
        memory = StubMemory(empty=False)
        handler = ContextHandler(
            prompt_engine=prompt_engine,
            llm_client=llm_client,
            dataset_profile=dataset_profile,
            registry=StubRegistry(),
            memory=memory,
            max_prompt_tokens=300,      # very tight budget
            initial_memory_top_k=10,
            min_memory_entries=1,
        )
        template = prompt_engine.load(propose_template_path)
        handler.build(template)
        # top_k should have been reduced to fit the budget
        assert memory.last_top_k is not None
        assert memory.last_top_k < 10

    def test_budget_loop_stops_at_min_entries(
        self,
        prompt_engine: PromptEngine,
        llm_client: LLMClient,
        dataset_profile: DatasetProfile,
        propose_template_path: Path,
    ) -> None:
        memory = StubMemory(empty=False)
        handler = ContextHandler(
            prompt_engine=prompt_engine,
            llm_client=llm_client,
            dataset_profile=dataset_profile,
            registry=StubRegistry(),
            memory=memory,
            max_prompt_tokens=1,        # impossible budget → trim to minimum
            initial_memory_top_k=5,
            min_memory_entries=2,
        )
        template = prompt_engine.load(propose_template_path)
        handler.build(template)
        # Must not go below min_memory_entries even if budget forbids
        assert memory.last_top_k == 2


# ---------------------------------------------------------------------------
# last_result slot serialization
# ---------------------------------------------------------------------------


class TestLastResultSlot:
    def test_last_experiment_rendered_in_markdown(
        self,
        prompt_engine: PromptEngine,
        llm_client: LLMClient,
        dataset_profile: DatasetProfile,
        propose_template_path: Path,
    ) -> None:
        last_exp = Experiment(
            experiment_id="exp_007",
            study_id="study_x",
            llm_model="gemma4:e4b",
            status=ExperimentStatus.COMPLETED,
            config=ModelConfig(
                architecture="cnn_small_v1",
                hyperparams={"lr": 1e-3, "epochs": 5},
            ),
            results=TrainingResults(
                metrics={"roc_auc_macro": 0.68, "loss": 0.42},
                duration_seconds=145.3,
            ),
            created_at=datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc),
        )
        memory = StubMemory(empty=False, last=last_exp)
        handler = ContextHandler(
            prompt_engine=prompt_engine,
            llm_client=llm_client,
            dataset_profile=dataset_profile,
            registry=StubRegistry(),
            memory=memory,
            max_prompt_tokens=100_000,
        )
        template = prompt_engine.load(propose_template_path)
        _, user = handler.build(template)
        assert "exp_007" in user
        assert "cnn_small_v1" in user
        assert "roc_auc_macro=0.6800" in user
