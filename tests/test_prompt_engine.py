"""Unit tests for `agent.prompt_engine.PromptEngine`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.models import PromptTemplate
from agent.prompt_engine import MissingSlotError, PromptEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_template_yaml(tmp_path: Path) -> Path:
    data = {
        "name": "test_template",
        "description": "Test",
        "system": "You are a test bot.",
        "template": "Hello {name}, your score is {score}.",
        "slots": ["name", "score"],
        "fallback_when_no_memory": "No memory yet.",
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def real_template_paths() -> list[Path]:
    """All prompt templates checked into the repo (versioned layout)."""
    # After migration: config/prompts/<task>/v1.yaml, not config/prompts/<task>.yaml
    return sorted(Path("config/prompts").glob("*/v*.yaml"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestPromptEngineLoad:
    def test_load_returns_prompt_template(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        template = engine.load(minimal_template_yaml)
        assert isinstance(template, PromptTemplate)
        assert template.name == "test_template"
        assert template.slots == ["name", "score"]

    def test_load_caches(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        first = engine.load(minimal_template_yaml)
        second = engine.load(minimal_template_yaml)
        assert first is second

    def test_clear_cache(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        first = engine.load(minimal_template_yaml)
        engine.clear_cache()
        second = engine.load(minimal_template_yaml)
        assert first is not second

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        engine = PromptEngine()
        with pytest.raises(FileNotFoundError):
            engine.load(tmp_path / "nope.yaml")

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just a list\n- not a mapping\n")
        engine = PromptEngine()
        with pytest.raises(ValueError, match="mapping"):
            engine.load(path)

    def test_real_templates_all_load(self, real_template_paths: list[Path]) -> None:
        """Smoke test against every checked-in template."""
        assert len(real_template_paths) >= 4, "expected at least 4 prompt templates"
        engine = PromptEngine()
        for path in real_template_paths:
            engine.load(path)  # must not raise


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------


class TestPromptEngineFill:
    def test_fill_basic(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        system, user = engine.fill(
            minimal_template_yaml, {"name": "Claude", "score": "99"}
        )
        assert system == "You are a test bot."
        assert user == "Hello Claude, your score is 99."

    def test_fill_with_dict_value(self, minimal_template_yaml: Path) -> None:
        """Dict slot values should be JSON-serialized."""
        engine = PromptEngine()
        _, user = engine.fill(
            minimal_template_yaml, {"name": "Claude", "score": {"roc_auc": 0.75}}
        )
        assert '"roc_auc"' in user
        assert "0.75" in user

    def test_missing_slot_raises(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        with pytest.raises(MissingSlotError, match="score"):
            engine.fill(minimal_template_yaml, {"name": "Claude"})

    def test_extra_slots_ignored(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        _, user = engine.fill(
            minimal_template_yaml,
            {"name": "Claude", "score": "100", "extra_key": "ignored"},
        )
        assert "Hello Claude, your score is 100." == user

    def test_fallback_path(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        system, user = engine.fill(
            minimal_template_yaml, {}, use_fallback=True
        )
        assert user == "No memory yet."
        assert system == "You are a test bot."

    def test_fallback_without_fallback_defined_raises(self, tmp_path: Path) -> None:
        data = {
            "name": "no_fallback",
            "description": "no fallback",
            "system": "sys",
            "template": "tmpl {x}",
            "slots": ["x"],
        }
        path = tmp_path / "x.yaml"
        path.write_text(yaml.safe_dump(data))
        engine = PromptEngine()
        with pytest.raises(ValueError, match="no fallback_when_no_memory"):
            engine.fill(path, {}, use_fallback=True)

    def test_accepts_preloaded_template(self, minimal_template_yaml: Path) -> None:
        engine = PromptEngine()
        template = engine.load(minimal_template_yaml)
        system, user = engine.fill(template, {"name": "x", "score": "1"})
        assert "x" in user

    def test_propose_architecture_fallback_requests_json(self) -> None:
        """Regression: the cold-start fallback for propose_architecture was
        pure English prose, so the LLM on the very first experiment replied
        with natural-language text and the JSON parser choked every time
        ('exp_001 is always broken'). The fallback MUST instruct the LLM to
        return valid JSON and include a concrete example.
        """
        engine = PromptEngine()
        template = engine.load("config/prompts/propose_architecture/v1.yaml")
        assert template.fallback_when_no_memory is not None

        _, user = engine.fill(template, slots={}, use_fallback=True)

        # The fallback text must tell the LLM to return JSON
        assert "JSON" in user
        assert '"architecture"' in user
        assert '"hyperparams"' in user
        # And it must parse as real JSON (so the example is actually valid)
        import json
        import re
        # Find the first balanced {...} block in the fallback
        start = user.find("{")
        depth = 0
        end = -1
        for i in range(start, len(user)):
            if user[i] == "{":
                depth += 1
            elif user[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        assert end != -1, "fallback has no balanced JSON object"
        parsed = json.loads(user[start : end + 1])
        # Must have the keys the downstream code expects
        assert "architecture" in parsed
        assert "hyperparams" in parsed
        assert "augmentation" in parsed
