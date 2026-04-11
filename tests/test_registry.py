"""Unit tests for `registry.registry.ModelRegistry`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.models import ModelRegistryEntry
from registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_registry_yaml(tmp_path: Path) -> Path:
    data = {
        "models": [
            {
                "name": "tiny_cnn",
                "family": "cnn",
                "input_shape": [1, 64, 128],
                "output_dim": 10,
                "parameters_millions": 0.1,
                "pretrained_on": "none",
                "suitability_notes": "Smoke test model.",
                "framework": "torch",
                "import_snippet": "model = None",
            },
            {
                "name": "other_cnn",
                "family": "cnn",
                "input_shape": [3, 224, 224],
                "output_dim": 100,
                "parameters_millions": 2.5,
                "pretrained_on": "imagenet",
                "suitability_notes": "Transfer learning baseline.",
                "framework": "torch",
                "import_snippet": "model = 'something'",
            },
        ]
    }
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_load(self, fake_registry_yaml: Path) -> None:
        registry = ModelRegistry(fake_registry_yaml)
        assert len(registry) == 2
        assert "tiny_cnn" in registry
        assert "other_cnn" in registry

    def test_get(self, fake_registry_yaml: Path) -> None:
        registry = ModelRegistry(fake_registry_yaml)
        entry = registry.get("tiny_cnn")
        assert isinstance(entry, ModelRegistryEntry)
        assert entry.name == "tiny_cnn"
        assert entry.output_dim == 10

    def test_get_unknown_raises(self, fake_registry_yaml: Path) -> None:
        registry = ModelRegistry(fake_registry_yaml)
        with pytest.raises(KeyError, match="not found"):
            registry.get("does_not_exist")

    def test_names_sorted(self, fake_registry_yaml: Path) -> None:
        registry = ModelRegistry(fake_registry_yaml)
        assert registry.names() == ["other_cnn", "tiny_cnn"]

    def test_list_all_sorted(self, fake_registry_yaml: Path) -> None:
        registry = ModelRegistry(fake_registry_yaml)
        entries = registry.list_all()
        assert [e.name for e in entries] == ["other_cnn", "tiny_cnn"]

    def test_to_markdown_includes_all_names(
        self, fake_registry_yaml: Path
    ) -> None:
        registry = ModelRegistry(fake_registry_yaml)
        md = registry.to_markdown()
        assert "### tiny_cnn" in md
        assert "### other_cnn" in md
        assert "Transfer learning baseline" in md
        assert "params: 0.1M" in md
        assert "input_shape: 1x64x128" in md

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ModelRegistry(tmp_path / "nope.yaml")

    def test_real_registry_loads(self) -> None:
        """Smoke test against the checked-in registry."""
        registry = ModelRegistry("registry/models.yaml")
        assert len(registry) >= 3
        assert "cnn_small_v1" in registry
        assert "efficientnet_b0" in registry

    def test_real_registry_uses_num_classes_sentinel(self) -> None:
        """The checked-in registry MUST use the `num_classes` sentinel
        for every entry. Hardcoded ints (like 234) are forbidden because
        the real dataset has 206 classes and that mismatch crashed
        multiple experiments."""
        registry = ModelRegistry("registry/models.yaml")
        for entry in registry.list_all():
            assert entry.output_dim == "num_classes", (
                f"{entry.name} uses hardcoded output_dim {entry.output_dim!r}; "
                "every registry entry must use the 'num_classes' sentinel"
            )

    def test_output_dim_rejects_unknown_string(self, tmp_path: Path) -> None:
        """Any string other than 'num_classes' is rejected by Pydantic."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="num_classes"):
            ModelRegistryEntry.model_validate(
                {
                    "name": "broken",
                    "family": "cnn",
                    "input_shape": [1, 64, 64],
                    "output_dim": "classes",  # wrong sentinel
                    "parameters_millions": 0.1,
                    "pretrained_on": "none",
                    "suitability_notes": "x",
                    "framework": "torch",
                    "import_snippet": "model = None",
                }
            )
