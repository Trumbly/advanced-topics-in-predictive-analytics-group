"""Read + persist `config/config.yaml` for the UI settings page.

ADR-002 says config is YAML only — UI edits go to disk and are picked up
by every fresh ``load_settings()`` call (cmd_run loads at study start).
The running uvicorn process keeps its captured ``Settings`` object until
it is restarted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from lab.config import Settings, load_settings

_GLOBAL_CONFIG = "config/config.yaml"


def repo_root_for(settings: Settings) -> Path:
    """Reverse the convention used by load_settings to find the repo root."""
    # paths.experiments_dir is "experiments/studies" relative to repo root.
    # The settings module's _default_root() is .../lab/.. so use that as a
    # robust fallback by importing the helper.
    from lab.config import _default_root

    return _default_root()


def read_global_yaml(repo_root: Path) -> dict[str, Any]:
    path = repo_root / _GLOBAL_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_global_yaml(repo_root: Path, payload: dict[str, Any]) -> None:
    path = repo_root / _GLOBAL_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_form_into_yaml(
    current: dict[str, Any], form: dict[str, str]
) -> dict[str, Any]:
    """Merge the flat ``section.field`` form keys back into nested YAML.

    Type coercion is best-effort: int / float / bool / str. Unknown nested
    keys are passed through so the loader's strict validation surfaces
    typos instead of silently dropping them.
    """
    new = _deep_copy(current)
    for raw_key, raw_value in form.items():
        if "." not in raw_key:
            continue
        section, _, field = raw_key.partition(".")
        section_obj = new.setdefault(section, {})
        if not isinstance(section_obj, dict):
            continue
        section_obj[field] = _coerce(raw_value, current.get(section, {}).get(field))
    return new


def validate_yaml(repo_root: Path, payload: dict[str, Any]) -> Settings:
    """Validate the prospective YAML by running it through ``Settings``.

    Re-loads the task config from disk (ADR-002 still applies) so we only
    swing the global tree. Raises pydantic.ValidationError on failure.
    """
    task_name = payload.get("default_task", "track_b")
    task_yaml = (
        repo_root / "config" / "tasks" / f"{task_name}.yaml"
    )
    raw = dict(payload)
    raw["task"] = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    raw["task_name"] = task_name
    return Settings(**raw)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _deep_copy(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _deep_copy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_deep_copy(v) for v in d]
    return d


def _coerce(value: str, current: Any) -> Any:
    """Coerce a form string back into the type its YAML neighbour suggests."""
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "on", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(current, float):
        try:
            return float(value)
        except ValueError:
            return value
    return value
