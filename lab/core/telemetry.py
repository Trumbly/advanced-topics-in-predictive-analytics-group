"""Structured + human telemetry (ADR-012).

One init per process. Two sinks per study:
- ``experiments/studies/<study_id>/run.log.jsonl`` -- one JSON object per line.
- ``experiments/studies/<study_id>/run.log`` -- human pretty form of same events.

Stdout mirrors the human sink. CLI and UI share the same module-level state,
so either entry point produces an identical event stream.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from lab.config import Settings

EventName = Literal[
    "study_start",
    "study_end",
    "experiment_start",
    "experiment_end",
    "llm_call",
    "validate",
    "execute",
    "recover",
    "judge",
    "submission",
    "epoch",
    "error",
]
LevelName = Literal["info", "warn", "error"]

_VALID_EVENTS: frozenset[str] = frozenset(
    {
        "study_start",
        "study_end",
        "experiment_start",
        "experiment_end",
        "llm_call",
        "validate",
        "execute",
        "recover",
        "judge",
        "submission",
        "epoch",
        "error",
    }
)


class _TelemetryState:
    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.study_id: str | None = None
        self.experiment_id: str | None = None
        self.jsonl_sink: Path | None = None
        self.human_sink: Path | None = None
        self.lock = threading.Lock()


_state = _TelemetryState()


def configure(settings: Settings, *, study_id: str | None = None) -> None:
    """Initialise telemetry sinks for the current process."""
    _state.settings = settings
    _state.study_id = study_id
    if study_id is not None:
        study_dir = Path(settings.paths.experiments_dir) / study_id
        study_dir.mkdir(parents=True, exist_ok=True)
        _state.jsonl_sink = study_dir / "run.log.jsonl"
        _state.human_sink = study_dir / "run.log"
    else:
        _state.jsonl_sink = None
        _state.human_sink = None


def set_experiment(experiment_id: str | None) -> None:
    _state.experiment_id = experiment_id


def log_event(event: EventName, level: LevelName = "info", **fields: Any) -> None:
    if event not in _VALID_EVENTS:
        raise ValueError(f"unknown event: {event}")

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "study_id": _state.study_id,
        "experiment_id": _state.experiment_id,
        "event": event,
        "level": level,
        "fields": _coerce_fields(fields),
    }

    line_jsonl = json.dumps(record, default=str)
    line_human = _format_human(record)

    with _state.lock:
        if _state.jsonl_sink is not None:
            with _state.jsonl_sink.open("a", encoding="utf-8") as fh:
                fh.write(line_jsonl + "\n")
        if _state.human_sink is not None:
            with _state.human_sink.open("a", encoding="utf-8") as fh:
                fh.write(line_human + "\n")
        sys.stdout.write(line_human + "\n")
        sys.stdout.flush()


def log_human(msg: str, *, level: LevelName = "info") -> None:
    """Free-form line to the human sink + stdout (skips JSONL)."""
    _emit_human(msg, level)


def _emit_human(msg: str, level: LevelName) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} [{level.upper()}] {msg}"
    with _state.lock:
        if _state.human_sink is not None:
            with _state.human_sink.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _format_human(record: dict[str, Any]) -> str:
    fields_repr = " ".join(f"{k}={v}" for k, v in record["fields"].items())
    return (
        f"{record['ts']} [{record['level'].upper()}] "
        f"{record['event']}{(' ' + fields_repr) if fields_repr else ''}"
    )


def _coerce_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


# ---------------------------------------------------------------------------
# test-only reset
# ---------------------------------------------------------------------------


def _reset_for_tests() -> None:
    """Wipe telemetry state. ONLY for tests."""
    _state.settings = None
    _state.study_id = None
    _state.experiment_id = None
    _state.jsonl_sink = None
    _state.human_sink = None
