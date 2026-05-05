"""Background watchdog: flag stalled studies as FAILED.

Runs alongside ``StudyRunner`` in a daemon thread. Polls activity files
every few seconds; when no file has been touched for ``timeout_s``, it
rewrites ``study.json`` with ``status="FAILED"`` and appends an ``error``
event so the UI surfaces the stall instead of a forever-RUNNING study.

Activity is the freshest mtime across:
- ``experiments/studies/<id>/run.log.jsonl`` (orchestrator telemetry)
- every ``sandbox/*/stdout.log`` and ``stderr.log`` (training subprocess)

The subprocess can pump stdout for minutes between JSONL events (per-batch
prints from the skeleton), so checking only the JSONL would have killed
real-but-slow runs. Watching stdout.log too keeps the watchdog aligned
with what the user actually sees in the dashboard's stdout panel.

The watchdog cannot terminate a thread hung in a C-level call (e.g.
urllib waiting on a slow LLM socket); it only marks the on-disk state.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from lab.core.models import Study, StudyNotFoundError


_DEFAULT_TIMEOUT_S = 1800  # 30 min — covers a long epoch on CPU
_POLL_FLOOR_S = 5
_POLL_CEILING_S = 30


class Watchdog:
    """Mark a study FAILED when neither JSONL nor any sandbox stdout has
    been touched for ``timeout_s``."""

    def __init__(
        self,
        study_id: str,
        studies_root: Path,
        *,
        timeout_s: int = _DEFAULT_TIMEOUT_S,
        sandbox_root: Path | None = None,
    ):
        self.study_id = study_id
        self.studies_root = Path(studies_root)
        self.sandbox_root = Path(sandbox_root) if sandbox_root else None
        self.timeout_s = max(_POLL_FLOOR_S * 2, timeout_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stalled = False

    # ------------------------------------------------------------------
    # public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"watchdog-{self.study_id}", daemon=True
        )
        self._thread.start()

    def stop(self, *, join_timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    @property
    def stalled(self) -> bool:
        return self._stalled

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        interval = max(_POLL_FLOOR_S, min(_POLL_CEILING_S, self.timeout_s // 4))
        while not self._stop.wait(interval):
            age = self._idle_seconds()
            if age >= self.timeout_s:
                self._mark_stalled(age)
                return

    def _activity_paths(self) -> list[Path]:
        paths: list[Path] = [
            self.studies_root / self.study_id / "run.log.jsonl"
        ]
        if self.sandbox_root and self.sandbox_root.exists():
            # Sandbox is per-study (sandbox/<study_id>/<exp_id>/) — only
            # watch THIS study's subdir so a long-running other study does
            # not keep us happy by accident.
            study_sandbox = self.sandbox_root / self.study_id
            roots = [study_sandbox] if study_sandbox.exists() else [self.sandbox_root]
            for root in roots:
                for stdout_log in root.glob("*/stdout.log"):
                    paths.append(stdout_log)
                for stderr_log in root.glob("*/stderr.log"):
                    paths.append(stderr_log)
        return paths

    def _idle_seconds(self) -> float:
        latest = 0.0
        any_present = False
        for p in self._activity_paths():
            try:
                latest = max(latest, p.stat().st_mtime)
                any_present = True
            except FileNotFoundError:
                continue
        if not any_present:
            return self.timeout_s + 1.0  # nothing on disk yet → treat as stalled
        return max(0.0, time.time() - latest)

    def _mark_stalled(self, age: float) -> None:
        self._stalled = True
        try:
            study = Study.load(self.studies_root, self.study_id)
        except StudyNotFoundError:
            return
        if study.status in {"COMPLETED", "FAILED", "ABORTED"}:
            return  # already terminal — leave it alone

        study.status = "FAILED"
        study.finished_at = datetime.now(timezone.utc)
        study.save(self.studies_root)

        log_path = self.studies_root / self.study_id / "run.log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "study_id": self.study_id,
            "experiment_id": None,
            "event": "error",
            "level": "error",
            "fields": {
                "reason": "stalled",
                "no_log_for_seconds": int(age),
                "watchdog_timeout_seconds": self.timeout_s,
            },
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
