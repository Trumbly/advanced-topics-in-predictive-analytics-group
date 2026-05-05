"""Background watchdog: flag stalled studies as FAILED.

Runs alongside ``StudyRunner`` in a daemon thread. Polls the JSONL telemetry
sink every few seconds; when no new lines have been written for
``timeout_s``, it rewrites ``study.json`` with ``status="FAILED"`` and appends
an ``error`` event so the UI surfaces the stall instead of showing a forever-
RUNNING study.

The watchdog cannot terminate a thread that is hung in a C-level call (e.g.
urllib waiting on a slow LLM socket); it only marks the on-disk state. The
runner thread may continue spinning until the OS reclaims it, but the user
gets a definitive FAILED status and a reason in the log.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from lab.core.models import Study, StudyNotFoundError


_DEFAULT_TIMEOUT_S = 600  # 10 minutes
_POLL_FLOOR_S = 5
_POLL_CEILING_S = 30


class Watchdog:
    """Mark a study FAILED when its log goes silent for too long."""

    def __init__(
        self,
        study_id: str,
        studies_root: Path,
        *,
        timeout_s: int = _DEFAULT_TIMEOUT_S,
    ):
        self.study_id = study_id
        self.studies_root = Path(studies_root)
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
        log_path = self.studies_root / self.study_id / "run.log.jsonl"
        # Bound poll interval so short timeouts react quickly but long ones
        # do not waste CPU.
        interval = max(_POLL_FLOOR_S, min(_POLL_CEILING_S, self.timeout_s // 4))
        deadline = time.time() + self.timeout_s

        while not self._stop.wait(interval):
            try:
                age = self._idle_seconds(log_path)
            except FileNotFoundError:
                age = self.timeout_s + 1  # nothing written yet either
            if age >= self.timeout_s:
                self._mark_stalled(age)
                return
            deadline = time.time() + self.timeout_s

    def _idle_seconds(self, log_path: Path) -> float:
        if not log_path.exists():
            raise FileNotFoundError(str(log_path))
        return max(0.0, time.time() - log_path.stat().st_mtime)

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
