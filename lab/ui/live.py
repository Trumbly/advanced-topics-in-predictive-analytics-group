"""Server-Sent Events (SSE) for live stdout tailing.

Each connection tails one ``sandbox/<exp_id>/stdout.log`` file with a tiny
poll loop. SSE is simpler than WebSockets for read-only streams.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator


async def tail_file(path: Path, *, poll_seconds: float = 0.5) -> AsyncIterator[str]:
    """Async generator that yields new lines from ``path`` as they appear.

    Starts by yielding the existing content, then polls for new lines. Stops
    when the client disconnects (consumer breaks the iteration).
    """
    # 1) Prime with existing content
    if path.exists():
        with path.open() as fh:
            existing = fh.read()
            if existing:
                for line in existing.splitlines():
                    yield line
            pos = fh.tell()
    else:
        pos = 0

    while True:
        await asyncio.sleep(poll_seconds)
        if not path.exists():
            continue
        with path.open() as fh:
            fh.seek(pos)
            chunk = fh.read()
            pos = fh.tell()
        if not chunk:
            continue
        for line in chunk.splitlines():
            yield line


def format_sse(data: str, *, event: str | None = None) -> str:
    """Serialize a string for an SSE frame. Multi-line data is split."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    for ln in data.splitlines() or [""]:
        lines.append(f"data: {ln}")
    lines.append("")  # frame terminator
    lines.append("")
    return "\n".join(lines)


__all__ = ["tail_file", "format_sse"]
