"""Server-Sent Events (SSE) helpers for live log tailing."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator


async def tail_file(
    path: Path,
    *,
    poll_seconds: float = 0.5,
    from_end: bool = False,
) -> AsyncIterator[str]:
    """Async generator that yields new lines from ``path`` as they appear.

    Starts by yielding the existing content, then polls for new lines. Stops
    when the client disconnects (consumer breaks the iteration).
    """
    # 1) Prime with existing content unless the caller explicitly wants to
    # follow only NEW lines (useful when the page already rendered an initial
    # tail and SSE should not duplicate it).
    if path.exists():
        with path.open() as fh:
            if not from_end:
                existing = fh.read()
                if existing:
                    for line in existing.splitlines():
                        yield line
            else:
                fh.seek(0, 2)
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
