"""Apply LLM-emitted SEARCH/REPLACE patches to existing code.

The recover_from_error prompt asks the LLM to fix only the lines implicated
by the error, in the SEARCH/REPLACE format::

    <<<<<<< SEARCH
    <exact lines from the current block>
    =======
    <replacement lines>
    >>>>>>> REPLACE

Multiple patches are allowed. ``apply_patches`` returns the new code, or
``None`` if any patch's SEARCH text is not present verbatim in the source
(so the caller can fall back to a full-block regeneration).
"""

from __future__ import annotations

import re

_PATCH_RE = re.compile(
    r"<{3,}\s*SEARCH\s*\n(?P<search>.*?)\n={3,}\s*\n(?P<replace>.*?)\n>{3,}\s*REPLACE",
    flags=re.DOTALL,
)


def parse_patches(text: str) -> list[tuple[str, str]]:
    """Return ``[(search, replace), ...]`` extracted from the LLM reply."""
    return [(m.group("search"), m.group("replace")) for m in _PATCH_RE.finditer(text)]


def apply_patches(code: str, patches: list[tuple[str, str]]) -> str | None:
    """Apply each SEARCH/REPLACE pair to ``code`` (first match per patch).

    Returns the patched code on success, ``None`` if any SEARCH text is not
    present verbatim — the caller should then fall back to the legacy
    full-block replacement path.
    """
    if not patches:
        return None
    out = code
    for search, replace in patches:
        if search not in out:
            return None
        out = out.replace(search, replace, 1)
    return out


def looks_like_patches(text: str) -> bool:
    return bool(_PATCH_RE.search(text))
