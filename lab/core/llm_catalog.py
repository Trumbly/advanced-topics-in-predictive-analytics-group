"""Discover available Ollama models via its HTTP API.

The UI's `/new` form populates its model dropdown from this helper; the agent
loop only ever talks to one model per study, but the launcher needs to know
what is on disk locally so the user can pick.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterable


def list_ollama_models(
    base_url: str = "http://localhost:11434",
    *,
    timeout_s: float = 2.0,
) -> list[str]:
    """Return the list of model names installed on the local Ollama instance.

    Returns an empty list when Ollama is unreachable so the UI can degrade
    gracefully (the form falls back to a free-text input).
    """
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []
    return sorted({m.get("name") for m in payload.get("models", []) if m.get("name")})


def candidate_models(
    settings_default: str,
    *,
    base_url: str = "http://localhost:11434",
) -> list[str]:
    """Models for the launch form: live Ollama list + the configured default,
    deduped, with the default sorted to the top."""
    live = list_ollama_models(base_url)
    if settings_default and settings_default not in live:
        live = [settings_default, *live]
    elif settings_default and live and settings_default in live:
        live = [settings_default] + [m for m in live if m != settings_default]
    return live
