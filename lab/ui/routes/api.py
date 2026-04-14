"""JSON endpoints consumed by HTMX fragments and test tooling."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from fastapi import APIRouter, Request

from lab.ui import loaders


router = APIRouter(prefix="/api")


@router.get("/studies")
async def list_studies(request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    studies = loaders.iter_studies(experiments_dir)
    return {"studies": [s.__dict__ for s in studies]}


@router.get("/studies/{study_id}")
async def study_json(study_id: str, request: Request):
    settings = request.app.state.settings
    experiments_dir = settings.abspath(settings.paths.experiments)
    study = loaders.load_study(experiments_dir, study_id)
    if not study:
        return {"error": "not_found"}, 404
    return study.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Provider model listing — used by /config to populate the model dropdown
# based on whatever provider the user selected. Errors are returned as a
# JSON payload rather than raised so the UI can show an inline hint.
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models(provider: str = "", base_url: str = "", api_key: str = "",
                      request: Request = None):
    provider = (provider or "ollama").strip().lower()
    base_url = (base_url or "").rstrip("/")

    # If api_key is empty on an OpenAI request, fall back to whatever is in
    # the live settings — the user may already have saved one.
    if provider == "openai" and not api_key and request is not None:
        api_key = request.app.state.settings.llm.api_key

    try:
        if provider == "ollama":
            models = _fetch_ollama_models(base_url or "http://localhost:11434/v1")
        elif provider == "openai":
            models = _fetch_openai_models(base_url or "https://api.openai.com/v1", api_key)
        elif provider == "anthropic":
            models = _fetch_anthropic_models(base_url or "https://api.anthropic.com/v1", api_key)
        else:
            return {"error": f"unknown provider: {provider!r}", "models": []}
    except PermissionError as exc:
        return {"error": f"auth failed: {exc}", "models": []}
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        return {
            "error": f"cannot reach {provider} at {base_url!r}: {exc}",
            "models": [],
        }
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return {"error": f"unexpected response: {exc}", "models": []}
    return {"models": models, "provider": provider}


def _http_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 10.0):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_ollama_models(base_url: str) -> list[str]:
    """Ollama exposes /api/tags for a native model list, and /v1/models
    for OpenAI-compat. Try native first, fall back to OpenAI-compat."""
    native_root = base_url[:-3] if base_url.endswith("/v1") else base_url
    try:
        data = _http_json(f"{native_root}/api/tags")
        return sorted(m["name"] for m in data.get("models", []) if isinstance(m, dict))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        data = _http_json(f"{base_url}/models")
        return sorted(m["id"] for m in data.get("data", []) if isinstance(m, dict))


def _fetch_anthropic_models(base_url: str, api_key: str) -> list[str]:
    if not api_key or api_key == "ollama":
        raise PermissionError("Anthropic requires an API key")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        data = _http_json(f"{base_url}/models", headers=headers, timeout=15.0)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError("API key rejected") from exc
        raise
    return sorted(m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id"))


def _fetch_openai_models(base_url: str, api_key: str) -> list[str]:
    if not api_key or api_key == "ollama":
        raise PermissionError("OpenAI requires an API key")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        data = _http_json(f"{base_url}/models", headers=headers, timeout=15.0)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError("API key rejected") from exc
        raise
    # OpenAI returns many utility models — keep only chat-capable ones by
    # convention (names containing "gpt" or starting with "o"). Falls back
    # to the full list if the filter produces nothing.
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    chat = [m for m in ids if ("gpt" in m.lower()) or m.startswith(("o1", "o3", "o4"))]
    return sorted(chat or ids)
