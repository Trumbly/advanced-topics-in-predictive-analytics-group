"""Edit ``config/config.yaml`` from the browser.

Two views:

  * structured form covering the fields teams actually change (LLM, budgets,
    training knobs, UI host/port, publishing defaults)
  * advanced raw-YAML textarea for anything not in the form

On save we validate the merged result against the ``Settings`` pydantic
model before writing — an invalid edit is rejected with an error banner
instead of corrupting the file. A ``.bak`` copy is always written first.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from lab.config import Settings, load_settings
from lab.tasks.registry import list_available_tasks


router = APIRouter(prefix="/config")


def _config_path(settings: Settings) -> Path:
    return settings.repo_root / "config" / "config.yaml"


def _read_raw(settings: Settings) -> dict[str, Any]:
    path = _config_path(settings)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _write_raw(settings: Settings, data: dict[str, Any]) -> Path:
    path = _config_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_suffix(".yaml.bak"))
    path.write_text(
        "# Edited via the lab UI.\n"
        "# To edit safely by hand, keep this as a YAML dict and run\n"
        "# `python -m lab config` to reprint the merged configuration.\n\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )
    return path


@router.get("", response_class=HTMLResponse)
async def edit_form(request: Request, error: str | None = None, saved: str | None = None):
    settings = request.app.state.settings
    raw = _read_raw(settings)
    return request.app.state.templates.TemplateResponse(
        request,
        "config.html",
        {
            "raw": raw,
            "raw_yaml": yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
            "tasks": list_available_tasks(settings),
            "error": error,
            "saved": saved,
        },
    )


@router.post("")
async def save_form(
    request: Request,
    mode: str = Form("form"),
    # Structured fields — all optional so missing ones don't blow away existing values.
    default_task: str = Form(""),
    env_prefix: str = Form(""),
    llm_provider: str = Form(""),
    llm_base_url: str = Form(""),
    llm_default_model: str = Form(""),
    llm_api_key: str = Form(""),
    llm_temperature: str = Form(""),
    llm_max_tokens: str = Form(""),
    llm_timeout_seconds: str = Form(""),
    budget_max_experiments: str = Form(""),
    budget_max_wallclock_minutes: str = Form(""),
    budget_max_experiment_seconds: str = Form(""),
    budget_max_epochs_per_run: str = Form(""),
    budget_max_recovery_attempts: str = Form(""),
    budget_max_codegen_retries: str = Form(""),
    context_max_prompt_tokens: str = Form(""),
    context_memory_top_k: str = Form(""),
    training_device: str = Form(""),
    training_batch_size: str = Form(""),
    training_num_workers: str = Form(""),
    training_persistent_workers: str = Form("false"),
    training_prefetch_factor: str = Form(""),
    logging_level: str = Form(""),
    logging_console: str = Form("false"),
    logging_structured: str = Form("false"),
    ui_host: str = Form(""),
    ui_port: str = Form(""),
    ui_enable_live_sse: str = Form("false"),
    publishing_default_publish: str = Form("false"),
    publishing_default_tags: str = Form(""),
    raw_yaml: str = Form(""),
):
    settings = request.app.state.settings

    if mode == "raw":
        try:
            data = yaml.safe_load(raw_yaml)
            if not isinstance(data, dict):
                raise ValueError("top-level YAML must be a mapping")
        except (yaml.YAMLError, ValueError) as exc:
            return RedirectResponse(f"/config?error=YAML+error%3A+{exc}", status_code=303)
    else:
        data = _read_raw(settings)
        _merge_form(data, {
            "default_task": default_task,
            "env_prefix": env_prefix,
            "llm": {
                "provider": llm_provider,
                "base_url": llm_base_url,
                "default_model": llm_default_model,
                "api_key": llm_api_key,
                "temperature": _maybe_float(llm_temperature),
                "max_tokens": _maybe_int(llm_max_tokens),
                "timeout_seconds": _maybe_float(llm_timeout_seconds),
            },
            "compute_budget": {
                "max_experiments": _maybe_int(budget_max_experiments),
                "max_wallclock_minutes": _maybe_int(budget_max_wallclock_minutes),
                "max_experiment_seconds": _maybe_int(budget_max_experiment_seconds),
                "max_epochs_per_run": _maybe_int(budget_max_epochs_per_run),
                "max_recovery_attempts": _maybe_int(budget_max_recovery_attempts),
                "max_codegen_retries": _maybe_int(budget_max_codegen_retries),
            },
            "context": {
                "max_prompt_tokens": _maybe_int(context_max_prompt_tokens),
                "memory_top_k": _maybe_int(context_memory_top_k),
            },
            "training": {
                "device": training_device,
                "batch_size": _maybe_int(training_batch_size),
                "num_workers": _maybe_int(training_num_workers),
                "persistent_workers": _as_bool(training_persistent_workers),
                "prefetch_factor": _maybe_int(training_prefetch_factor),
            },
            "logging": {
                "level": logging_level,
                "console": _as_bool(logging_console),
                "structured": _as_bool(logging_structured),
            },
            "ui": {
                "host": ui_host,
                "port": _maybe_int(ui_port),
                "enable_live_sse": _as_bool(ui_enable_live_sse),
            },
            "publishing": {
                "default_publish": _as_bool(publishing_default_publish),
                "default_tags": [t.strip() for t in publishing_default_tags.split(",") if t.strip()],
            },
        })

    # Validate against Settings before writing. Anything that would prevent
    # the app from booting gets rejected up front with a readable message.
    try:
        Settings.model_validate({**data, "repo_root": settings.repo_root, "task_config": {}})
    except Exception as exc:  # noqa: BLE001
        return RedirectResponse(f"/config?error=validation+failed%3A+{exc}", status_code=303)

    path = _write_raw(settings, data)

    # Hot-swap the settings object in app state so subsequent requests see
    # the new values without a server restart.
    request.app.state.settings = load_settings(repo_root=settings.repo_root)
    return RedirectResponse(f"/config?saved={path.name}", status_code=303)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_bool(raw: str) -> bool:
    return raw.lower() in {"1", "true", "on", "yes"}


def _maybe_int(raw: str) -> int | None:
    try:
        return int(raw) if raw != "" else None
    except ValueError:
        return None


def _maybe_float(raw: str) -> float | None:
    try:
        return float(raw) if raw != "" else None
    except ValueError:
        return None


def _merge_form(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """Deep-merge ``updates`` into ``target`` skipping empty strings and
    ``None`` values so blank form fields don't blow away existing config."""
    for k, v in updates.items():
        if isinstance(v, dict):
            node = target.setdefault(k, {})
            if not isinstance(node, dict):
                node = {}
                target[k] = node
            _merge_form(node, v)
        elif v is None:
            continue
        elif isinstance(v, str):
            if v != "":
                target[k] = v
        elif isinstance(v, list):
            target[k] = v
        else:
            target[k] = v
