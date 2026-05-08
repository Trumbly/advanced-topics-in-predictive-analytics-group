"""Provider-agnostic LLM client (ADR-001).

Single ``chat(messages) -> str`` surface across Ollama, OpenAI, and Anthropic.
Stdlib ``urllib`` only — no ``requests``, ``openai``, or ``anthropic`` deps.
Exponential backoff on transient HTTP failures (429 + 5xx + network).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol

from lab.config import LLMConfig

_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger("lab.llm")


@dataclass(frozen=True)
class LLMCallStats:
    """One-call generation metrics for the dashboard.

    ``ttft_seconds`` is best-effort: most OpenAI-compatible servers (incl.
    Ollama via ``/v1/chat/completions``) ship the response in a single
    non-streamed shot, so we cannot measure the first-token latency
    independently of total time. When the response advertises a separate
    ``prompt_eval_duration_ns`` (Ollama), we use it; otherwise the field
    stays ``None``.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_seconds: float
    tps: float | None
    ttft_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LLMTransientError(Exception):
    """Retryable failure (429, 5xx, network error)."""


class LLMPermanentError(Exception):
    """Non-retryable failure (4xx, malformed body, configuration error)."""


class HTTPPoster(Protocol):
    def post(
        self, url: str, body: bytes, headers: dict[str, str]
    ) -> tuple[int, bytes]:  # pragma: no cover - protocol
        ...


class _UrllibPoster:
    """Default HTTPPoster backed by stdlib ``urllib.request``."""

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds

    def post(
        self, url: str, body: bytes, headers: dict[str, str]
    ) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read() if hasattr(exc, "read") else b""
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMTransientError(f"network error: {exc}") from exc


class LLMClient:
    """Single ``chat(messages)`` interface that fans out per provider."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        http: HTTPPoster | None = None,
        sleep: "callable[[float], None] | None" = None,
    ):
        self.cfg = cfg
        # Honour the LLMConfig timeout (default 300s) for stdlib urllib;
        # injected HTTPPoster instances (tests) ignore this since they
        # don't make real network calls.
        self.http: HTTPPoster = http or _UrllibPoster(timeout_seconds=cfg.timeout_seconds)
        self._sleep = sleep or time.sleep
        # Last successful call's stats. Read by the orchestrator after
        # each chat() to stamp the surrounding Task with token counts +
        # throughput. None until the first call lands.
        self.last_stats: LLMCallStats | None = None
        # Verbatim messages + response of the last successful call so the
        # orchestrator can stash them on the surrounding Task. Stored as
        # plain dicts/strings (not pydantic) since the messages list is
        # already JSON-serialisable. None until the first call lands.
        self.last_messages: list[dict[str, str]] | None = None
        self.last_response: str | None = None

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url, body, headers = self._build_request(
            messages,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
        )

        last_status: int | None = None
        for attempt in range(self.cfg.retry_attempts + 1):
            t0 = time.monotonic()
            try:
                status, payload = self.http.post(url, body, headers)
            except LLMTransientError as exc:
                self._log_call(0, "network_error")
                if attempt >= self.cfg.retry_attempts:
                    raise
                self._sleep(self._backoff(attempt))
                continue

            duration_s = time.monotonic() - t0
            self._log_call(int(duration_s * 1000), status)
            last_status = status

            if 200 <= status < 300:
                content, stats = self._parse_response_with_stats(
                    payload, total_seconds=duration_s
                )
                self.last_stats = stats
                # Snapshot the in/out for the orchestrator so the UI can
                # show "what did the LLM see, what did it say" per step.
                # ``messages`` is the caller's list (no defensive copy
                # needed — caller never mutates it after passing it in).
                self.last_messages = list(messages)
                self.last_response = content
                return content

            if status in _TRANSIENT_STATUSES:
                if attempt >= self.cfg.retry_attempts:
                    raise LLMTransientError(
                        f"transient HTTP {status} after {attempt + 1} attempts: {_safe_decode(payload)}"
                    )
                self._sleep(self._backoff(attempt))
                continue

            # 4xx non-429: permanent
            raise LLMPermanentError(
                f"permanent HTTP {status}: {_safe_decode(payload)}"
            )

        raise LLMTransientError(
            f"exhausted retries (last_status={last_status})"
        )  # pragma: no cover - guarded above

    # ------------------------------------------------------------------
    # provider routing
    # ------------------------------------------------------------------

    def _build_request(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, bytes, dict[str, str]]:
        provider = self.cfg.provider
        if provider in ("ollama", "openai"):
            return self._build_openai_compatible(messages, temperature, max_tokens)
        if provider == "anthropic":
            return self._build_anthropic(messages, temperature, max_tokens)
        raise LLMPermanentError(f"unsupported provider: {provider}")  # pragma: no cover

    def _build_openai_compatible(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> tuple[str, bytes, dict[str, str]]:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key and self.cfg.provider == "openai":
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return url, _encode(payload), headers

    def _build_anthropic(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> tuple[str, bytes, dict[str, str]]:
        url = self.cfg.base_url.rstrip("/") + "/v1/messages"
        system_parts: list[str] = []
        rest: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                rest.append(m)
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.cfg.api_key:
            headers["x-api-key"] = self.cfg.api_key
        payload: dict[str, object] = {
            "model": self.cfg.model,
            "messages": rest,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return url, _encode(payload), headers

    # ------------------------------------------------------------------
    # parsing + helpers
    # ------------------------------------------------------------------

    def _parse_response(self, payload: bytes) -> str:
        # Kept for back-compat; new code goes through _parse_response_with_stats.
        text, _ = self._parse_response_with_stats(payload, total_seconds=0.0)
        return text

    def _parse_response_with_stats(
        self, payload: bytes, *, total_seconds: float
    ) -> tuple[str, LLMCallStats]:
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMPermanentError(f"invalid JSON in response: {exc}") from exc

        if self.cfg.provider in ("ollama", "openai"):
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMPermanentError(
                    f"unexpected OpenAI-compatible response shape: {data!r}"
                ) from exc
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            ttft_seconds = _ollama_prompt_eval_seconds(data)
        elif self.cfg.provider == "anthropic":
            try:
                blocks: Iterable[dict[str, object]] = data["content"]
                content = "".join(
                    b.get("text", "") for b in blocks if b.get("type") == "text"
                )
            except (KeyError, TypeError) as exc:
                raise LLMPermanentError(
                    f"unexpected Anthropic response shape: {data!r}"
                ) from exc
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("output_tokens") or 0)
            ttft_seconds = None
        else:
            raise LLMPermanentError(  # pragma: no cover
                f"unsupported provider in response parser: {self.cfg.provider}"
            )

        tps = (
            completion_tokens / total_seconds
            if completion_tokens > 0 and total_seconds > 0
            else None
        )
        return content, LLMCallStats(
            provider=self.cfg.provider,
            model=self.cfg.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_seconds=round(total_seconds, 4),
            tps=round(tps, 2) if tps is not None else None,
            ttft_seconds=round(ttft_seconds, 4) if ttft_seconds is not None else None,
        )

    def _backoff(self, attempt: int) -> float:
        return self.cfg.retry_backoff_seconds * (2 ** attempt)

    def _log_call(self, duration_ms: int, status: int | str) -> None:
        logger.info(
            "llm_call",
            extra={
                "event": "llm_call",
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "duration_ms": duration_ms,
                "status": status,
            },
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _encode(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _safe_decode(buf: bytes) -> str:
    try:
        return buf.decode("utf-8", errors="replace")[:500]
    except Exception:  # pragma: no cover
        return repr(buf)[:500]


def _ollama_prompt_eval_seconds(data: dict) -> float | None:
    """Ollama exposes ``prompt_eval_duration`` (ns) on its OpenAI-compat
    responses; treat that as a TTFT proxy. Returns None when absent."""
    raw = data.get("prompt_eval_duration")
    if raw is None:
        return None
    try:
        return float(raw) / 1e9
    except (TypeError, ValueError):
        return None
