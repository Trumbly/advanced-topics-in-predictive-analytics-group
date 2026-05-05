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
from typing import Iterable, Protocol

from lab.config import LLMConfig

_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger("lab.llm")


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
        self.http: HTTPPoster = http or _UrllibPoster()
        self._sleep = sleep or time.sleep

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

            duration_ms = int((time.monotonic() - t0) * 1000)
            self._log_call(duration_ms, status)
            last_status = status

            if 200 <= status < 300:
                return self._parse_response(payload)

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
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMPermanentError(f"invalid JSON in response: {exc}") from exc

        if self.cfg.provider in ("ollama", "openai"):
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMPermanentError(
                    f"unexpected OpenAI-compatible response shape: {data!r}"
                ) from exc

        if self.cfg.provider == "anthropic":
            try:
                blocks: Iterable[dict[str, object]] = data["content"]
                texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                return "".join(texts)
            except (KeyError, TypeError) as exc:
                raise LLMPermanentError(
                    f"unexpected Anthropic response shape: {data!r}"
                ) from exc

        raise LLMPermanentError(  # pragma: no cover
            f"unsupported provider in response parser: {self.cfg.provider}"
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
