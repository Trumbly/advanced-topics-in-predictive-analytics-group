"""Minimal LLM client with Ollama, OpenAI and Anthropic support.

Uses plain ``urllib`` so the core package has no hard third-party dependency
beyond what pydantic + PyYAML already pull in. Retries with exponential
backoff on transient failures.

Ollama and OpenAI share the same request/response shape (OpenAI-compatible
``/chat/completions``). Anthropic has its own ``/messages`` endpoint with
the ``system`` role pulled out of the messages array, different auth
headers (``x-api-key`` + ``anthropic-version``), and a different response
envelope — handled in ``_chat_anthropic``.

This client is **safe to use** inside the agent itself; the validator's
ban on ``urllib`` applies to the LLM-generated training code, not to the
agent's own code.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable


logger = logging.getLogger("lab.llm")


ANTHROPIC_API_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    pass


class LLMPermanentError(LLMError):
    """Non-retryable (bad request, auth, malformed response, ...)."""


class LLMTransientError(LLMError):
    """Retryable (timeout, 5xx, connection reset, ...)."""


@dataclass
class LLMClient:
    base_url: str = "http://localhost:11434/v1"
    model: str = "gemma4:e4b"
    api_key: str = "ollama"
    provider: str = "ollama"           # ollama | openai | anthropic
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        """Send a chat request. Returns the assistant's message content."""
        provider = (self.provider or "ollama").strip().lower()
        if provider == "anthropic":
            return self._chat_anthropic(messages, temperature=temperature)
        return self._chat_openai_compatible(messages, temperature=temperature)

    # ------------------------------------------------------------------
    # OpenAI / Ollama (same wire format)
    # ------------------------------------------------------------------

    def _chat_openai_compatible(
        self, messages: list[dict[str, str]], *, temperature: float | None
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = self._post_with_retry(url, payload, headers)
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError(f"Malformed LLM response: {exc}")

    # ------------------------------------------------------------------
    # Anthropic (Messages API)
    # ------------------------------------------------------------------

    def _chat_anthropic(
        self, messages: list[dict[str, str]], *, temperature: float | None
    ) -> str:
        # Anthropic takes `system` as a top-level field, not a role. Pull
        # every system message out of the array and concatenate them.
        system_parts: list[str] = []
        user_assistant: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                user_assistant.append({"role": role, "content": content})
            else:
                # Unknown roles get dropped silently — there's no Anthropic
                # equivalent of "function" / "tool" in our prompt engine.
                continue

        payload: dict = {
            "model": self.model,
            "messages": user_assistant,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        url = self.base_url.rstrip("/") + "/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        body = self._post_with_retry(url, payload, headers)
        # Response shape: {"content": [{"type": "text", "text": "..."}, ...], ...}
        try:
            parts = body.get("content", [])
            text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            if not text_parts:
                raise KeyError("no text blocks in content[]")
            return "".join(text_parts)
        except (KeyError, AttributeError, TypeError) as exc:
            raise LLMPermanentError(f"Malformed Anthropic response: {exc}")

    # ------------------------------------------------------------------
    # Shared HTTP + retry
    # ------------------------------------------------------------------

    def _post_with_retry(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        data = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.retry_attempts + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if 500 <= exc.code < 600:
                    logger.warning("LLM HTTP %s on attempt %d", exc.code, attempt)
                    self._sleep_retry(attempt)
                    continue
                raise LLMPermanentError(
                    f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}"
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                logger.warning("LLM transient error on attempt %d: %s", attempt, exc)
                self._sleep_retry(attempt)
                continue
            except json.JSONDecodeError as exc:
                raise LLMPermanentError(f"Malformed LLM response: {exc}")
        raise LLMTransientError(f"LLM call failed after {self.retry_attempts} attempts")

    def _sleep_retry(self, attempt: int) -> None:
        if attempt < self.retry_attempts:
            delay = self.retry_backoff_seconds * (2 ** (attempt - 1))
            logger.info("Backing off %.1fs before retry", delay)
            time.sleep(delay)

    @staticmethod
    def count_tokens(text: str) -> int:
        """Rough token count (~4 chars / token)."""
        return max(1, len(text) // 4)


def format_messages(system: str, user: str, *, extras: Iterable[dict[str, str]] = ()) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    msgs.extend(list(extras))
    msgs.append({"role": "user", "content": user})
    return msgs


__all__ = [
    "LLMClient", "LLMError", "LLMTransientError", "LLMPermanentError",
    "format_messages", "ANTHROPIC_API_VERSION",
]
