"""Minimal OpenAI-compatible LLM client (works with Ollama or OpenAI).

Uses plain ``urllib`` so the core package has no hard third-party dependency
beyond what pydantic + PyYAML already pull in. Retries with exponential
backoff on transient failures.

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
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        """Send a chat request. Returns the assistant's message content."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(1, self.retry_attempts + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                    parsed = json.loads(body)
                    return parsed["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                if 500 <= exc.code < 600:
                    logger.warning("LLM HTTP %s on attempt %d", exc.code, attempt)
                    self._sleep_retry(attempt)
                    continue
                raise LLMPermanentError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                logger.warning("LLM transient error on attempt %d: %s", attempt, exc)
                self._sleep_retry(attempt)
                continue
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
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


__all__ = ["LLMClient", "LLMError", "LLMTransientError", "LLMPermanentError", "format_messages"]
