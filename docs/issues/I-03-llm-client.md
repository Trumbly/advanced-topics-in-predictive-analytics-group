# I-03 — LLMClient (provider switch + retries)

**Labels:** `track-foundation`, `p0`
**Milestone:** week-1-foundations
**Owner:** Dev A

## Context
PDF: "Abstract the LLM provider". ADR-001 — single `chat()` interface for Ollama/OpenAI/Anthropic.

## Scope
One class, one public method. stdlib `urllib` only, no `requests`/`openai`/`anthropic` deps. Exponential backoff. Error classification.

## Interface
```python
class LLMClient:
    def __init__(self, cfg: LLMConfig, *, http: HTTPPoster | None = None): ...
    def chat(self, messages: list[dict[str, str]], *,
             temperature: float | None = None,
             max_tokens: int | None = None) -> str: ...

class LLMTransientError(Exception): ...
class LLMPermanentError(Exception): ...

class HTTPPoster(Protocol):
    def post(self, url: str, body: bytes, headers: dict[str,str]) -> tuple[int, bytes]: ...
```

Provider routing:
- `ollama` + `openai` → `/v1/chat/completions` OpenAI-compatible
- `anthropic` → `/v1/messages` with `system` extracted from messages

Retry: status in {429, 500, 502, 503, 504, network-error} → transient, exponential backoff `retry_backoff_seconds * 2**attempt`. Status 4xx (non-429) → permanent, fail fast.

## Files
- Create `lab/core/llm.py`
- Create `tests/test_llm_client.py` with fake HTTPPoster

## Acceptance
- Fake HTTP test: success on first 200, retry-then-succeed on 500→200, permanent on 400.
- Provider switch test: same `chat()` call hits different URL shape per provider.
- Structured log event emitted for each call: `{"event":"llm_call","provider":...,"duration_ms":..,"status":..}`.

## Depends on
I-01 (uses `LLMConfig`).
