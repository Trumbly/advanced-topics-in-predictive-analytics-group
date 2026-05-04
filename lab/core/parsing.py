"""Strict JSON proposal parsing with one retry (ADR-018)."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from lab.core.llm import LLMClient
from lab.core.models import Proposal


class ProposalParseError(Exception):
    """Raised when the LLM cannot produce a valid Proposal JSON after one retry."""


def parse_proposal(
    text: str,
    *,
    retry_client: LLMClient | None = None,
    retry_messages: list[dict] | None = None,
) -> Proposal:
    try:
        return _parse_once(text)
    except (json.JSONDecodeError, ValidationError) as exc:
        if retry_client is None or retry_messages is None:
            raise ProposalParseError(f"invalid proposal JSON: {exc}") from exc
        followup = list(retry_messages) + [
            {
                "role": "user",
                "content": (
                    f"Your previous response could not be parsed: {exc}.\n"
                    "Reply with ONLY the proposal JSON object, no prose, no code fences."
                ),
            }
        ]
        retried = retry_client.chat(followup)
        try:
            return _parse_once(retried)
        except (json.JSONDecodeError, ValidationError) as exc2:
            raise ProposalParseError(
                f"proposal still invalid after retry: {exc2}"
            ) from exc2


def _parse_once(text: str) -> Proposal:
    cleaned = _strip_fences(text).strip()
    if not cleaned:
        raise json.JSONDecodeError("empty response", "", 0)
    payload = json.loads(cleaned)
    return Proposal.model_validate(payload)


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?", flags=re.IGNORECASE)


def _strip_fences(text: str) -> str:
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) >= 3:
        body = parts[1].lstrip()
        # Drop language hint (json, python, ...) on the first line.
        first_line, sep, rest = body.partition("\n")
        if sep and first_line.strip().lower() in {"json", "python"}:
            return rest
        return body
    return _FENCE_RE.sub("", text)
