"""FastAPI + HTMX web dashboard.

Read-only file-based except for three write endpoints (publish toggle,
tag editor, prompt version draft). The orchestrator is never imported
directly by the UI — interaction is mediated by writing files that the
CLI picks up in a separate process.
"""
