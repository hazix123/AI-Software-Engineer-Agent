"""Structured logging.

Local dev gets a human-readable console renderer; deployed environments get
newline-delimited JSON that a log pipeline can index.

Two properties matter more than formatting:

1. **Context propagation.** `bind_request_context()` uses contextvars, so every
   log line emitted while handling a request carries `request_id` (and later
   `run_id`, `organization_id`) without any function having to pass it down.

2. **Redaction.** `_redact` runs before any renderer. Secrets cannot leak into
   a log sink even if someone logs a whole settings object or a tool argument
   payload by accident. This is a hard requirement once the agent starts
   handling repository credentials (M3).
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.typing import EventDict, Processor

from aiswe.config.constants import CTX_ORG_ID, CTX_REQUEST_ID, CTX_RUN_ID, CTX_USER_ID

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

    from aiswe.config.settings import LogSettings

_REDACTED = "[redacted]"
_MAX_DEPTH = 6


def _make_redactor(redact_keys: list[str]) -> Processor:
    """Build a processor that masks any key whose name looks sensitive."""
    needles = tuple(k.lower() for k in redact_keys)

    def _scrub(value: Any, depth: int = 0) -> Any:
        if depth >= _MAX_DEPTH:
            return value
        if isinstance(value, dict):
            return {
                k: (
                    _REDACTED
                    if any(n in str(k).lower() for n in needles)
                    else _scrub(v, depth + 1)
                )
                for k, v in value.items()
            }
        if isinstance(value, list | tuple):
            return [_scrub(v, depth + 1) for v in value]
        return value

    def _redact(
        _logger: object,
        _name: str,
        event_dict: EventDict,
    ) -> EventDict:
        return dict(_scrub(dict(event_dict)))  # type: ignore[arg-type]

    return _redact


def configure_logging(settings: LogSettings) -> None:
    """Install structlog. Call exactly once, from the composition root."""
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _make_redactor(settings.redact_keys),
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.format == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.level]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers (uvicorn, sqlalchemy, alembic) through the same sink
    # so we get one uniform stream instead of two competing formats.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.level,
        force=True,
    )
    for noisy in ("uvicorn.access", "uvicorn.error", "sqlalchemy.engine", "alembic"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True


def get_logger(name: str | None = None) -> BoundLogger:
    """Return a bound logger. Prefer `get_logger(__name__)` at module scope."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_request_context(
    *,
    request_id: str,
    user_id: str | None = None,
    organization_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Attach identifiers to every subsequent log line in this async context."""
    ctx: dict[str, str] = {CTX_REQUEST_ID: request_id}
    if user_id:
        ctx[CTX_USER_ID] = user_id
    if organization_id:
        ctx[CTX_ORG_ID] = organization_id
    if run_id:
        ctx[CTX_RUN_ID] = run_id
    bind_contextvars(**ctx)


def clear_request_context() -> None:
    """Drop bound context. Called at the end of each request."""
    clear_contextvars()
