"""Request correlation middleware.

Accepts an inbound `X-Request-ID` (so a trace survives across service hops) or
mints one. Binds it into the structlog contextvars so every downstream log
line -- in middleware, use cases, adapters, and later in agent nodes -- carries
it without anyone passing it around.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from aiswe.config.constants import HEADER_REQUEST_ID
from aiswe.infrastructure.observability.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)

if TYPE_CHECKING:
    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.requests import Request
    from starlette.responses import Response

logger = get_logger(__name__)

_SILENT_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get(HEADER_REQUEST_ID) or uuid.uuid4().hex
        request.state.request_id = request_id
        bind_request_context(request_id=request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The error handler owns the response body; we only record timing
            # and re-raise so the exception middleware can do its job.
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "http.request.failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
            )
            raise
        finally:
            clear_request_context()

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[HEADER_REQUEST_ID] = request_id

        # Probe endpoints fire every few seconds; logging them buries the signal.
        if request.url.path not in _SILENT_PATHS:
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
        return response
