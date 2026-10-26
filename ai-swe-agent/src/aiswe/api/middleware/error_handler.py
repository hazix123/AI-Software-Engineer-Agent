"""Exception handlers.

The ONLY place in the codebase where a domain error becomes an HTTP status
code. Because this mapping is centralised:

* the domain layer never imports FastAPI;
* every error the API can emit has exactly one shape (RFC 9457);
* an unhandled exception cannot leak a stack trace to a client in production.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from aiswe.api.schemas.common import ProblemDetail
from aiswe.config.constants import PROBLEM_CONTENT_TYPE, PROBLEM_TYPE_BASE
from aiswe.domain.errors import (
    BudgetExceededError,
    ConfigurationError,
    ConflictError,
    DomainError,
    ForbiddenError,
    InfrastructureError,
    NotFoundError,
    RateLimitError,
    UnauthorizedError,
    ValidationError,
)
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from starlette.responses import Response

logger = get_logger(__name__)

# Domain error -> HTTP status. Add a new error type here and nowhere else.
_STATUS_MAP: Final[dict[type[DomainError], int]] = {
    ValidationError: HTTPStatus.UNPROCESSABLE_ENTITY,
    NotFoundError: HTTPStatus.NOT_FOUND,
    ConflictError: HTTPStatus.CONFLICT,
    UnauthorizedError: HTTPStatus.UNAUTHORIZED,
    ForbiddenError: HTTPStatus.FORBIDDEN,
    RateLimitError: HTTPStatus.TOO_MANY_REQUESTS,
    BudgetExceededError: HTTPStatus.PAYMENT_REQUIRED,
    InfrastructureError: HTTPStatus.SERVICE_UNAVAILABLE,
    ConfigurationError: HTTPStatus.INTERNAL_SERVER_ERROR,
}


def _problem_response(
    request: Request,
    *,
    status: int,
    code: str,
    detail: str,
    errors: list[dict[str, object]] | None = None,
) -> JSONResponse:
    problem = ProblemDetail(
        type=f"{PROBLEM_TYPE_BASE}/{code}",
        title=code,
        status=status,
        detail=detail,
        instance=str(request.url.path),
        request_id=getattr(request.state, "request_id", None),
        errors=errors,  # type: ignore[arg-type]
    )
    return JSONResponse(
        status_code=status,
        content=problem.model_dump(exclude_none=True),
        media_type=PROBLEM_CONTENT_TYPE,
    )


async def domain_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, DomainError)  # noqa: S101 - guaranteed by registration
    status = _STATUS_MAP.get(type(exc), HTTPStatus.BAD_REQUEST)

    log = logger.exception if status >= HTTPStatus.INTERNAL_SERVER_ERROR else logger.warning
    log("domain_error", code=exc.code, status=status, detail=exc.message)

    return _problem_response(
        request,
        status=status,
        code=exc.code,
        detail=exc.message,
        errors=[exc.details] if exc.details else None,
    )


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    """Pydantic/FastAPI schema failures. Reshaped into the same problem format."""
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    errors = [
        {
            "field": ".".join(str(p) for p in err["loc"][1:]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return _problem_response(
        request,
        status=HTTPStatus.UNPROCESSABLE_ENTITY,
        code="validation_error",
        detail="The request payload failed validation.",
        errors=errors,
    )


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Starlette's own 404/405/etc, normalised into the problem format."""
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    code = HTTPStatus(exc.status_code).name.lower()
    return _problem_response(
        request,
        status=exc.status_code,
        code=code,
        detail=str(exc.detail),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Last line of defence.

    The stack trace goes to the logs. The client gets a request id and nothing
    else -- internal details are never echoed back.
    """
    logger.exception("unhandled_exception", error=type(exc).__name__)
    return _problem_response(
        request,
        status=HTTPStatus.INTERNAL_SERVER_ERROR,
        code="internal_error",
        detail="An unexpected error occurred. Quote the request id when reporting this.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install all handlers. Called once from the app factory."""
    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
