"""Shared transport schemas.

These are DTOs at the edge. Domain entities are never serialised directly --
that would couple the public API contract to internal modelling decisions and
make every refactor a breaking change.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details. The ONE error shape for the whole API.

    Clients switch on `type` (a stable URI), never on `detail` (prose that may
    be reworded at any time).
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "type": "https://errors.aiswe.dev/not_found",
        "title": "not_found",
        "status": 404,
        "detail": "The requested resource was not found.",
        "instance": "/v1/projects/2f1c...",
        "request_id": "01J8Z9...",
    }})

    type: str = Field(description="Stable URI identifying the error class.")
    title: str = Field(description="Machine-readable error code.")
    status: int = Field(description="HTTP status code.")
    detail: str = Field(description="Human-readable explanation.")
    instance: str | None = Field(default=None, description="URI of the failing request.")
    request_id: str | None = Field(default=None, description="Correlation id for support.")
    errors: list[dict[str, Any]] | None = Field(
        default=None, description="Field-level validation failures, when applicable."
    )


class HealthResponse(BaseModel):
    """Liveness. Answers only: is this process running?"""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str


class DependencyStatus(BaseModel):
    name: str
    healthy: bool


class ReadinessResponse(BaseModel):
    """Readiness. Answers: can this process actually serve traffic?

    Returns 503 when any dependency is down, so the orchestrator pulls the pod
    out of the load balancer instead of routing requests into a black hole.
    """

    status: Literal["ready", "degraded"]
    dependencies: list[DependencyStatus]


class Page(BaseModel, Generic[T]):
    """Cursor-paginated envelope.

    Cursor, not offset: offset pagination silently skips and duplicates rows
    when the underlying table is being written to, which for a run log is
    always.
    """

    items: list[T]
    next_cursor: str | None = None
    has_more: bool = False
