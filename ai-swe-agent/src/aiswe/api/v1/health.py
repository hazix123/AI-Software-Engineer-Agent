"""Health endpoints.

The liveness/readiness split is operationally load-bearing:

* `/healthz` -- is the process alive? It MUST NOT touch Postgres or Redis. If
  it did, a transient Redis blip would fail the liveness probe and Kubernetes
  would restart every healthy pod in the fleet, turning a small outage into a
  large one.
* `/readyz` -- can this process serve traffic? It DOES check dependencies and
  returns 503 when they are down, so the pod is pulled from the load balancer
  but left running to recover.
"""

from __future__ import annotations

import asyncio
from http import HTTPStatus

from fastapi import APIRouter, Response

from aiswe.api.deps import ContainerDep, SettingsDep
from aiswe.api.schemas.common import (
    DependencyStatus,
    HealthResponse,
    ReadinessResponse,
)

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz(settings: SettingsDep) -> HealthResponse:
    """Process is alive. No dependency checks. Must never fail on a hot process."""
    return HealthResponse(
        service=settings.app.name,
        version=settings.app.version,
        environment=str(settings.app.env),
    )


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(container: ContainerDep, response: Response) -> ReadinessResponse:
    """Every dependency is reachable. 503 if not."""
    checks = container.health_checks()

    # Probe in parallel: a serial check would take as long as the slowest
    # dependency's timeout multiplied by the number of dependencies.
    results = await asyncio.gather(
        *(check.ping() for check in checks.values()),
        return_exceptions=True,
    )

    dependencies = [
        DependencyStatus(name=name, healthy=(result is True))
        for name, result in zip(checks.keys(), results, strict=True)
    ]
    all_healthy = all(dep.healthy for dep in dependencies)

    if not all_healthy:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if all_healthy else "degraded",
        dependencies=dependencies,
    )
