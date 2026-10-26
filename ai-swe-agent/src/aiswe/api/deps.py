"""FastAPI dependency providers.

Thin by design. FastAPI's `Depends` is a *transport-layer* mechanism; it must
not become the application's DI system, or every use case ends up importing
FastAPI and the dependency rule collapses.

So: the real object graph lives in `config/container.py`. These functions just
hand pieces of it to routes, typed as PORTS (`Cache`, not `RedisCache`).

IMPORTANT -- why the imports below are NOT under `TYPE_CHECKING`:

FastAPI resolves every endpoint's annotations at import time via
`typing.get_type_hints()`. Combined with `from __future__ import annotations`,
a name that only exists under `TYPE_CHECKING` becomes an unresolvable
ForwardRef and the app raises NameError on the first request. Dependency
annotations must therefore be resolvable at RUNTIME. This is the one place the
usual "push imports into TYPE_CHECKING" habit is actively wrong, so `TCH` is
disabled for this file in pyproject.toml.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request

from aiswe.config.container import Container
from aiswe.config.settings import Settings
from aiswe.domain.ports.cache import Cache, LockManager
from aiswe.domain.ports.clock import Clock
from aiswe.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork


def get_container(request: Request) -> Container:
    """The container is attached to `app.state` by the lifespan handler."""
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings_dep(container: ContainerDep) -> Settings:
    return container.settings


def get_cache(container: ContainerDep) -> Cache:
    return container.cache


def get_locks(container: ContainerDep) -> LockManager:
    return container.locks


def get_clock(container: ContainerDep) -> Clock:
    return container.clock


async def get_uow(container: ContainerDep) -> AsyncIterator[SqlAlchemyUnitOfWork]:
    """One UnitOfWork per request, disposed when the request ends.

    Deliberately NOT committed here. The use case decides whether its work
    should be persisted; an implicit commit-on-success would make partial
    writes invisible and turn a bug into corrupted data.
    """
    async with container.unit_of_work() as uow:
        yield uow


# Typed aliases. A route writes `cache: CacheDep` and receives a PORT, so it
# cannot reach for a Redis-specific method even by accident.
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
CacheDep = Annotated[Cache, Depends(get_cache)]
LockDep = Annotated[LockManager, Depends(get_locks)]
ClockDep = Annotated[Clock, Depends(get_clock)]
UowDep = Annotated[SqlAlchemyUnitOfWork, Depends(get_uow)]

# Added in M1:
#   CurrentUserDep = Annotated[AuthenticatedUser, Depends(get_current_user)]
#   RequireScope("run:create") -- a dependency factory enforcing RBAC scopes.
