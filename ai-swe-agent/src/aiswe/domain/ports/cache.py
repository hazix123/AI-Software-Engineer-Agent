"""Cache and distributed-lock ports.

Application and agent code depends on these Protocols, never on `redis`
directly. The Redis adapter lives in `infrastructure/redis/`.

Note the Interface Segregation: `Cache` and `LockManager` are separate roles.
A consumer that only needs read-through caching should not be handed the
ability to take cluster-wide locks.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Key/value cache with TTL semantics. Values are opaque bytes-as-str."""

    async def get(self, key: str) -> str | None:
        """Return the cached value, or None on miss."""
        ...

    async def set(self, key: str, value: str, *, ttl: int | None = None) -> None:
        """Store a value. `ttl` in seconds; None uses the adapter default."""
        ...

    async def delete(self, *keys: str) -> int:
        """Remove keys. Returns the number actually deleted."""
        ...

    async def exists(self, key: str) -> bool:
        """Cheap existence check."""
        ...

    async def incr(self, key: str, *, amount: int = 1, ttl: int | None = None) -> int:
        """Atomically increment a counter. Used by rate limiting (M1)."""
        ...


@runtime_checkable
class LockManager(Protocol):
    """Distributed mutual exclusion.

    From M3 this guards workspace mutation: only one writer agent may hold a
    given workspace at a time.
    """

    def lock(
        self,
        name: str,
        *,
        timeout: float = 30.0,
        blocking_timeout: float | None = 10.0,
    ) -> AbstractAsyncContextManager[None]:
        """Acquire a named lock as an async context manager.

        Raises:
            InfrastructureError: if the lock cannot be acquired in time.
        """
        ...


@runtime_checkable
class HealthCheck(Protocol):
    """Anything that can report readiness. Implemented by db and cache adapters."""

    async def ping(self) -> bool:
        """Return True if the dependency is reachable and healthy."""
        ...
