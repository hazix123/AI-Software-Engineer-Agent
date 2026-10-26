"""Redis adapters for the Cache and LockManager ports.

These classes are the ONLY place in the codebase that knows Redis exists.
Everything upstream depends on `aiswe.domain.ports.cache.Cache` /
`LockManager`, which is what makes the dependency inversion real rather than
decorative.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

from redis.exceptions import LockError, RedisError

from aiswe.config.constants import CACHE_NS, LOCK_NS
from aiswe.domain.errors import InfrastructureError
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


class RedisCache:
    """Implements `domain.ports.cache.Cache`."""

    def __init__(
        self,
        client: Redis,
        *,
        default_ttl: int = 3600,
        namespace: str = CACHE_NS,
    ) -> None:
        self._client = client
        self._default_ttl = default_ttl
        self._ns = namespace

    def _key(self, key: str) -> str:
        return f"{self._ns}:{key}"

    async def get(self, key: str) -> str | None:
        try:
            value: str | None = await self._client.get(self._key(key))
        except RedisError as exc:
            # A cache miss is always survivable; a cache outage should degrade,
            # not take the request down. Callers treat None as "not cached".
            logger.warning("cache.get.failed", key=key, error=str(exc))
            return None
        return value

    async def set(self, key: str, value: str, *, ttl: int | None = None) -> None:
        try:
            await self._client.set(self._key(key), value, ex=ttl or self._default_ttl)
        except RedisError as exc:
            logger.warning("cache.set.failed", key=key, error=str(exc))

    async def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        try:
            deleted: int = await self._client.delete(*(self._key(k) for k in keys))
        except RedisError as exc:
            logger.warning("cache.delete.failed", error=str(exc))
            return 0
        return deleted

    async def exists(self, key: str) -> bool:
        try:
            return bool(await self._client.exists(self._key(key)))
        except RedisError as exc:
            logger.warning("cache.exists.failed", key=key, error=str(exc))
            return False

    async def incr(self, key: str, *, amount: int = 1, ttl: int | None = None) -> int:
        """Atomic counter. Backs rate limiting and budget metering.

        Unlike get/set, a failure here MUST raise: silently returning 0 would
        let a caller bypass a rate limit or a spend cap.
        """
        full_key = self._key(key)
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incrby(full_key, amount)
                pipe.expire(full_key, ttl or self._default_ttl, nx=True)
                results = await pipe.execute()
        except RedisError as exc:
            logger.exception("cache.incr.failed", key=key)
            msg = "Counter increment failed."
            raise InfrastructureError(msg, code="cache_incr_failed") from exc
        return int(results[0])


class RedisLockManager:
    """Implements `domain.ports.cache.LockManager`.

    From M3 this serialises workspace mutation: only one writer agent may hold
    a given workspace at a time. `timeout` is the lock's own TTL -- a crashed
    holder must not deadlock the workspace forever.
    """

    def __init__(self, client: Redis, *, namespace: str = LOCK_NS) -> None:
        self._client = client
        self._ns = namespace

    def lock(
        self,
        name: str,
        *,
        timeout: float = 30.0,
        blocking_timeout: float | None = 10.0,
    ) -> AbstractAsyncContextManager[None]:
        return self._lock(name, timeout=timeout, blocking_timeout=blocking_timeout)

    @asynccontextmanager
    async def _lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float | None,
    ) -> AsyncIterator[None]:
        key = f"{self._ns}:{name}"
        redis_lock = self._client.lock(
            key,
            timeout=timeout,
            blocking=blocking_timeout is not None,
            blocking_timeout=blocking_timeout,
        )
        acquired = False
        try:
            acquired = await redis_lock.acquire()
            if not acquired:
                msg = f"Could not acquire lock {name!r} within {blocking_timeout}s."
                raise InfrastructureError(msg, code="lock_timeout", details={"lock": name})
            logger.debug("lock.acquired", lock=name)
            yield
        finally:
            if acquired:
                try:
                    await redis_lock.release()
                    logger.debug("lock.released", lock=name)
                except LockError:  # pragma: no cover - lock already expired
                    logger.warning("lock.release_failed", lock=name)
