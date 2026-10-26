"""Redis connection pool.

Redis serves four distinct roles in this system (cache, distributed locks,
rate limiting, and -- from M5 -- the run event bus and task queue). They all
share one pool but are namespaced by key prefix (see `config/constants.py`) so
a shared instance stays debuggable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError

from aiswe.domain.errors import InfrastructureError
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.config.settings import RedisSettings

logger = get_logger(__name__)


class RedisClient:
    """Owns the pool. Implements the HealthCheck port."""

    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings
        self._pool: ConnectionPool = ConnectionPool.from_url(
            settings.dsn,
            max_connections=settings.max_connections,
            socket_timeout=settings.socket_timeout,
            socket_connect_timeout=settings.socket_timeout,
            # Reconnect transparently if a proxy drops an idle connection.
            health_check_interval=30,
            # Everything we store is UTF-8 text (JSON); decoding here keeps the
            # adapters free of bytes/str conversion noise.
            decode_responses=True,
        )
        self._client: Redis = Redis(connection_pool=self._pool)

    @property
    def client(self) -> Redis:
        return self._client

    async def ping(self) -> bool:
        """Readiness probe. Never raises."""
        try:
            return bool(await self._client.ping())
        except RedisError as exc:
            logger.warning("redis.ping.failed", error=str(exc))
            return False

    async def connect(self) -> None:
        """Verify connectivity at startup."""
        if not await self.ping():
            msg = "Redis is unreachable at startup."
            raise InfrastructureError(msg, code="redis_unavailable")
        logger.info("redis.connected", max_connections=self._settings.max_connections)

    async def disconnect(self) -> None:
        await self._client.aclose()
        await self._pool.aclose()
        logger.info("redis.disconnected")
