"""Async database engine and session lifecycle.

Owned by the DI container, created once at startup, disposed at shutdown.
Nothing in the application layer imports this module -- it receives an
`AbstractUnitOfWork` instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

from aiswe.domain.errors import InfrastructureError
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.config.settings import DatabaseSettings

logger = get_logger(__name__)


class Database:
    """Owns the engine and the session factory. Implements the HealthCheck port."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            settings.dsn,
            echo=settings.echo,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            # Detect connections killed by a proxy/failover before handing them out.
            pool_pre_ping=True,
            connect_args={
                "timeout": settings.pool_timeout,
                "server_settings": {
                    # A runaway query must never pin a connection forever.
                    "statement_timeout": str(settings.statement_timeout_ms),
                    "application_name": "aiswe",
                },
            },
        )
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            # Critical for async: with expire_on_commit=True, touching any
            # attribute after commit triggers an implicit lazy refresh, which
            # raises MissingGreenlet outside a session context.
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        return self._sessionmaker

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session. Used by the UnitOfWork; not by application code."""
        session = self._sessionmaker()
        try:
            yield session
        finally:
            await session.close()

    async def ping(self) -> bool:
        """Readiness probe. Never raises -- returns False on any failure."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.warning("db.ping.failed", error=str(exc))
            return False
        return True

    async def connect(self) -> None:
        """Verify connectivity at startup. Fail loudly rather than at first request."""
        if not await self.ping():
            msg = "Database is unreachable at startup."
            raise InfrastructureError(msg, code="db_unavailable")
        logger.info("db.connected", pool_size=self._settings.pool_size)

    async def disconnect(self) -> None:
        """Drain the pool on shutdown."""
        await self._engine.dispose()
        logger.info("db.disconnected")
