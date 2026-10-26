"""Shared pytest fixtures.

The app under test is built by the SAME `create_app()` that production uses.
There is no test-only wiring path -- if there were, the suite would slowly stop
proving anything about the real system.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from aiswe.api.app import create_app
from aiswe.config.settings import (
    AppSettings,
    DatabaseSettings,
    Environment,
    LogSettings,
    RedisSettings,
    Settings,
)


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Test settings. Points at the services CI (or docker compose) provides."""
    return Settings(
        app=AppSettings(env=Environment.DEV, debug=True, name="aiswe-test"),
        log=LogSettings(level="WARNING", format="console"),
        db=DatabaseSettings(
            dsn=os.getenv(
                "DB__DSN",
                "postgresql+asyncpg://aiswe:aiswe@localhost:5432/aiswe_test",
            ),
            pool_size=2,
            max_overflow=0,
        ),
        redis=RedisSettings(dsn=os.getenv("REDIS__DSN", "redis://localhost:6379/1")),
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    """In-process ASGI client.

    `ASGITransport` runs the app without binding a socket: the full middleware
    and routing stack executes, but there is no network in the loop -- so tests
    are fast and cannot flake on a port.

    The `async with app.router.lifespan_context` is essential: without it the DI
    container is never built and `app.state.container` does not exist.
    """
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
