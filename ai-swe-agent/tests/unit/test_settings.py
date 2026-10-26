"""Configuration unit tests.

Config is typed precisely so that misconfiguration is a startup failure rather
than a 3am incident. These tests assert exactly that.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiswe.config.settings import (
    AppSettings,
    DatabaseSettings,
    Environment,
    RedisSettings,
    Settings,
)

pytestmark = pytest.mark.unit


class TestDatabaseSettings:
    def test_rejects_sync_driver(self) -> None:
        """A psycopg DSN would silently block the event loop. Refuse it."""
        with pytest.raises(ValidationError, match="asyncpg"):
            DatabaseSettings(dsn="postgresql://user:pass@localhost/db")

    def test_accepts_async_driver(self) -> None:
        settings = DatabaseSettings(dsn="postgresql+asyncpg://u:p@localhost:5432/db")
        assert settings.dsn.startswith("postgresql+asyncpg://")

    @pytest.mark.parametrize("pool_size", [0, -1, 101])
    def test_pool_size_bounds(self, pool_size: int) -> None:
        with pytest.raises(ValidationError):
            DatabaseSettings(pool_size=pool_size)


class TestAppSettings:
    def test_is_production_flag(self) -> None:
        assert AppSettings(env=Environment.PRODUCTION).is_production is True
        assert AppSettings(env=Environment.LOCAL).is_production is False

    def test_settings_are_frozen(self) -> None:
        """Config must not mutate at runtime -- that is a whole class of bugs."""
        settings = AppSettings()
        with pytest.raises(ValidationError):
            settings.port = 9999  # type: ignore[misc]


class TestSettingsComposition:
    def test_defaults_compose(self) -> None:
        settings = Settings()
        assert settings.app.name == "ai-swe-agent"
        assert settings.redis.default_ttl > 0

    def test_nested_override(self) -> None:
        settings = Settings(redis=RedisSettings(max_connections=99))
        assert settings.redis.max_connections == 99
