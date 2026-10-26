"""Application configuration.

One typed object, validated once, at startup. If configuration is wrong the
process refuses to boot -- we never discover a bad DSN on the first request.

Nesting is expressed with a double-underscore delimiter:

    DB__POOL_SIZE=20   ->   settings.db.pool_size

`get_settings()` is cached, so the object is a process-wide singleton. It is
injected through the DI container, never imported ad-hoc inside business logic
(that would be a hidden global dependency and would break testability).
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


def _config(prefix: str = "") -> SettingsConfigDict:
    """Shared settings behaviour, plus a per-section env prefix.

    `frozen=True` matters: configuration that can be mutated at runtime is a
    whole category of bug (a test changes a value, a later test inherits it).
    """
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix=prefix,
        extra="ignore",
        frozen=True,
    )


class AppSettings(BaseSettings):
    model_config = _config("APP__")

    env: Environment = Environment.LOCAL
    debug: bool = False
    name: str = "ai-swe-agent"
    version: str = "0.1.0"
    # Binding 0.0.0.0 is correct inside a container; the network, not the
    # process, controls exposure.
    host: str = "0.0.0.0"  # noqa: S104
    port: int = Field(default=8000, ge=1, le=65535)
    root_path: str = ""
    cors_origins: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.env is Environment.PRODUCTION


class LogSettings(BaseSettings):
    model_config = _config("LOG__")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    redact_keys: list[str] = Field(
        default_factory=lambda: [
            "password",
            "token",
            "api_key",
            "secret",
            "authorization",
            "credentials_ref",
        ]
    )


class DatabaseSettings(BaseSettings):
    model_config = _config("DB__")

    dsn: str = "postgresql+asyncpg://aiswe:aiswe@localhost:5432/aiswe"
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=200)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=1800, ge=60)
    echo: bool = False
    statement_timeout_ms: int = Field(default=30_000, ge=1_000)

    @field_validator("dsn")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        """A sync driver would block the event loop under load. Refuse it here,
        loudly, rather than debug mysterious latency in production."""
        if not v.startswith("postgresql+asyncpg://"):
            msg = "DB__DSN must use the asyncpg driver: postgresql+asyncpg://..."
            raise ValueError(msg)
        return v


class RedisSettings(BaseSettings):
    model_config = _config("REDIS__")

    dsn: str = "redis://localhost:6379/0"
    max_connections: int = Field(default=50, ge=1)
    socket_timeout: int = Field(default=5, ge=1)
    default_ttl: int = Field(default=3600, ge=1)


class SecuritySettings(BaseSettings):
    """Declared at M0, consumed at M1. Keys stay optional until then."""

    model_config = _config("SECURITY__")

    jwt_algorithm: Literal["RS256", "HS256"] = "RS256"
    jwt_private_key_path: Path | None = None
    jwt_public_key_path: Path | None = None
    jwt_secret: SecretStr | None = None  # HS256 / local dev only
    access_token_ttl_seconds: int = Field(default=900, ge=60)
    refresh_token_ttl_seconds: int = Field(default=604_800, ge=3600)
    api_key_prefix: str = "ak_"


class OtelSettings(BaseSettings):
    model_config = _config("OTEL__")

    enabled: bool = False
    service_name: str = "ai-swe-agent"
    exporter_endpoint: str = "http://localhost:4317"


class Settings(BaseSettings):
    """Root configuration. The only object the composition root reads."""

    model_config = _config()

    app: AppSettings = Field(default_factory=AppSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    otel: OtelSettings = Field(default_factory=OtelSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so `.env` is parsed exactly once."""
    return Settings()
