"""Composition root.

The ONE file permitted to know about both abstractions and their concrete
implementations. Everything else depends on ports.

Deliberately hand-rolled rather than using a DI framework:

* Fully type-checked by mypy strict -- a framework's runtime wiring is not.
* No service locator, no global mutable state. Nothing "reaches out" for a
  dependency; dependencies are handed in.
* Startup order is explicit and readable, which matters when a boot failure
  wakes someone up.

Lifecycle is owned by the FastAPI lifespan AND by the worker entrypoint, so both
processes share exactly one wiring path and cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiswe.domain.errors import ConfigurationError
from aiswe.domain.ports.cache import Cache, HealthCheck, LockManager
from aiswe.domain.ports.clock import Clock, SystemClock
from aiswe.domain.ports.event_bus import EventBus
from aiswe.domain.ports.llm import LLMProvider
from aiswe.domain.ports.memory import MemoryStore, ScratchpadStore
from aiswe.domain.ports.tool import ToolRegistry
from aiswe.infrastructure.db.session import Database
from aiswe.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from aiswe.infrastructure.llm.stub_provider import StubLLMProvider
from aiswe.infrastructure.memory.stores import NullMemoryStore, RedisScratchpad
from aiswe.infrastructure.observability.logging import configure_logging, get_logger
from aiswe.infrastructure.redis.cache import RedisCache, RedisLockManager
from aiswe.infrastructure.redis.client import RedisClient
from aiswe.infrastructure.redis.event_bus import RedisEventBus
from aiswe.infrastructure.redis.task_queue import RedisTaskQueue
from aiswe.tools.registry import InMemoryToolRegistry

if TYPE_CHECKING:
    from aiswe.config.settings import Settings

logger = get_logger(__name__)


@dataclass(slots=True)
class Container:
    """Every wired dependency, for the process lifetime.

    Note the field TYPES: `cache` is the `Cache` Protocol, not `RedisCache`;
    `llm` is `LLMProvider`, not `StubLLMProvider`. Consumers cannot reach for an
    adapter-specific method even by accident, and swapping an adapter is a
    one-line change in `build()`.
    """

    settings: Settings

    # --- Infrastructure owners (concrete: they manage lifecycle) -------------
    database: Database
    redis_client: RedisClient

    # --- Ports (abstract: what the rest of the system consumes) --------------
    cache: Cache
    locks: LockManager
    clock: Clock

    # --- Agent framework -----------------------------------------------------
    event_bus: EventBus
    task_queue: RedisTaskQueue  # concrete: the worker calls ensure_group()
    llm: LLMProvider
    tools: ToolRegistry
    memory_store: MemoryStore
    scratchpad: ScratchpadStore

    # --- Lifecycle -----------------------------------------------------------

    @classmethod
    def build(cls, settings: Settings) -> Container:
        """Construct the object graph. No I/O here -- see `startup()`."""
        configure_logging(settings.log)

        database = Database(settings.db)
        redis_client = RedisClient(settings.redis)
        redis = redis_client.client

        return cls(
            settings=settings,
            database=database,
            redis_client=redis_client,
            cache=RedisCache(redis, default_ttl=settings.redis.default_ttl),
            locks=RedisLockManager(redis),
            clock=SystemClock(),
            event_bus=RedisEventBus(redis),
            task_queue=RedisTaskQueue(redis),
            # M2 replaces this with ModelRouter(anthropic, openai, gemini).
            # The guard in startup() refuses to let the stub reach production.
            llm=StubLLMProvider(),
            # Empty until M3 registers filesystem/git/shell/test tools. The
            # framework, the policy engine, and the middleware chain are all
            # live now -- only the tools themselves are missing.
            tools=InMemoryToolRegistry(),
            # M8 replaces this with PgVectorMemoryStore. Every call site is
            # already written against the real interface.
            memory_store=NullMemoryStore(),
            scratchpad=RedisScratchpad(redis),
        )

    async def startup(self) -> None:
        """Open connections. Raises if any dependency is unreachable.

        Failing here is the point: a process that cannot reach its database
        should never accept traffic.
        """
        logger.info(
            "container.starting",
            env=self.settings.app.env,
            version=self.settings.app.version,
        )

        self._guard_production_config()

        await self.database.connect()
        await self.redis_client.connect()
        logger.info("container.started")

    def _guard_production_config(self) -> None:
        """Refuse to boot production with development stand-ins wired in.

        A deployed system that silently answers with a STUB model is far worse
        than one that refuses to start: the failure is invisible, the output is
        confidently wrong, and nobody notices until a customer does.
        """
        if not self.settings.app.is_production:
            return

        if isinstance(self.llm, StubLLMProvider):
            msg = (
                "StubLLMProvider is wired in a production environment. "
                "Configure a real provider (LLM__ANTHROPIC_API_KEY, ...) before deploying."
            )
            raise ConfigurationError(msg)

    async def shutdown(self) -> None:
        """Drain connections. Best-effort: shutdown must not raise."""
        await self.redis_client.disconnect()
        await self.database.disconnect()
        logger.info("container.stopped")

    # --- Factories -----------------------------------------------------------

    def unit_of_work(self) -> SqlAlchemyUnitOfWork:
        """A fresh UoW per use case. Never share one across requests."""
        return SqlAlchemyUnitOfWork(self.database.sessionmaker)

    def health_checks(self) -> dict[str, HealthCheck]:
        """Named readiness probes, consumed by `/readyz`."""
        return {"database": self.database, "redis": self.redis_client}
