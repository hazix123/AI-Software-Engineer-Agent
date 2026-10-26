"""Graph checkpointer.

Durable LangGraph state, keyed by `thread_id` (which we set to the run id).

WHY THIS IS NOT OPTIONAL
------------------------
An agent run is long, expensive, and stateful. A run that reaches step 30 of 40
and then loses its worker -- OOM, deploy, spot reclaim -- must resume at step 30.

Without a checkpointer it restarts at step 1, and re-pays for all 29 steps. With
one, another worker reclaims the lease, loads the last committed node's state,
and carries on. Combined with the task queue's lease mechanism (see
`ports/task_queue.py`), this is the entire crash-recovery story.

The checkpoint schema is owned by LangGraph and lives in its own tables. We do
not autogenerate migrations against it -- `.setup()` manages it -- which is why
`migrations/env.py` excludes it from autogenerate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiswe.domain.errors import ConfigurationError
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = get_logger(__name__)


async def build_checkpointer(dsn: str) -> BaseCheckpointSaver[Any]:
    """Create and initialise the Postgres checkpointer.

    Args:
        dsn: The application's asyncpg DSN. Translated to the psycopg form the
            LangGraph saver expects -- derived, never configured separately, so
            the two cannot drift.
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:  # pragma: no cover
        msg = (
            "langgraph-checkpoint-postgres is not installed. "
            "Install the `agents` extra: `uv sync --extra agents`."
        )
        raise ConfigurationError(msg) from exc

    saver_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)

    saver_cm = AsyncPostgresSaver.from_conn_string(saver_dsn)
    saver = await saver_cm.__aenter__()

    # Idempotent. Creates the checkpoint tables on first run.
    await saver.setup()

    logger.info("checkpointer.ready", backend="postgres")
    return saver  # type: ignore[no-any-return]


def build_memory_checkpointer() -> BaseCheckpointSaver[Any]:
    """In-process checkpointer, for tests and local experiments.

    NEVER use this in a deployed worker: the checkpoint dies with the process,
    which defeats the entire purpose of having one.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    logger.warning("checkpointer.in_memory", note="not durable; tests only")
    return InMemorySaver()
