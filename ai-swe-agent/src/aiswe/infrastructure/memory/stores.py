"""Memory adapters.

    RedisScratchpad  -- SHORT-TERM. Real, working, shipped now.
    NullMemoryStore  -- LONG-TERM.  Placeholder until M8.

WHY A NULL OBJECT RATHER THAN `if self._store is not None:`
-----------------------------------------------------------
Optional dependencies rot. Every call site grows a None-check, the checks drift
apart, and one of them eventually forgets -- producing an AttributeError in
production on the one path nobody exercised.

The Null Object pattern removes the question. `MemoryAgent` calls `recall()` and
`remember()` unconditionally, today, against the real interface. When the pgvector
store lands in M8, one line changes in the container and every call site is
already correct because it was never conditional.

The null store LOGS what it would have written, so the memory pipeline is
observable end-to-end long before it is functional.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

from aiswe.domain.ports.memory import MemoryQuery, MemoryRecord
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

_SCRATCH_NS = "scratch"
_DEFAULT_TTL = 86_400  # 24h: longer than any sane run, short enough to self-clean


class RedisScratchpad:
    """Implements `domain.ports.memory.ScratchpadStore`.

    Short-term memory: run-scoped, TTL-bound key/value storage for payloads too
    large for the graph state but too transient for the database.

    TTL is not optional. A run that crashes without calling `clear()` would
    otherwise leak its scratchpad forever, and enough crashed runs fill Redis.
    """

    def __init__(self, client: Redis, *, ttl: int = _DEFAULT_TTL) -> None:
        self._redis = client
        self._ttl = ttl

    def _key(self, run_id: uuid.UUID, key: str) -> str:
        return f"{_SCRATCH_NS}:{run_id}:{key}"

    def _pattern(self, run_id: uuid.UUID) -> str:
        return f"{_SCRATCH_NS}:{run_id}:*"

    async def get(self, run_id: uuid.UUID, key: str) -> str | None:
        try:
            value: str | None = await self._redis.get(self._key(run_id, key))
        except RedisError as exc:
            # A scratchpad miss must degrade, not fail: it is a cache, and the
            # agent can always recompute.
            logger.warning("scratchpad.get_failed", key=key, error=str(exc))
            return None
        return value

    async def set(
        self,
        run_id: uuid.UUID,
        key: str,
        value: str,
        *,
        ttl: int | None = None,
    ) -> None:
        try:
            await self._redis.set(self._key(run_id, key), value, ex=ttl or self._ttl)
        except RedisError as exc:
            logger.warning("scratchpad.set_failed", key=key, error=str(exc))

    async def delete(self, run_id: uuid.UUID, key: str) -> None:
        await self._redis.delete(self._key(run_id, key))

    async def keys(self, run_id: uuid.UUID) -> list[str]:
        """SCAN, never KEYS: KEYS blocks the whole Redis event loop, and it will
        do so at precisely the moment the system is busiest."""
        prefix = f"{_SCRATCH_NS}:{run_id}:"
        found: list[str] = []
        async for key in self._redis.scan_iter(match=self._pattern(run_id), count=100):
            found.append(str(key).removeprefix(prefix))
        return found

    async def clear(self, run_id: uuid.UUID) -> None:
        """Drop the whole scratchpad. Called when a run reaches a terminal state."""
        batch: list[str] = []
        async for key in self._redis.scan_iter(match=self._pattern(run_id), count=100):
            batch.append(str(key))
            if len(batch) >= 100:
                await self._redis.delete(*batch)
                batch.clear()
        if batch:
            await self._redis.delete(*batch)
        logger.debug("scratchpad.cleared", run_id=str(run_id))


class NullMemoryStore:
    """Implements `domain.ports.memory.MemoryStore`. A no-op until M8.

    Recall returns nothing; the system simply starts every run cold, which is
    correct behaviour, just not clever behaviour.

    Writes are LOGGED rather than discarded silently. That means the whole memory
    pipeline -- extraction, the procedural-promotion guard, dedupe candidates --
    is observable and reviewable today, and M8 becomes "persist what we are
    already computing" rather than "build memory from scratch".
    """

    def __init__(self) -> None:
        logger.info(
            "memory.null_store",
            note="long-term memory is a no-op until M8; runs start cold",
        )

    async def recall(self, query: MemoryQuery) -> list[MemoryRecord]:
        logger.debug("memory.recall.noop", query=query.text[:80])
        return []

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        # Logged, not stored. This is the observable seam.
        logger.info(
            "memory.remember.noop",
            type=str(record.type),
            scope=str(record.scope),
            confidence=record.confidence,
            content=record.content[:160],
            note="would persist in M8",
        )
        return record

    async def forget(self, memory_id: uuid.UUID) -> None:
        logger.debug("memory.forget.noop", memory_id=str(memory_id))

    async def reinforce(self, memory_id: uuid.UUID) -> None:
        logger.debug("memory.reinforce.noop", memory_id=str(memory_id))
