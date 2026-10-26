"""Redis event bus.

Two Redis primitives, two jobs:

    STREAM  (run:{id}:events)  durable, ordered, replayable. Backs reconnect.
    PUBSUB  (run:{id}:live)    fan-out to attached subscribers. Backs live SSE.

THE SUBTLE PART: `stream()`
---------------------------
The naive implementation replays the log, then subscribes. It has a hole: any
event published in the window between "replay finished" and "subscription
active" is lost forever. On a busy run that window is small but it is not zero,
and the symptom -- an SSE stream that silently skips an event on reconnect -- is
close to impossible to reproduce.

So: SUBSCRIBE FIRST, then replay, then de-duplicate on `seq`. Overlap is
harmless (we drop the duplicates); a gap is not (we cannot invent the missing
event). Prefer the failure you can fix.

PUBLISH MUST NOT RAISE
----------------------
A browser that closed its tab is not a reason to fail a 20-minute agent run.
Publish failures are logged and swallowed. The durable write is attempted first,
so even a total pub/sub outage costs only live-ness, not history.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from redis.exceptions import RedisError

from aiswe.config.constants import RUN_EVENTS_NS
from aiswe.domain.events.domain_events import DomainEvent, EventType
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)

#: Events live for 24h after a run ends. Long enough for a user to come back
#: from lunch and reconnect; short enough not to grow without bound. The
#: permanent record lives in Postgres (`run_events`), not here.
_STREAM_TTL_SECONDS = 86_400

#: Hard cap per run. A pathological run cannot exhaust Redis.
_MAX_STREAM_LEN = 10_000


class RedisEventBus:
    """Implements `domain.ports.event_bus.EventBus`."""

    def __init__(self, client: Redis) -> None:
        self._redis = client

    # --- Keys ----------------------------------------------------------------

    def _stream_key(self, run_id: uuid.UUID) -> str:
        return f"{RUN_EVENTS_NS}:{run_id}:events"

    def _channel(self, run_id: uuid.UUID) -> str:
        return f"{RUN_EVENTS_NS}:{run_id}:live"

    def _seq_key(self, run_id: uuid.UUID) -> str:
        return f"{RUN_EVENTS_NS}:{run_id}:seq"

    # --- Publish -------------------------------------------------------------

    async def publish(self, event: DomainEvent) -> int:
        """Append to the durable log, then fan out. Never raises."""
        try:
            # A single atomic INCR is the sequence source. Deriving `seq` from
            # the stream's own entry id would tie the wire format to Redis's
            # internal id scheme -- and break the moment we mirror to Postgres.
            seq = int(await self._redis.incr(self._seq_key(event.run_id)))
            sequenced = _with_seq(event, seq)
            payload = json.dumps(sequenced.to_dict())

            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.xadd(
                    self._stream_key(event.run_id),
                    {"data": payload},
                    maxlen=_MAX_STREAM_LEN,
                    approximate=True,
                )
                pipe.expire(self._stream_key(event.run_id), _STREAM_TTL_SECONDS)
                pipe.expire(self._seq_key(event.run_id), _STREAM_TTL_SECONDS)
                pipe.publish(self._channel(event.run_id), payload)
                await pipe.execute()

        except RedisError as exc:
            # Deliberately swallowed. See the module docstring: a dropped event
            # must never take down a run that is otherwise succeeding.
            logger.warning(
                "event.publish_failed",
                event_type=str(event.type),
                run_id=str(event.run_id),
                error=str(exc),
            )
            return -1

        return seq

    # --- Replay --------------------------------------------------------------

    async def replay(self, run_id: uuid.UUID, *, from_seq: int = 0) -> list[DomainEvent]:
        """Every event with seq > from_seq, in order."""
        try:
            entries = await self._redis.xrange(self._stream_key(run_id), min="-", max="+")
        except RedisError as exc:
            logger.warning("event.replay_failed", run_id=str(run_id), error=str(exc))
            return []

        out: list[DomainEvent] = []
        for _entry_id, fields in entries:
            event = _decode(fields.get("data"))
            if event and (event.seq or 0) > from_seq:
                out.append(event)
        return out

    # --- Live ----------------------------------------------------------------

    async def subscribe(self, run_id: uuid.UUID) -> AsyncIterator[DomainEvent]:
        """Live events only, from this moment on."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel(run_id))
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                event = _decode(message.get("data"))
                if event is not None:
                    yield event
        finally:
            await pubsub.unsubscribe(self._channel(run_id))
            await pubsub.aclose()

    async def stream(
        self,
        run_id: uuid.UUID,
        *,
        from_seq: int = 0,
    ) -> AsyncIterator[DomainEvent]:
        """Replay-then-live, with no gap and no duplicates.

        Order of operations is the whole point -- see the module docstring.
        """
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._channel(run_id))  # (1) subscribe FIRST

        try:
            seen: set[int] = set()

            # (2) Then replay history. Anything published during the replay is
            #     already buffered on the subscription, so nothing is lost.
            for event in await self.replay(run_id, from_seq=from_seq):
                if event.seq is not None:
                    seen.add(event.seq)
                yield event

            # (3) Drain the buffered live stream, discarding anything the replay
            #     already delivered.
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                event = _decode(message.get("data"))
                if event is None:
                    continue
                if event.seq is not None and event.seq in seen:
                    continue  # overlap: harmless, dropped
                if event.seq is not None:
                    seen.add(event.seq)

                yield event

                # Close the stream cleanly once the run ends, so an SSE
                # connection does not hang open forever on a finished run.
                if event.type in _TERMINAL_EVENTS:
                    return
        finally:
            await pubsub.unsubscribe(self._channel(run_id))
            await pubsub.aclose()

    async def clear(self, run_id: uuid.UUID) -> None:
        """Drop a run's live event data. The Postgres record is the archive."""
        await self._redis.delete(self._stream_key(run_id), self._seq_key(run_id))


_TERMINAL_EVENTS = frozenset(
    {EventType.RUN_FINISHED, EventType.RUN_FAILED, EventType.RUN_CANCELLED}
)


def _with_seq(event: DomainEvent, seq: int) -> DomainEvent:
    """Frozen dataclass: rebuild rather than mutate."""
    return DomainEvent(
        type=event.type,
        run_id=event.run_id,
        payload=event.payload,
        event_id=event.event_id,
        occurred_at=event.occurred_at,
        seq=seq,
        agent=event.agent,
    )


def _decode(raw: Any) -> DomainEvent | None:
    """Parse a wire event. A malformed event is dropped, never fatal."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return DomainEvent(
            type=EventType(data["type"]),
            run_id=uuid.UUID(data["run_id"]),
            payload=data.get("payload", {}),
            event_id=uuid.UUID(data["event_id"]),
            occurred_at=datetime.fromisoformat(data["occurred_at"]),
            seq=data.get("seq"),
            agent=data.get("agent"),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("event.decode_failed", error=str(exc))
        return None
