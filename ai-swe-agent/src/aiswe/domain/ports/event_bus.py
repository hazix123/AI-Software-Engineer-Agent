"""Event bus port.

`replay(from_seq)` is the requirement that shapes this interface.

A user watches a 20-minute agent run over SSE. Their laptop sleeps. They
reconnect. Pure pub/sub has already dropped everything they missed, and the run
appears to have skipped ten minutes of work.

So the bus is pub/sub for LIVE delivery plus a durable, monotonically sequenced
log for REPLAY. On reconnect the client sends `Last-Event-ID: <seq>` and the
transport replays from the log, then attaches to the live stream. Sequence
numbers are assigned by the bus, because only the bus can guarantee
monotonicity per run.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiswe.domain.events.domain_events import DomainEvent


@runtime_checkable
class EventBus(Protocol):
    """Fan-out of run events to live subscribers and to durable storage."""

    async def publish(self, event: DomainEvent) -> int:
        """Append to the durable log, then fan out to live subscribers.

        Returns:
            The sequence number assigned to this event, monotonic per run.

        Publishing MUST NOT raise on a subscriber failure: a browser that
        disconnected is not a reason to fail an agent run.
        """
        ...

    def subscribe(self, run_id: uuid.UUID) -> AsyncIterator[DomainEvent]:
        """Live events for one run, from the moment of subscription."""
        ...

    async def replay(self, run_id: uuid.UUID, *, from_seq: int = 0) -> list[DomainEvent]:
        """Every event after `from_seq`. Backs SSE reconnect."""
        ...

    async def stream(
        self,
        run_id: uuid.UUID,
        *,
        from_seq: int = 0,
    ) -> AsyncIterator[DomainEvent]:
        """Replay-then-live, with no gap and no duplicates.

        The seam between the two is the whole difficulty: subscribe FIRST, then
        replay, then de-duplicate on seq. Replaying first would drop any event
        published in the window between the replay finishing and the
        subscription starting.
        """
        ...
