"""Task queue port.

At-least-once delivery with leases. The lease is the important part.

A worker claims a run and dies -- OOM, deploy, spot-instance reclaim. Without a
lease the task is either lost forever (ack-on-claim) or duplicated immediately
(no tracking). With a lease it is neither: the lease expires, another worker
reclaims the task, and LangGraph resumes it FROM THE LAST CHECKPOINT rather
than starting the agent run over from scratch and re-billing the whole thing.

At-least-once, not exactly-once: exactly-once across a network is a fiction.
The system is made safe by resuming from checkpoints and by idempotent tool
design, not by pretending redelivery cannot happen.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Task:
    """A unit of queued work. Deliberately small.

    Carries a run_id, not a payload: the run's state lives in Postgres and its
    graph state in the checkpointer. A fat queue message would be a third,
    unversioned copy of the truth -- guaranteed to drift.
    """

    id: str  # queue-assigned message id
    run_id: uuid.UUID
    kind: str = "run"  # run | reindex | consolidate  (extensible)
    payload: dict[str, Any] = field(default_factory=dict)

    attempts: int = 0  # delivery count, incremented by the broker
    enqueued_at: datetime | None = None
    claimed_at: datetime | None = None


@runtime_checkable
class TaskQueue(Protocol):
    """Durable work distribution across a worker pool."""

    async def enqueue(
        self,
        run_id: uuid.UUID,
        *,
        kind: str = "run",
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Submit work. Returns the message id."""
        ...

    async def claim(
        self,
        *,
        consumer: str,
        count: int = 1,
        block_ms: int = 5_000,
    ) -> list[Task]:
        """Take ownership of up to `count` tasks, acquiring a lease on each.

        Blocks up to `block_ms` waiting for work, so an idle worker costs one
        connection rather than a spin loop.
        """
        ...

    async def ack(self, task: Task) -> None:
        """Work completed. Removes the task permanently."""
        ...

    async def nack(self, task: Task, *, requeue: bool = True) -> None:
        """Work failed. Requeue for another attempt, or dead-letter it."""
        ...

    async def extend_lease(self, task: Task, *, ttl_ms: int) -> None:
        """Heartbeat.

        A 40-minute refactor is not a stuck worker. Long-running tasks must
        extend their lease, or a healthy worker gets its task stolen mid-run.
        """
        ...

    async def reclaim_expired(
        self,
        *,
        consumer: str,
        min_idle_ms: int = 300_000,
        count: int = 10,
    ) -> list[Task]:
        """Recover tasks whose owner died without acking.

        This is the crash-recovery path. Run periodically by the scheduler.
        """
        ...

    async def dead_letter(self, task: Task, *, reason: str) -> None:
        """Park a poison task.

        A task that has failed N times must stop being retried. Otherwise one
        malformed run consumes the entire worker pool, forever, and the queue
        never drains.
        """
        ...

    async def depth(self) -> int:
        """Pending task count. Drives autoscaling and the /admin dashboard."""
        ...
