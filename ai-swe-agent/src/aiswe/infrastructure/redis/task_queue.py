"""Redis Streams task queue.

Consumer groups give us exactly the semantics we need, without a broker:

    XADD        enqueue
    XREADGROUP  claim, and start a lease (the message enters the group's PEL)
    XACK        complete, removing it from the pending list
    XAUTOCLAIM  steal messages whose owner died -- this IS crash recovery
    XPENDING    observe leases

WHY NOT A LIST + BRPOP
----------------------
`BRPOP` removes the message the instant it is taken. If the worker then dies,
the task is gone: no lease, no redelivery, no way to know it ever existed. For a
20-minute agent run that a user is watching, silently losing the work is the
worst possible failure.

Consumer groups keep the message in a pending-entries list until it is acked, so
a dead worker's task can be reclaimed and RESUMED FROM ITS CHECKPOINT.

THE DEAD LETTER CAP
-------------------
`max_attempts` is not a nicety. A task that fails deterministically -- a
malformed objective, a repo that no longer exists -- will be reclaimed, retried,
and fail again, forever, occupying a worker each time. Enough of them and the
pool never drains and the queue never moves. The cap is what makes the system
self-healing rather than self-strangling.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from redis.exceptions import RedisError, ResponseError

from aiswe.config.constants import QUEUE_NS
from aiswe.domain.errors import InfrastructureError
from aiswe.domain.ports.task_queue import Task
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


class RedisTaskQueue:
    """Implements `domain.ports.task_queue.TaskQueue`."""

    def __init__(
        self,
        client: Redis,
        *,
        stream: str = f"{QUEUE_NS}:runs",
        group: str = "workers",
        max_attempts: int = 3,
    ) -> None:
        self._redis = client
        self._stream = stream
        self._group = group
        self._dlq = f"{stream}:dead"
        self._max_attempts = max_attempts

    # --- Lifecycle -----------------------------------------------------------

    async def ensure_group(self) -> None:
        """Create the consumer group. Idempotent; safe to call on every boot.

        `mkstream=True` creates the stream if it does not exist, so a cold start
        with an empty queue does not error.
        """
        try:
            await self._redis.xgroup_create(
                self._stream,
                self._group,
                id="0",
                mkstream=True,
            )
            logger.info("queue.group_created", stream=self._stream, group=self._group)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            # Already exists. Expected on every boot after the first.

    # --- Produce -------------------------------------------------------------

    async def enqueue(
        self,
        run_id: uuid.UUID,
        *,
        kind: str = "run",
        payload: dict[str, Any] | None = None,
    ) -> str:
        try:
            message_id: str = await self._redis.xadd(
                self._stream,
                {
                    "run_id": str(run_id),
                    "kind": kind,
                    "payload": json.dumps(payload or {}),
                    "enqueued_at": datetime.now(UTC).isoformat(),
                },
            )
        except RedisError as exc:
            msg = "Failed to enqueue the task."
            raise InfrastructureError(msg, code="queue_enqueue_failed") from exc

        logger.info("queue.enqueued", run_id=str(run_id), kind=kind, message_id=message_id)
        return message_id

    # --- Consume -------------------------------------------------------------

    async def claim(
        self,
        *,
        consumer: str,
        count: int = 1,
        block_ms: int = 5_000,
    ) -> list[Task]:
        """Claim tasks, blocking briefly for work.

        Blocking is what keeps an idle worker at ~0% CPU. A polling loop would
        burn a core doing nothing, times every replica.
        """
        try:
            response = await self._redis.xreadgroup(
                groupname=self._group,
                consumername=consumer,
                streams={self._stream: ">"},  # ">" = messages never delivered before
                count=count,
                block=block_ms,
            )
        except RedisError as exc:
            logger.warning("queue.claim_failed", error=str(exc))
            return []

        return _to_tasks(response)

    async def ack(self, task: Task) -> None:
        """Done. Remove from the pending list, then trim the stream entry."""
        await self._redis.xack(self._stream, self._group, task.id)
        await self._redis.xdel(self._stream, task.id)
        logger.info("queue.acked", run_id=str(task.run_id), message_id=task.id)

    async def nack(self, task: Task, *, requeue: bool = True) -> None:
        """Failed.

        Ack + re-enqueue rather than "leave it pending": leaving it pending means
        it is only redelivered when the lease expires, which could be five
        minutes away. An explicit re-enqueue retries promptly.
        """
        attempts = task.attempts

        if not requeue or attempts >= self._max_attempts:
            await self.dead_letter(task, reason=f"failed after {attempts} attempts")
            return

        await self._redis.xack(self._stream, self._group, task.id)
        await self._redis.xdel(self._stream, task.id)
        await self.enqueue(task.run_id, kind=task.kind, payload=task.payload)
        logger.warning("queue.requeued", run_id=str(task.run_id), attempts=attempts)

    async def extend_lease(self, task: Task, *, ttl_ms: int) -> None:  # noqa: ARG002
        """Heartbeat.

        Redis Streams have no explicit lease TTL: idleness is measured from the
        last delivery. `XCLAIM` with `justid` re-delivers the message to the same
        consumer, resetting its idle clock -- which is exactly a heartbeat.

        Without this, a legitimate 40-minute refactor looks identical to a dead
        worker and gets its task stolen mid-run.
        """
        try:
            await self._redis.xclaim(
                self._stream,
                self._group,
                task.id,  # reclaim for the SAME consumer
                min_idle_time=0,
                message_ids=[task.id],
                justid=True,
            )
        except RedisError as exc:
            logger.warning("queue.heartbeat_failed", message_id=task.id, error=str(exc))

    async def reclaim_expired(
        self,
        *,
        consumer: str,
        min_idle_ms: int = 300_000,
        count: int = 10,
    ) -> list[Task]:
        """THE CRASH-RECOVERY PATH.

        Any message idle for longer than `min_idle_ms` had a worker that died
        without acking. XAUTOCLAIM transfers ownership to us; the runtime then
        resumes the run FROM ITS CHECKPOINT, not from the beginning.
        """
        try:
            _cursor, messages, _deleted = await self._redis.xautoclaim(
                name=self._stream,
                groupname=self._group,
                consumername=consumer,
                min_idle_time=min_idle_ms,
                count=count,
            )
        except RedisError as exc:
            logger.warning("queue.reclaim_failed", error=str(exc))
            return []

        tasks = _to_tasks([(self._stream, messages)])

        for task in tasks:
            # A task reclaimed too many times is poison. Park it, or it will
            # occupy a worker on every sweep, forever.
            if task.attempts > self._max_attempts:
                await self.dead_letter(
                    task, reason=f"reclaimed {task.attempts} times without completing"
                )
                continue
            logger.warning(
                "queue.reclaimed",
                run_id=str(task.run_id),
                attempts=task.attempts,
                note="a previous worker died; resuming from checkpoint",
            )

        return [t for t in tasks if t.attempts <= self._max_attempts]

    async def dead_letter(self, task: Task, *, reason: str) -> None:
        """Park a poison task, with everything needed to diagnose it."""
        await self._redis.xadd(
            self._dlq,
            {
                "run_id": str(task.run_id),
                "kind": task.kind,
                "payload": json.dumps(task.payload),
                "attempts": str(task.attempts),
                "reason": reason,
                "at": datetime.now(UTC).isoformat(),
            },
        )
        await self._redis.xack(self._stream, self._group, task.id)
        await self._redis.xdel(self._stream, task.id)
        logger.error(
            "queue.dead_lettered",
            run_id=str(task.run_id),
            attempts=task.attempts,
            reason=reason,
        )

    # --- Observability -------------------------------------------------------

    async def depth(self) -> int:
        """Undelivered tasks. Drives autoscaling."""
        try:
            return int(await self._redis.xlen(self._stream))
        except RedisError:
            return 0

    async def in_flight(self) -> int:
        """Claimed but unacked. A rising number with a flat depth means workers
        are stuck, not busy -- the distinction that matters at 3am."""
        try:
            info = await self._redis.xpending(self._stream, self._group)
            return int(info["pending"]) if info else 0
        except RedisError:
            return 0


def _to_tasks(response: Any) -> list[Task]:
    """Decode XREADGROUP / XAUTOCLAIM output into domain Tasks."""
    tasks: list[Task] = []
    if not response:
        return tasks

    for _stream_name, messages in response:
        for message_id, fields in messages:
            try:
                tasks.append(
                    Task(
                        id=message_id,
                        run_id=uuid.UUID(fields["run_id"]),
                        kind=fields.get("kind", "run"),
                        payload=json.loads(fields.get("payload", "{}")),
                        # Redis appends a delivery counter on redelivery; absent
                        # on first delivery.
                        attempts=int(fields.get("delivery_count", 1)),
                        enqueued_at=_parse_dt(fields.get("enqueued_at")),
                        claimed_at=datetime.now(UTC),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.error("queue.malformed_message", message_id=message_id, error=str(exc))
    return tasks


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
