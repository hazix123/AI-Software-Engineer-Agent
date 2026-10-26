"""Agent run worker.

The process that actually executes runs.

    claim -> heartbeat (concurrently) -> execute graph -> ack / nack

THREE THINGS THIS GETS RIGHT
----------------------------
1. **Heartbeating runs CONCURRENTLY with the run.** A 40-minute refactor is not
   a stuck worker, but to a queue watching for idle messages the two look
   identical. Without the heartbeat, a healthy worker gets its task stolen
   mid-run and two workers execute the same run at once.

2. **SIGTERM is graceful.** A deploy must not kill a run mid-step. On SIGTERM the
   worker stops CLAIMING new work and lets the in-flight run finish. If it is
   killed anyway (SIGKILL after the grace period), the lease expires, another
   worker reclaims it, and the checkpoint means it resumes rather than restarts.

3. **A crashed run is nacked, not lost.** `nack` re-queues up to `max_attempts`,
   then dead-letters. A poison run cannot occupy the pool forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import socket
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from aiswe.agents.bootstrap import build_agent_registry
from aiswe.agents.graph.builder import GraphBuilder
from aiswe.agents.graph.checkpointer import build_checkpointer
from aiswe.agents.runtime import LangGraphRuntime
from aiswe.config.container import Container
from aiswe.config.settings import get_settings
from aiswe.domain.value_objects.run_status import RunStatus
from aiswe.domain.value_objects.token_budget import TokenBudget
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.domain.ports.task_queue import Task

logger = get_logger(__name__)

#: Lease heartbeat interval. Must be comfortably shorter than the reclaim
#: threshold, or a healthy worker races its own lease expiry.
_HEARTBEAT_INTERVAL_S = 30.0

#: Idle time after which a task is presumed abandoned.
_RECLAIM_AFTER_MS = 300_000  # 5 minutes

#: Default per-run ceilings. Overridden per run from M5's API.
_DEFAULT_BUDGET = TokenBudget(
    max_tokens=500_000,
    max_cost_usd=Decimal("5.00"),
    max_steps=40,
)


class RunWorker:
    """Consumes the task queue and executes agent runs."""

    def __init__(self, container: Container, *, concurrency: int = 2) -> None:
        self._container = container
        self._concurrency = concurrency
        self._consumer = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._shutdown = asyncio.Event()
        self._runtime: LangGraphRuntime | None = None

    # --- Lifecycle -----------------------------------------------------------

    async def setup(self) -> None:
        """Wire the graph once, at startup. Not per run -- compiling the graph on
        every task would waste real time on a hot path."""
        settings = self._container.settings

        registry = build_agent_registry(
            llm=self._container.llm,
            memory_store=self._container.memory_store,
            tools=self._container.tools,
            event_bus=self._container.event_bus,
        )

        checkpointer = await build_checkpointer(settings.db.dsn)
        graph = GraphBuilder(registry).build(checkpointer=checkpointer)

        self._runtime = LangGraphRuntime(graph, event_bus=self._container.event_bus)

        await self._container.task_queue.ensure_group()

        logger.info(
            "worker.ready",
            consumer=self._consumer,
            agents=len(registry),
            concurrency=self._concurrency,
        )

    async def run(self) -> None:
        """The main loop. Returns when a graceful shutdown completes."""
        await self.setup()
        self._install_signal_handlers()

        reclaimer = asyncio.create_task(self._reclaim_loop())
        workers = [asyncio.create_task(self._consume()) for _ in range(self._concurrency)]

        await self._shutdown.wait()
        logger.info("worker.draining", note="finishing in-flight runs")

        reclaimer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reclaimer

        # In-flight runs are allowed to finish. Only claiming stops.
        await asyncio.gather(*workers, return_exceptions=True)
        logger.info("worker.stopped", consumer=self._consumer)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown.set)

    # --- Consumption ---------------------------------------------------------

    async def _consume(self) -> None:
        queue = self._container.task_queue

        while not self._shutdown.is_set():
            tasks = await queue.claim(consumer=self._consumer, count=1, block_ms=5_000)
            if not tasks:
                continue  # the block timed out; loop and check for shutdown
            await self._handle(tasks[0])

    async def _handle(self, task: Task) -> None:
        """Execute one run, heartbeating its lease throughout."""
        assert self._runtime is not None  # noqa: S101 - guaranteed by setup()
        queue = self._container.task_queue

        logger.info("worker.claimed", run_id=str(task.run_id), attempt=task.attempts)

        # THE CONCURRENCY THAT MATTERS. Without this, a long run looks dead.
        heartbeat = asyncio.create_task(self._heartbeat(task))

        try:
            resuming = task.attempts > 1

            if resuming:
                # A previous worker died. Resume from the checkpoint; do NOT
                # start over -- the work already paid for is still valid.
                logger.info("worker.resuming", run_id=str(task.run_id))
                result = await self._runtime.resume(task.run_id)
            else:
                objective = str(task.payload.get("objective", ""))
                result = await self._runtime.start(
                    run_id=task.run_id,
                    objective=objective,
                    budget=_budget_from(task),
                    organization_id=_uuid_or_none(task.payload.get("organization_id")),
                    project_id=_uuid_or_none(task.payload.get("project_id")),
                    repository_id=_uuid_or_none(task.payload.get("repository_id")),
                )

            # An open approval gate is a PAUSE, not a failure. Ack the task: the
            # run resumes on a fresh task when a human decides, so no worker sits
            # blocked waiting for a person.
            if result.status is RunStatus.AWAITING_APPROVAL:
                logger.info("worker.parked_for_approval", run_id=str(task.run_id))
                await queue.ack(task)
                return

            await queue.ack(task)
            await self._container.scratchpad.clear(task.run_id)

            logger.info(
                "worker.completed",
                run_id=str(task.run_id),
                status=str(result.status),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("worker.run_failed", run_id=str(task.run_id))
            # Re-queue (up to max_attempts), then dead-letter. Never silently drop.
            await queue.nack(task, requeue=True)
            _ = exc

        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat(self, task: Task) -> None:
        """Keep the lease alive for as long as the run is genuinely working."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            await self._container.task_queue.extend_lease(task, ttl_ms=_RECLAIM_AFTER_MS)
            logger.debug("worker.heartbeat", run_id=str(task.run_id))

    # --- Crash recovery ------------------------------------------------------

    async def _reclaim_loop(self) -> None:
        """Periodically adopt tasks abandoned by dead workers."""
        while not self._shutdown.is_set():
            await asyncio.sleep(60)
            try:
                orphans = await self._container.task_queue.reclaim_expired(
                    consumer=self._consumer,
                    min_idle_ms=_RECLAIM_AFTER_MS,
                )
                for task in orphans:
                    logger.warning("worker.adopted_orphan", run_id=str(task.run_id))
                    await self._handle(task)
            except Exception:  # noqa: BLE001
                # The reclaim loop must never die: it is the mechanism that
                # recovers from other things dying.
                logger.exception("worker.reclaim_loop_error")


# --- Helpers -----------------------------------------------------------------


def _budget_from(task: Task) -> TokenBudget:
    raw = task.payload.get("budget") or {}
    if not raw:
        return _DEFAULT_BUDGET
    return TokenBudget(
        max_tokens=int(raw.get("max_tokens", _DEFAULT_BUDGET.max_tokens)),
        max_cost_usd=Decimal(str(raw.get("max_cost_usd", _DEFAULT_BUDGET.max_cost_usd))),
        max_steps=int(raw.get("max_steps", _DEFAULT_BUDGET.max_steps)),
    )


def _uuid_or_none(raw: object) -> uuid.UUID | None:
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        return None


async def main() -> None:
    """Entrypoint: `aiswe worker` / `python -m aiswe.workers.run_worker`."""
    container = Container.build(get_settings())
    await container.startup()
    try:
        await RunWorker(container).run()
    finally:
        await container.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
