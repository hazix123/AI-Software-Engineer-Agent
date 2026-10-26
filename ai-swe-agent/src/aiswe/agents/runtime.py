"""Agent runtime.

The adapter between the application layer and LangGraph.

Use cases and workers depend on `AgentRuntime` (the port). They never import
`StateGraph`, never see a checkpoint, and never know what a "thread_id" is. If
LangGraph were replaced tomorrow, this file, `builder.py`, and `checkpointer.py`
would change -- and nothing else would.

That is not hypothetical purism. Agent orchestration libraries are eighteen
months old and moving fast; the odds of still being on this one in two years are
not high, and the cost of that migration should be three files, not thirty.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiswe.agents.graph.state import AgentState, initial_state
from aiswe.domain.errors import BudgetExceededError, MaxIterationsError
from aiswe.domain.events import domain_events as events
from aiswe.domain.value_objects.run_status import RunStatus
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from aiswe.domain.ports.event_bus import EventBus
    from aiswe.domain.value_objects.token_budget import TokenBudget

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of a run. What the worker persists and the API returns."""

    run_id: uuid.UUID
    status: RunStatus
    summary: str
    state: AgentState
    failure_reason: str | None = None


class LangGraphRuntime:
    """Implements `application.interfaces.agent_runtime.AgentRuntime`."""

    def __init__(
        self,
        graph: CompiledStateGraph[AgentState, Any, Any, Any],
        *,
        event_bus: EventBus | None = None,
        recursion_limit: int = 100,
    ) -> None:
        self._graph = graph
        self._events = event_bus
        # The OUTER guard. TokenBudget already caps steps, and the supervisor
        # already caps consecutive visits -- this is the backstop for when both
        # of those have a bug. Defence in depth on the one failure mode that
        # costs real money.
        self._recursion_limit = recursion_limit

    # --- Execution -----------------------------------------------------------

    async def start(
        self,
        *,
        run_id: uuid.UUID,
        objective: str,
        budget: TokenBudget,
        organization_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        repository_id: uuid.UUID | None = None,
    ) -> RunResult:
        """Execute a run from the beginning."""
        state = initial_state(
            run_id=run_id,
            objective=objective,
            budget=budget,
            organization_id=organization_id,
            project_id=project_id,
            repository_id=repository_id,
        )
        await self._publish(
            events.DomainEvent(
                type=events.EventType.RUN_STARTED,
                run_id=run_id,
                payload={"objective": objective},
            )
        )
        return await self._execute(run_id, state)

    async def resume(self, run_id: uuid.UUID) -> RunResult:
        """Resume from the last checkpoint.

        `None` as the input is LangGraph's "continue from where you stopped".
        This is the path taken after a worker crash and after a human resolves
        an approval gate -- the same mechanism serves both.
        """
        return await self._execute(run_id, None)

    async def _execute(self, run_id: uuid.UUID, state: AgentState | None) -> RunResult:
        config = self._config(run_id)

        try:
            raw = await self._graph.ainvoke(state, config=config)

        except BudgetExceededError as exc:
            logger.warning("run.budget_exceeded", run_id=str(run_id), reason=exc.message)
            return await self._terminal(
                run_id, RunStatus.FAILED, exc.message, failure="budget_exceeded"
            )

        except MaxIterationsError as exc:
            logger.warning("run.livelock", run_id=str(run_id), reason=exc.message)
            return await self._terminal(
                run_id, RunStatus.FAILED, exc.message, failure="max_iterations"
            )

        except Exception as exc:
            logger.exception("run.crashed", run_id=str(run_id))
            return await self._terminal(
                run_id,
                RunStatus.FAILED,
                f"The run crashed: {type(exc).__name__}: {exc}",
                failure="internal_error",
            )

        final = AgentState.model_validate(raw)

        # An open approval gate is NOT a failure. The run is parked, and the
        # graph will be resumed from this exact checkpoint once a human decides.
        if final.awaiting_approval is not None:
            logger.info("run.awaiting_approval", run_id=str(run_id))
            return RunResult(
                run_id=run_id,
                status=RunStatus.AWAITING_APPROVAL,
                summary=f"Awaiting approval: {final.awaiting_approval.summary}",
                state=final,
            )

        status = RunStatus.FAILED if _failed(final) else RunStatus.SUCCEEDED
        summary = final.final_summary or "Run complete."

        await self._publish(events.run_finished(run_id, status=str(status), summary=summary))
        logger.info("run.finished", run_id=str(run_id), status=str(status))

        return RunResult(run_id=run_id, status=status, summary=summary, state=final)

    # --- Streaming -----------------------------------------------------------

    async def stream(
        self,
        *,
        run_id: uuid.UUID,
        objective: str,
        budget: TokenBudget,
        **scope: uuid.UUID | None,
    ) -> AsyncIterator[AgentState]:
        """Yield state after every node. Backs the SSE endpoint (M5)."""
        state = initial_state(
            run_id=run_id,
            objective=objective,
            budget=budget,
            organization_id=scope.get("organization_id"),
            project_id=scope.get("project_id"),
            repository_id=scope.get("repository_id"),
        )
        async for chunk in self._graph.astream(
            state,
            config=self._config(run_id),
            stream_mode="values",
        ):
            yield AgentState.model_validate(chunk)

    # --- State access --------------------------------------------------------

    async def get_state(self, run_id: uuid.UUID) -> AgentState | None:
        """Read the checkpointed state without advancing the graph."""
        snapshot = await self._graph.aget_state(self._config(run_id))
        if not snapshot or not snapshot.values:
            return None
        return AgentState.model_validate(snapshot.values)

    async def approve(self, run_id: uuid.UUID, *, approved: bool, reason: str = "") -> RunResult:
        """Resolve an approval gate and continue.

        Clearing `awaiting_approval` is what un-blocks `ApprovalPendingRule`; a
        rejection instead records an error, which routes the supervisor to the
        planner for a different approach rather than simply stopping.
        """
        update: dict[str, Any] = {"awaiting_approval": None}
        if not approved:
            from aiswe.agents.graph.state import AgentError as StateError

            update["errors"] = [
                StateError(
                    agent="human",
                    code="approval_rejected",
                    message=reason or "A human rejected this operation.",
                    retryable=False,
                )
            ]

        await self._graph.aupdate_state(self._config(run_id), update)
        await self._publish(
            events.DomainEvent(
                type=events.EventType.APPROVAL_RESOLVED,
                run_id=run_id,
                payload={"approved": approved, "reason": reason},
            )
        )
        return await self.resume(run_id)

    # --- Internals -----------------------------------------------------------

    def _config(self, run_id: uuid.UUID) -> dict[str, Any]:
        """The run id IS the thread id. One run, one conversation, one checkpoint."""
        return {
            "configurable": {"thread_id": str(run_id)},
            "recursion_limit": self._recursion_limit,
        }

    async def _publish(self, event: events.DomainEvent) -> None:
        if self._events is None:
            return
        try:
            await self._events.publish(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("event.publish_failed", error=str(exc))

    async def _terminal(
        self,
        run_id: uuid.UUID,
        status: RunStatus,
        summary: str,
        *,
        failure: str,
    ) -> RunResult:
        """Build a failed result, recovering whatever state was checkpointed.

        Recovering the state matters: a failed run's partial work (the plan it
        got to, the errors it hit) is exactly what a human needs to debug it.
        """
        await self._publish(events.run_finished(run_id, status=str(status), summary=summary))
        state = await self.get_state(run_id)
        return RunResult(
            run_id=run_id,
            status=status,
            summary=summary,
            state=state,  # type: ignore[arg-type]
            failure_reason=failure,
        )


def _failed(state: AgentState) -> bool:
    """A run failed if the plan failed, or if it stopped without completing one."""
    if state.plan is None:
        return bool(state.errors)
    return state.plan.has_failure or not state.plan.is_complete
