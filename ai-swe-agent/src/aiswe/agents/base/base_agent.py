"""Base agent.

A Template Method. Subclasses implement ONE method -- `execute()` -- and inherit
every cross-cutting concern:

    budget check -> visit guard -> event(entered) -> retry loop
                 -> execute()   -> validate -> event(exited)
                 -> error capture

WHY A FAILING AGENT MUST NOT RAISE
----------------------------------
`__call__` catches everything. A failure becomes a structured `AgentError`
written INTO the state, and control returns to the supervisor.

That is the difference between a system that degrades and a system that dies.
If the Testing agent raises, the whole run is over and the user gets a stack
trace. If it records "pytest not found, code=tool_execution_failed" into state,
the supervisor can route to the Planner, which can replan around it. One is an
outage; the other is a Tuesday.

The only exceptions that escape are the ones nothing downstream can act on:
budget exhaustion and step-limit breach. Those genuinely end the run.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from aiswe.agents.graph.state import AgentError as StateError
from aiswe.agents.graph.state import AgentState, Message
from aiswe.domain.errors import (
    BudgetExceededError,
    DomainError,
    MaxIterationsError,
)
from aiswe.domain.events import domain_events as events
from aiswe.domain.ports.agent import AgentDescriptor, AgentOutput
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.domain.ports.event_bus import EventBus
    from aiswe.domain.ports.llm import LLMProvider
    from aiswe.domain.ports.tool import ToolContext, ToolRegistry, ToolResult

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Every agent extends this. Nothing else.

    Dependencies arrive by injection. An agent that reaches for a global to find
    its LLM cannot be tested without one, and cannot be given a cheaper model for
    a cheap task.
    """

    #: Subclasses MUST override. The framework reads only this.
    descriptor: AgentDescriptor

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._events = event_bus

        if not hasattr(self, "descriptor"):
            msg = f"{type(self).__name__} must declare a `descriptor` class attribute."
            raise TypeError(msg)

    # --- Subclass contract ---------------------------------------------------

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentOutput:
        """The agent's actual work.

        May raise. The base class handles retry, event emission, and error
        capture -- so an agent's own code stays about its job, not plumbing.
        """

    # --- Template method -----------------------------------------------------

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """Graph entrypoint. Returns a partial state update, never raises
        (except for genuinely unrecoverable conditions).

        LangGraph merges the returned dict into state using the reducers
        declared on AgentState.
        """
        role = str(self.descriptor.role)
        started = time.perf_counter()

        # --- Deterministic guards, BEFORE any spend --------------------------
        # Checked in code, not asked of a model. A budget guard that an LLM can
        # talk its way past is not a guard.
        if state.budget.exhausted:
            reason = state.budget.exhausted_reason or "budget exhausted"
            await self._emit(events.budget_exceeded(state.run_id, reason=reason))
            raise BudgetExceededError(reason)

        visits = state.visits(role) + 1
        if visits > self.descriptor.max_consecutive_visits:
            msg = (
                f"Agent {role!r} was entered {visits} times consecutively "
                f"(limit {self.descriptor.max_consecutive_visits}). Likely livelock."
            )
            raise MaxIterationsError(msg, agent=role)

        await self._emit(events.agent_entered(state.run_id, role, iteration=state.iteration))

        # --- Retry loop ------------------------------------------------------
        policy = self.descriptor.retry
        last_error: Exception | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                output = await self.execute(state)
            except Exception as exc:  # noqa: BLE001 - deliberate: see module docstring
                last_error = exc

                # Unrecoverable: let these kill the run.
                if isinstance(exc, BudgetExceededError | MaxIterationsError):
                    raise

                if policy.should_retry(exc, attempt=attempt):
                    delay = policy.delay_for(attempt)
                    await self._emit(
                        events.agent_retrying(
                            state.run_id,
                            role,
                            attempt=attempt,
                            max_attempts=policy.max_attempts,
                            delay_s=delay,
                            reason=str(exc),
                        )
                    )
                    logger.warning(
                        "agent.retrying",
                        agent=role,
                        attempt=attempt,
                        delay_s=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue

                # Out of attempts, or not worth retrying.
                return self._failure_update(state, role, exc, attempt=attempt)

            # --- Success -----------------------------------------------------
            duration_ms = (time.perf_counter() - started) * 1000
            await self._emit(
                events.agent_exited(
                    state.run_id, role, summary=output.summary, duration_ms=duration_ms
                )
            )
            logger.info("agent.completed", agent=role, duration_ms=round(duration_ms, 2))
            return self._success_update(state, role, output, visits=visits)

        # Unreachable: the loop either returns or raises. Belt and braces.
        return self._failure_update(
            state,
            role,
            last_error or RuntimeError("agent exhausted retries"),
            attempt=policy.max_attempts,
        )

    # --- State updates -------------------------------------------------------

    def _success_update(
        self,
        state: AgentState,
        role: str,
        output: AgentOutput,
        *,
        visits: int,
    ) -> dict[str, Any]:
        """Translate AgentOutput into a partial state update."""
        update: dict[str, Any] = dict(output.state_delta)

        update["iteration"] = 1  # `increment` reducer
        update["budget"] = state.budget.step()

        # Reset every other agent's consecutive-visit counter. The guard is
        # about consecutive re-entry (A -> A -> A), not lifetime usage: an agent
        # legitimately visited twenty times across a long plan is not a livelock.
        update["visit_counts"] = {
            **{k: 0 for k in state.visit_counts if k != role},
            role: visits,
        }

        update.setdefault(
            "messages",
            [Message(role="assistant", content=output.summary, agent=role)],
        )

        if output.status == "needs_approval" and "awaiting_approval" not in update:
            logger.warning("agent.approval_without_request", agent=role)

        return update

    def _failure_update(
        self,
        state: AgentState,
        role: str,
        exc: Exception,
        *,
        attempt: int,
    ) -> dict[str, Any]:
        """A failure is DATA, not an exception. The supervisor routes on it."""
        code = exc.code if isinstance(exc, DomainError) else type(exc).__name__
        retryable = bool(getattr(exc, "retryable", False))

        logger.error("agent.failed", agent=role, code=code, error=str(exc), attempt=attempt)

        # Fire-and-forget: we are already in a failure path and must not mask
        # the original error with an event-bus hiccup.
        if self._events:
            import contextlib

            with contextlib.suppress(Exception):
                asyncio.get_running_loop().create_task(  # noqa: RUF006
                    self._events.publish(
                        events.agent_failed(state.run_id, role, error=str(exc), code=code)
                    )
                )

        return {
            "iteration": 1,
            "budget": state.budget.step(),
            "errors": [
                StateError(
                    agent=role,
                    code=code,
                    message=str(exc),
                    retryable=retryable,
                    attempt=attempt,
                )
            ],
            "visit_counts": {**state.visit_counts, role: state.visits(role) + 1},
            # Hand control back. The supervisor decides whether to replan,
            # route elsewhere, or give up -- the failing agent does not.
            "next_agent": None,
        }

    # --- Helpers for subclasses ---------------------------------------------

    async def _emit(self, event: Any) -> None:
        """Publish an event. Never fails the agent -- see EventBus.publish."""
        if self._events is None:
            return
        try:
            await self._events.publish(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning("event.publish_failed", error=str(exc))

    async def call_tool(
        self,
        name: str,
        ctx: ToolContext,
        args: dict[str, Any],
    ) -> ToolResult:
        """Invoke a tool, scoped to THIS agent's declared capabilities.

        The registry enforces the scope. An agent cannot widen its own
        permissions by asking nicely, because it never receives the tool.
        """
        if self._tools is None:
            msg = f"Agent {self.descriptor.role!r} has no tool registry bound."
            raise RuntimeError(msg)
        return await self._tools.invoke(
            name,
            ctx,
            args,
            capabilities=self.descriptor.capabilities,
        )

    @property
    def llm(self) -> LLMProvider:
        """The model. Raises if the agent declared `requires_llm=False`."""
        if self._llm is None:
            msg = f"Agent {self.descriptor.role!r} requires an LLM but none was injected."
            raise RuntimeError(msg)
        return self._llm

    def __repr__(self) -> str:
        return f"<{type(self).__name__} role={self.descriptor.role!r}>"
