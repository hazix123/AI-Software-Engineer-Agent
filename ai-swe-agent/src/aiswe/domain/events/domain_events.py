"""Domain events.

One vocabulary, two consumers: the SSE stream the user watches, and the durable
`run_events` log used for audit and reconnect-replay. Because they share a
definition, the UI can never display something the audit trail does not have.

Events are pure data. Publishing them is the EventBus port's job; these classes
know nothing about Redis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    """Wire-level event names. Clients switch on these."""

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"

    AGENT_ENTERED = "agent.entered"
    AGENT_EXITED = "agent.exited"
    AGENT_FAILED = "agent.failed"
    AGENT_RETRYING = "agent.retrying"

    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"

    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_DENIED = "tool.denied"

    TOKEN_DELTA = "token.delta"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESOLVED = "approval.resolved"

    MEMORY_RECALLED = "memory.recalled"
    MEMORY_WRITTEN = "memory.written"

    BUDGET_WARNING = "budget.warning"
    BUDGET_EXCEEDED = "budget.exceeded"


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Base event.

    `seq` is assigned by the EventBus at publish time, not here: only the bus
    can guarantee monotonicity per run. A client that reconnects sends
    `Last-Event-ID: <seq>` and the bus replays from the durable log.
    """

    type: EventType
    run_id: uuid.UUID
    payload: dict[str, Any] = field(default_factory=dict)

    event_id: uuid.UUID = field(default_factory=uuid.uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    seq: int | None = None
    agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Wire format. The SSE layer serialises this directly."""
        return {
            "event_id": str(self.event_id),
            "type": str(self.type),
            "run_id": str(self.run_id),
            "seq": self.seq,
            "agent": self.agent,
            "occurred_at": self.occurred_at.isoformat(),
            "payload": self.payload,
        }


# --- Constructors ------------------------------------------------------------
# Thin helpers so call sites read as prose and payload keys stay consistent.
# A typo in a payload key is otherwise invisible until a client breaks.


def agent_entered(run_id: uuid.UUID, agent: str, *, iteration: int) -> DomainEvent:
    return DomainEvent(
        type=EventType.AGENT_ENTERED,
        run_id=run_id,
        agent=agent,
        payload={"iteration": iteration},
    )


def agent_exited(
    run_id: uuid.UUID,
    agent: str,
    *,
    summary: str,
    duration_ms: float,
) -> DomainEvent:
    return DomainEvent(
        type=EventType.AGENT_EXITED,
        run_id=run_id,
        agent=agent,
        payload={"summary": summary, "duration_ms": round(duration_ms, 2)},
    )


def agent_failed(run_id: uuid.UUID, agent: str, *, error: str, code: str) -> DomainEvent:
    return DomainEvent(
        type=EventType.AGENT_FAILED,
        run_id=run_id,
        agent=agent,
        payload={"error": error, "code": code},
    )


def agent_retrying(
    run_id: uuid.UUID,
    agent: str,
    *,
    attempt: int,
    max_attempts: int,
    delay_s: float,
    reason: str,
) -> DomainEvent:
    return DomainEvent(
        type=EventType.AGENT_RETRYING,
        run_id=run_id,
        agent=agent,
        payload={
            "attempt": attempt,
            "max_attempts": max_attempts,
            "delay_s": round(delay_s, 2),
            "reason": reason,
        },
    )


def tool_called(run_id: uuid.UUID, agent: str, *, tool: str, args_preview: str) -> DomainEvent:
    return DomainEvent(
        type=EventType.TOOL_CALLED,
        run_id=run_id,
        agent=agent,
        payload={"tool": tool, "args_preview": args_preview},
    )


def tool_result(
    run_id: uuid.UUID,
    agent: str,
    *,
    tool: str,
    ok: bool,
    preview: str,
    duration_ms: float,
) -> DomainEvent:
    return DomainEvent(
        type=EventType.TOOL_RESULT,
        run_id=run_id,
        agent=agent,
        payload={
            "tool": tool,
            "ok": ok,
            "preview": preview,
            "duration_ms": round(duration_ms, 2),
        },
    )


def budget_exceeded(run_id: uuid.UUID, *, reason: str) -> DomainEvent:
    return DomainEvent(
        type=EventType.BUDGET_EXCEEDED,
        run_id=run_id,
        payload={"reason": reason},
    )


def run_finished(run_id: uuid.UUID, *, status: str, summary: str) -> DomainEvent:
    return DomainEvent(
        type=EventType.RUN_FINISHED,
        run_id=run_id,
        payload={"status": status, "summary": summary},
    )
