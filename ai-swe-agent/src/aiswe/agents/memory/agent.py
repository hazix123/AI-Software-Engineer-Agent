"""Memory agent (the Memory Manager).

One node, two modes, chosen by where the run is:

    RECALL       -- at run start, before planning. You plan differently when you
                    know "this repo runs tests with `make test`, not pytest".
    CONSOLIDATE  -- at run end. Extracts durable lessons from what happened.

THE RULE THAT MATTERS
---------------------
Procedural memories ("how we do things here") are promoted ONLY from runs that
SUCCEEDED. A failed run's "lessons" are recorded as episodic with low confidence
and never promoted.

Without this rule the system confidently learns its own mistakes: a run that
failed because it used the wrong test command writes down that command as The
Way, and every subsequent run repeats it -- with growing confidence, because
usage_count keeps climbing. Memory then makes the system worse over time, which
is a strictly worse outcome than having no memory at all.

Semantic search over long-term memory lands in M8. The MemoryStore port is bound
to a null adapter until then; every call site here is already correct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from aiswe.agents.base.base_agent import BaseAgent
from aiswe.agents.registry import register_agent
from aiswe.domain.events.domain_events import DomainEvent, EventType
from aiswe.domain.ports.agent import AgentDescriptor, AgentOutput
from aiswe.domain.ports.llm import Message as LLMMessage
from aiswe.domain.ports.memory import (
    MemoryQuery,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from aiswe.domain.value_objects.agent_role import MEMORY
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.agents.graph.state import AgentState
    from aiswe.domain.ports.event_bus import EventBus
    from aiswe.domain.ports.llm import LLMProvider
    from aiswe.domain.ports.memory import MemoryStore

logger = get_logger(__name__)


class ExtractedMemory(BaseModel):
    """A candidate memory proposed by the model."""

    content: str = Field(description="One durable, self-contained fact.")
    type: str = Field(description="semantic | procedural | episodic | preference")
    title: str = Field(default="", description="Optional short label.")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class Consolidation(BaseModel):
    memories: list[ExtractedMemory] = Field(
        default_factory=list,
        description="Only durable, reusable facts. Omit anything run-specific.",
    )


_CONSOLIDATE_SYSTEM = """\
You extract durable lessons from a completed software engineering run.

Record ONLY facts that will still be useful on a DIFFERENT task in this same \
codebase.

Record:
- semantic:   structural truths ("auth lives in core/security/jwt.py")
- procedural: how work is done here ("tests run via `make test`, not pytest")
- preference: how the user wants things ("small, single-purpose PRs")

Do NOT record:
- what happened in this specific run (that is already in the run log)
- anything you are not confident about
- restatements of the objective

Prefer zero memories over a speculative one. A wrong memory is worse than no \
memory: it will be recalled, trusted, and acted on for months.
"""


@register_agent
class MemoryAgent(BaseAgent):
    """Recalls relevant context before work; consolidates lessons after it."""

    descriptor = AgentDescriptor(
        role=MEMORY,
        description=(
            "Recalls what is already known about this project, and records durable "
            "lessons once work is complete."
        ),
        when_to_use=(
            "At the start of a run, before planning. And at the end of a successful "
            "run, to persist what was learned."
        ),
        capabilities=frozenset({"memory"}),
        requires_llm=True,
        max_consecutive_visits=2,
    )

    def __init__(
        self,
        *,
        store: MemoryStore,
        llm: LLMProvider | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        super().__init__(llm=llm, event_bus=event_bus)
        self._store = store

    async def execute(self, state: AgentState) -> AgentOutput:
        # Mode is inferred from the run's shape, not passed in: a plan that is
        # complete means we are winding down; no plan at all means we are
        # starting up. Inferring keeps the graph edge simple.
        if state.plan is None:
            return await self._recall(state)
        return await self._consolidate(state)

    # --- Recall --------------------------------------------------------------

    async def _recall(self, state: AgentState) -> AgentOutput:
        records = await self._store.recall(
            MemoryQuery(
                text=state.objective,
                project_id=state.project_id,
                repository_id=state.repository_id,
                limit=10,
            )
        )

        # Reinforce what we used. Usage is the strongest available signal of a
        # memory's worth, and it drives both ranking and retention.
        for record in records:
            await self._store.reinforce(record.id)

        await self._emit(
            DomainEvent(
                type=EventType.MEMORY_RECALLED,
                run_id=state.run_id,
                agent=str(MEMORY),
                payload={"count": len(records)},
            )
        )

        logger.info("memory.recalled", count=len(records))

        delta: dict[str, Any] = {
            "memories": records,
            "next_agent": None,  # back to the supervisor, which will route to the planner
        }

        summary = (
            f"Recalled {len(records)} relevant memories."
            if records
            else "No prior memories for this project; starting cold."
        )
        return AgentOutput(status="ok", summary=summary, state_delta=delta)

    # --- Consolidate ---------------------------------------------------------

    async def _consolidate(self, state: AgentState) -> AgentOutput:
        succeeded = state.plan is not None and state.plan.is_complete and not state.errors

        completion = await self.llm.structured(
            [LLMMessage(role="user", content=_render_run(state))],
            schema=Consolidation,
            system=_CONSOLIDATE_SYSTEM,
            temperature=0.0,
            max_tokens=1_024,
        )

        written = 0
        for candidate in completion.value.memories:
            mem_type = _coerce_type(candidate.type)

            # THE GUARD. A failed run may not teach procedure. Demote to episodic
            # and halve the confidence, so it is recorded but never trusted as
            # "how we do things here".
            confidence = candidate.confidence
            if not succeeded and mem_type in {MemoryType.PROCEDURAL, MemoryType.SEMANTIC}:
                logger.info(
                    "memory.demoted_unverified",
                    original_type=str(mem_type),
                    reason="run did not succeed",
                )
                mem_type = MemoryType.EPISODIC
                confidence = min(confidence, 0.3)

            await self._store.remember(
                MemoryRecord(
                    content=candidate.content,
                    title=candidate.title,
                    type=mem_type,
                    scope=MemoryScope.PROJECT,
                    organization_id=state.organization_id,
                    project_id=state.project_id,
                    repository_id=state.repository_id,
                    source_run_id=state.run_id,  # provenance: a bad memory is traceable
                    confidence=confidence,
                    importance=candidate.importance,
                )
            )
            written += 1

        await self._emit(
            DomainEvent(
                type=EventType.MEMORY_WRITTEN,
                run_id=state.run_id,
                agent=str(MEMORY),
                payload={"count": written, "verified": succeeded},
            )
        )

        logger.info("memory.consolidated", count=written, verified=succeeded)

        return AgentOutput(
            status="ok",
            summary=f"Recorded {written} memories from this run.",
            state_delta={"done": True},
        )


# --- Helpers -----------------------------------------------------------------


def _coerce_type(raw: str) -> MemoryType:
    """Never trust a model to return a valid enum member."""
    try:
        return MemoryType(raw.strip().lower())
    except ValueError:
        logger.warning("memory.unknown_type", raw=raw)
        return MemoryType.EPISODIC  # the safest bucket: recorded, never promoted


def _render_run(state: AgentState) -> str:
    parts = [f"Objective:\n{state.objective}\n"]

    if state.plan:
        steps = "\n".join(
            f"  [{s.status}] {s.title}"
            + (f" -- {s.result_summary}" if s.result_summary else "")
            for s in state.plan.steps
        )
        parts.append(f"Plan outcome:\n{steps}\n")

    if state.errors:
        errs = "\n".join(f"  {e.agent}: [{e.code}] {e.message}" for e in state.errors[-5:])
        parts.append(f"Failures:\n{errs}\n")

    transcript = "\n".join(
        f"  [{m.agent or m.role}] {m.content[:300]}" for m in state.recent_messages(20)
    )
    parts.append(f"Activity:\n{transcript}\n")
    parts.append("Extract only durable, reusable lessons.")

    return "\n".join(parts)


__all__ = ["Consolidation", "ExtractedMemory", "MemoryAgent"]
