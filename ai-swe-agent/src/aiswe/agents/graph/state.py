"""The shared agent state — the blackboard.

Agents do not call each other. They read this, and they write this. That is the
entire inter-agent communication mechanism, and the constraint is deliberate:
direct agent-to-agent calls produce a call graph nobody can reason about, and
cycles nobody can bound.

TWO THINGS HERE ARE LOAD-BEARING
--------------------------------
1. **Reducers.** Fields carry an `Annotated[..., reducer]` merge function.
   LangGraph applies them when merging partial updates -- including from
   PARALLEL branches. In M7, Security / Documentation / Testing fan out
   simultaneously; without reducers the last branch to finish silently
   overwrites the other two's findings. `append_unique` is what makes that
   merge correct instead of lossy.

2. **Size discipline.** State is checkpointed on every node transition. A state
   that grows without bound turns a 40-step run into a serialisation death
   spiral, and blows the model's context window long before that. So large
   payloads (file contents, diffs, test logs) live in the artifact store, and
   the state holds a REFERENCE plus a summary. `enforce_limits()` is the
   backstop.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from aiswe.domain.ports.memory import MemoryRecord
from aiswe.domain.value_objects.agent_role import AgentRole
from aiswe.domain.value_objects.token_budget import TokenBudget

# --- Reducers ----------------------------------------------------------------
# LangGraph calls these to merge a node's partial update into the running state.
# Without them the default is "replace", which is wrong for anything a parallel
# branch contributes to.


def append_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Append, then window.

    Keeping every message forever guarantees context overflow on a long run.
    The window keeps the head (the objective and the plan, which must never be
    forgotten) and the tail (recent, relevant work), and drops the middle --
    which is where the least useful turns live.
    """
    merged = [*left, *right]
    if len(merged) <= _MAX_MESSAGES:
        return merged
    head = merged[:_MESSAGE_HEAD]
    tail = merged[-(_MAX_MESSAGES - _MESSAGE_HEAD) :]
    return [*head, *tail]


def append_unique(left: list[Any], right: list[Any]) -> list[Any]:
    """Append, de-duplicating on a stable key.

    Two parallel agents that both flag the same finding should produce one
    finding, not two. Falls back to identity for unhashable items.
    """
    merged = [*left, *right]
    seen: set[Any] = set()
    out: list[Any] = []
    for item in merged:
        key = getattr(item, "dedupe_key", None)
        key = key() if callable(key) else repr(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def merge_scratchpads(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Per-agent private notes. Each agent owns its own key; no clobbering."""
    return {**left, **right}


def take_latest(left: Any, right: Any) -> Any:  # noqa: ARG001
    """Last write wins. Correct for single-writer fields (plan, next_agent)."""
    return right


def increment(left: int, right: int) -> int:
    """Additive counter. Survives parallel branches correctly."""
    return left + right


_MAX_MESSAGES = 60
_MESSAGE_HEAD = 6
_MAX_SUMMARY_CHARS = 2_000


# --- State components --------------------------------------------------------


class Message(BaseModel):
    """A turn in the run's narrative. Not the LLM wire format -- that is built
    per-call by the ContextBuilder from this plus retrieved context."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    agent: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlanStep(BaseModel):
    """One unit of the plan. Owned by the Planner, consumed by the Supervisor."""

    id: str
    title: str
    description: str
    assigned_agent: str | None = Field(
        default=None,
        description="Role name. None means the supervisor decides at dispatch time.",
    )
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "done", "skipped", "failed"] = "pending"
    result_summary: str = ""
    attempts: int = 0

    def dedupe_key(self) -> str:
        return self.id


class Plan(BaseModel):
    """A DAG of steps. Versioned, because replanning is normal, not exceptional."""

    version: int = 1
    rationale: str = ""
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(s.status in {"done", "skipped"} for s in self.steps)

    @property
    def has_failure(self) -> bool:
        return any(s.status == "failed" for s in self.steps)

    def next_ready_step(self) -> PlanStep | None:
        """First pending step whose dependencies are all satisfied.

        This is what makes the plan a DAG rather than a list: steps become
        eligible when their prerequisites complete, not merely when their turn
        comes around.
        """
        done = {s.id for s in self.steps if s.status in {"done", "skipped"}}
        for step in self.steps:
            if step.status == "pending" and all(d in done for d in step.depends_on):
                return step
        return None


class ContextItem(BaseModel):
    """A retrieved snippet, always with a citation.

    `source` is mandatory. An agent that reports a fact it cannot cite is
    hallucinating, and the citation is what lets a human check it in one click.
    """

    content: str
    source: str  # "src/auth/jwt.py:42-88"
    relevance: float = 1.0


class AgentError(BaseModel):
    """A failure, recorded in state so the supervisor can route around it."""

    agent: str
    code: str
    message: str
    retryable: bool = False
    attempt: int = 1
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def dedupe_key(self) -> str:
        return f"{self.agent}:{self.code}:{self.attempt}"


class ApprovalRequest(BaseModel):
    """A human gate. The graph halts here until it is resolved."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kind: str  # plan | patch | push | dependency | migration | shell
    agent: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    """A pointer to something too large to live in state.

    THE context-discipline primitive. A 40,000-line test log becomes: a summary,
    a size, and a key. The agent sees three lines instead of blowing its window.
    """

    kind: str  # diff | test_report | log | doc | screenshot
    key: str  # object-store key
    summary: str
    size_bytes: int = 0
    produced_by: str = ""


# --- The blackboard ----------------------------------------------------------


class AgentState(BaseModel):
    """The single shared state object, checkpointed after every node.

    Every field carries an explicit reducer. If you add a field and do not think
    about its merge semantics, you have added a race condition that will only
    appear once agents run in parallel -- i.e. in M7, under load, in production.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Identity (immutable for the life of the run) ------------------------
    run_id: uuid.UUID
    objective: str
    organization_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    repository_id: uuid.UUID | None = None

    # --- Conversation --------------------------------------------------------
    messages: Annotated[list[Message], append_messages] = Field(default_factory=list)

    # --- Planning (single writer: the Planner) -------------------------------
    plan: Annotated[Plan | None, take_latest] = None
    current_step_id: Annotated[str | None, take_latest] = None

    # --- Working memory ------------------------------------------------------
    # Per-agent private notes. An agent writes only its own key, so two agents
    # cannot corrupt each other's reasoning.
    scratchpad: Annotated[dict[str, str], merge_scratchpads] = Field(default_factory=dict)

    # Retrieved context and recalled memories are REPLACED per step, not
    # accumulated: yesterday's retrieval for a different sub-task is noise, and
    # noise in the context window is the most expensive thing in the system.
    retrieved_context: Annotated[list[ContextItem], take_latest] = Field(default_factory=list)
    memories: Annotated[list[MemoryRecord], take_latest] = Field(default_factory=list)

    # --- Accumulated outputs (parallel-safe) ---------------------------------
    artifacts: Annotated[list[ArtifactRef], append_unique] = Field(default_factory=list)
    errors: Annotated[list[AgentError], append_unique] = Field(default_factory=list)

    # --- Control -------------------------------------------------------------
    next_agent: Annotated[str | None, take_latest] = None
    awaiting_approval: Annotated[ApprovalRequest | None, take_latest] = None
    done: Annotated[bool, take_latest] = False
    final_summary: Annotated[str, take_latest] = ""

    iteration: Annotated[int, increment] = 0
    budget: Annotated[TokenBudget, take_latest]

    # Livelock guard. Counts consecutive visits per agent so a Refactor <-> Test
    # ping-pong is caught in three hops, not three hundred.
    visit_counts: Annotated[dict[str, int], merge_scratchpads] = Field(default_factory=dict)

    # --- Queries -------------------------------------------------------------

    @property
    def current_step(self) -> PlanStep | None:
        if not self.plan or not self.current_step_id:
            return None
        return next((s for s in self.plan.steps if s.id == self.current_step_id), None)

    @property
    def last_error(self) -> AgentError | None:
        return self.errors[-1] if self.errors else None

    def visits(self, agent: str) -> int:
        return self.visit_counts.get(agent, 0)

    def scratch(self, agent: str) -> str:
        return self.scratchpad.get(agent, "")

    def recent_messages(self, n: int = 10) -> list[Message]:
        return self.messages[-n:]

    def context_block(self) -> str:
        """Retrieved snippets, rendered for a prompt, with citations intact."""
        if not self.retrieved_context:
            return ""
        return "\n\n".join(f"--- {c.source} ---\n{c.content}" for c in self.retrieved_context)

    def memory_block(self) -> str:
        """Recalled memories, rendered for a prompt."""
        if not self.memories:
            return ""
        return "\n".join(f"- {m.as_context()}" for m in self.memories)

    # --- Guards --------------------------------------------------------------

    def enforce_limits(self) -> Self:
        """Backstop against state bloat.

        The reducers already window messages; this catches everything else --
        an agent that dumped a 200KB summary into its scratchpad, for instance.
        Truncate loudly rather than fail mysteriously three nodes later.
        """
        trimmed = {
            agent: (
                note
                if len(note) <= _MAX_SUMMARY_CHARS
                else note[:_MAX_SUMMARY_CHARS] + ' …[truncated]'
            )
            for agent, note in self.scratchpad.items()
        }
        return self.model_copy(update={"scratchpad": trimmed})


def initial_state(
    *,
    run_id: uuid.UUID,
    objective: str,
    budget: TokenBudget,
    organization_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    repository_id: uuid.UUID | None = None,
) -> AgentState:
    """Construct the starting blackboard for a run."""
    return AgentState(
        run_id=run_id,
        objective=objective,
        budget=budget,
        organization_id=organization_id,
        project_id=project_id,
        repository_id=repository_id,
        messages=[Message(role="user", content=objective)],
        next_agent=str(AgentRole("memory")),  # always recall before planning
    )


__all__ = [
    "AgentError",
    "AgentState",
    "ApprovalRequest",
    "ArtifactRef",
    "ContextItem",
    "Message",
    "Plan",
    "PlanStep",
    "initial_state",
]
