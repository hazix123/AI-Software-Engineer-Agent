"""Agent port.

THIS IS THE FILE THAT DELIVERS "ADD AN AGENT WITHOUT MODIFYING EXISTING CODE".

An agent DECLARES itself via an `AgentDescriptor`: its role, what it can do,
which tools it may touch, and when it is applicable. The graph builder, the
router, and the tool registry all consume that declaration and nothing else.

So registering the Security Review Agent in M7 is:

    @register_agent
    class SecurityReviewAgent(BaseAgent):
        descriptor = AgentDescriptor(role=AgentRole("security_review"), ...)

and it is automatically: a node in the graph, a legal routing destination, a
budget-tracked participant, and scoped to exactly the tools it declared. No
enum grows. No `if role == ...` chain grows. No existing file is touched.

The alternative -- a central enum plus a match statement in the supervisor --
looks fine with three agents and is a merge-conflict factory at ten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

from aiswe.domain.policies.retry_policy import DEFAULT_AGENT_RETRY, RetryPolicy
from aiswe.domain.value_objects.agent_role import AgentRole

#: The blackboard type an agent operates on.
#:
#: Deliberately a TypeVar, not `AgentState`. The domain must not import the
#: orchestration layer -- that would invert the dependency rule and make the
#: business core depend on the graph. Generics let the domain state the CONTRACT
#: ("an agent maps some state to an AgentOutput") while the agents layer supplies
#: the concrete state. The architecture test in tests/unit/test_architecture.py
#: is what caught this; it is not a hypothetical.
StateT = TypeVar("StateT", contravariant=True)


class AgentOutput(BaseModel):
    """The handoff contract. EVERY agent returns exactly this.

    `next_hint` is deliberately a HINT. Routing authority stays with the
    supervisor. If an agent could *command* the next hop, two agents would
    eventually command each other and the run would ping-pong until the budget
    died. Centralised routing is what makes agent loops bounded and debuggable.
    """

    status: str = Field(description="ok | failed | needs_approval | needs_replan")
    summary: str = Field(description="One line, human-readable. Shown in the UI.")

    state_delta: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial AgentState update. Merged by the graph's reducers.",
    )
    next_hint: str | None = Field(
        default=None,
        description="Suggested next agent. The supervisor may ignore it.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """An agent's self-declaration. The framework's only view of it.

    `description` and `when_to_use` are NOT decoration: they are injected into
    the supervisor's routing prompt. A vague `when_to_use` produces a vague
    router. Write them as instructions to a colleague who has never seen the
    codebase.
    """

    role: AgentRole
    description: str
    when_to_use: str

    # Tool tags this agent may access. The registry intersects these with each
    # tool's tags -- an agent literally cannot call a tool outside this set.
    capabilities: frozenset[str] = field(default_factory=frozenset)

    # Routing metadata.
    routable: bool = True  # can the supervisor send work here?
    terminal: bool = False  # does entering this node end the run?
    parallel_safe: bool = False  # may it fan out alongside other agents?

    # Execution policy.
    retry: RetryPolicy = DEFAULT_AGENT_RETRY
    max_consecutive_visits: int = 3  # livelock guard, per agent
    requires_llm: bool = True

    def __post_init__(self) -> None:
        if not self.description or not self.when_to_use:
            msg = f"Agent {self.role!r} must declare a description and when_to_use."
            raise ValueError(msg)


@runtime_checkable
class Agent(Protocol[StateT]):
    """What the graph needs from an agent. Nothing more.

    Note the signature: state in, output out. An agent cannot reach into the
    database, publish events, or call another agent directly. Everything it is
    allowed to do arrives through injected ports, and everything it wants to
    say goes into AgentOutput. That is what keeps agents unit-testable without
    a running graph.
    """

    @property
    def descriptor(self) -> AgentDescriptor:
        ...

    async def run(self, state: StateT) -> AgentOutput:
        ...


@runtime_checkable
class AgentRegistry(Protocol):
    """The single source of truth about which agents exist."""

    def register(self, agent: Agent[Any]) -> None:
        ...

    def get(self, role: AgentRole) -> Agent[Any]:
        """Raises AgentNotFoundError if the role is unknown."""
        ...

    def has(self, role: AgentRole) -> bool:
        ...

    def all(self) -> list[Agent[Any]]:
        ...

    def routable(self) -> list[Agent[Any]]:
        """Agents the supervisor may dispatch to."""
        ...

    def descriptors(self) -> list[AgentDescriptor]:
        """Consumed by the graph builder and the routing prompt."""
        ...
