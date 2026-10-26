"""Agent registry.

THE OPEN/CLOSED SEAM.

Two layers, and the split matters:

* `AgentClassRegistry` -- a module-level catalogue of agent CLASSES, populated
  by the `@register_agent` decorator at import time. It holds no dependencies
  and does no I/O, so importing an agent module is free and side-effect-safe.

* `InMemoryAgentRegistry` -- a per-process catalogue of agent INSTANCES, built
  once at startup by the composition root, with dependencies injected.

Why not one registry of pre-built instances? Because then a class could not be
declared without an LLM already existing, and the import graph would dictate the
startup order. Classes register themselves; the container decides what to build
and what to inject. That is dependency inversion applied to the registry itself.

Adding the Security Review Agent in M7:

    @register_agent
    class SecurityReviewAgent(BaseAgent):
        descriptor = AgentDescriptor(
            role=AgentRole("security_review"),
            description="Scans a patch for vulnerabilities.",
            when_to_use="After code is modified and before it is committed.",
            capabilities=frozenset({"security", "read"}),
        )
        async def execute(self, state): ...

That is the entire change. It is now a graph node, a legal routing target, a
budget participant, and scoped to security+read tools. No file in this package
is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from aiswe.domain.errors import AgentNotFoundError, ConflictError
from aiswe.domain.value_objects.agent_role import AgentRole
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.agents.base.base_agent import BaseAgent
    from aiswe.domain.ports.agent import Agent, AgentDescriptor

logger = get_logger(__name__)

AgentT = TypeVar("AgentT", bound="type[BaseAgent]")


# --- Class registry (import-time, dependency-free) ---------------------------

_AGENT_CLASSES: dict[AgentRole, type[BaseAgent]] = {}


def register_agent(cls: AgentT) -> AgentT:
    """Class decorator. Declares an agent to the framework.

    Raises:
        ConflictError: if two agents claim the same role. Failing loudly at
            import time is right -- a silently shadowed agent is a bug that
            surfaces as "why is my new agent never called?" three days later.
    """
    descriptor: AgentDescriptor = cls.descriptor  # type: ignore[attr-defined]
    role = descriptor.role

    if role in _AGENT_CLASSES and _AGENT_CLASSES[role] is not cls:
        existing = _AGENT_CLASSES[role].__name__
        msg = f"Agent role {role!r} is already claimed by {existing}."
        raise ConflictError(msg, code="duplicate_agent_role")

    _AGENT_CLASSES[role] = cls
    logger.debug("agent.class_registered", role=str(role), cls=cls.__name__)
    return cls


def registered_agent_classes() -> dict[AgentRole, type[BaseAgent]]:
    """Every agent class declared so far. Consumed by the bootstrap."""
    return dict(_AGENT_CLASSES)


def clear_agent_classes() -> None:
    """Test-only. Resets the module-level registry between test cases."""
    _AGENT_CLASSES.clear()


# --- Instance registry (runtime, dependency-injected) ------------------------


class InMemoryAgentRegistry:
    """Implements `domain.ports.agent.AgentRegistry`.

    Built once by the composition root. Immutable in spirit after startup: an
    agent appearing mid-run would make the graph non-deterministic and the
    checkpoint un-resumable.
    """

    def __init__(self, agents: list[Agent[Any]] | None = None) -> None:
        self._agents: dict[AgentRole, Agent[Any]] = {}
        for agent in agents or []:
            self.register(agent)

    def register(self, agent: Agent[Any]) -> None:
        role = agent.descriptor.role
        if role in self._agents:
            msg = f"Agent role {role!r} is already registered."
            raise ConflictError(msg, code="duplicate_agent_role")
        self._agents[role] = agent
        logger.debug("agent.registered", role=str(role))

    def get(self, role: AgentRole) -> Agent[Any]:
        try:
            return self._agents[role]
        except KeyError as exc:
            known = sorted(str(r) for r in self._agents)
            msg = f"No agent registered for role {role!r}. Known roles: {known}"
            raise AgentNotFoundError(msg, agent=str(role)) from exc

    def has(self, role: AgentRole) -> bool:
        return role in self._agents

    def all(self) -> list[Agent[Any]]:
        return list(self._agents.values())

    def routable(self) -> list[Agent[Any]]:
        """Agents the supervisor may dispatch to.

        The supervisor itself is excluded -- it cannot route to itself, and a
        model offered "supervisor" as a destination will eventually pick it.
        """
        return [
            a
            for a in self._agents.values()
            if a.descriptor.routable and a.descriptor.role != AgentRole("supervisor")
        ]

    def descriptors(self) -> list[AgentDescriptor]:
        return [a.descriptor for a in self._agents.values()]

    def routing_catalogue(self) -> str:
        """The routable agents, rendered for the supervisor's prompt.

        This is why `when_to_use` is mandatory on the descriptor: it IS the
        routing prompt. A vague `when_to_use` produces a vague router, and a
        vague router burns budget bouncing between the wrong agents.
        """
        lines: list[str] = []
        for agent in sorted(self.routable(), key=lambda a: str(a.descriptor.role)):
            d = agent.descriptor
            lines.append(f"- {d.role}: {d.description}\n    Use when: {d.when_to_use}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, role: object) -> bool:
        return role in self._agents
