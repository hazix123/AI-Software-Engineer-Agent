"""Agent bootstrap.

Where the open/closed promise is actually cashed.

Three steps:
    1. IMPORT   agent modules, which triggers their `@register_agent` decorators.
    2. BUILD    each registered class, injecting its dependencies.
    3. RETURN   a wired registry for the graph builder.

ADDING AN AGENT IN A LATER MILESTONE
------------------------------------
Append one line to `_AGENT_MODULES`. That is the whole integration.

The agent then becomes a graph node, a legal routing destination, a budget
participant, a planner-assignable target, and is scoped to exactly the tools its
descriptor declares -- with no edit to the builder, the router, the supervisor,
the planner, or the state.

Why explicit imports rather than filesystem auto-discovery? Because auto-discovery
makes the set of live agents depend on what happens to be on disk. Import order
becomes load-bearing, a stray file changes behaviour, and a typo produces silence
rather than an error. An explicit list is one line longer and infinitely easier
to reason about at 3am.

INJECTION: some agents need constructor arguments beyond the common three (the
supervisor needs the registry; the memory agent needs a store). `_FACTORIES` maps
role -> builder. An agent needing nothing special is built generically and does
not appear there at all.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from aiswe.agents.registry import InMemoryAgentRegistry, registered_agent_classes
from aiswe.domain.value_objects.agent_role import MEMORY, PLANNER, SUPERVISOR, AgentRole
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.agents.base.base_agent import BaseAgent
    from aiswe.agents.graph.routing import RoutingPolicy
    from aiswe.domain.ports.event_bus import EventBus
    from aiswe.domain.ports.llm import LLMProvider
    from aiswe.domain.ports.memory import MemoryStore
    from aiswe.domain.ports.tool import ToolRegistry

logger = get_logger(__name__)


#: Agent modules to import. Importing runs their @register_agent decorators.
#:
#: >>> ADD NEW AGENTS HERE. THIS IS THE ONLY FILE THAT CHANGES. <<<
_AGENT_MODULES: list[str] = [
    "aiswe.agents.supervisor.agent",
    "aiswe.agents.planner.agent",
    "aiswe.agents.memory.agent",
    # --- Later milestones -------------------------------------------------
    # "aiswe.agents.indexing.agent",       # M4
    # "aiswe.agents.code_analysis.agent",  # M5
    # "aiswe.agents.refactoring.agent",    # M6
    # "aiswe.agents.testing.agent",        # M6
    # "aiswe.agents.git.agent",            # M6
    # "aiswe.agents.security.agent",       # M7
    # "aiswe.agents.documentation.agent",  # M7
]


def import_agents() -> None:
    """Import every agent module, populating the class registry."""
    for module in _AGENT_MODULES:
        importlib.import_module(module)
    logger.debug("agents.imported", count=len(_AGENT_MODULES))


def build_agent_registry(
    *,
    llm: LLMProvider,
    memory_store: MemoryStore,
    tools: ToolRegistry | None = None,
    event_bus: EventBus | None = None,
    routing_policy: RoutingPolicy | None = None,
) -> InMemoryAgentRegistry:
    """Discover, construct, and wire every agent.

    Built in two passes because the supervisor needs the registry that contains
    the other agents -- it cannot be constructed until they exist. A chicken/egg
    that is trivial to resolve, and confusing if you do not name it.
    """
    import_agents()

    classes = registered_agent_classes()
    registry = InMemoryAgentRegistry()

    # --- Pass 1: everything except the supervisor ----------------------------
    specialists: dict[AgentRole, BaseAgent] = {}
    for role, cls in classes.items():
        if role == SUPERVISOR:
            continue
        factory = _FACTORIES.get(role, _generic_factory)
        agent = factory(
            cls,
            llm=llm,
            tools=tools,
            event_bus=event_bus,
            memory_store=memory_store,
            registry=registry,
        )
        specialists[role] = agent
        registry.register(agent)

    # --- Pass 2: the supervisor, which now sees a populated registry ----------
    supervisor_cls = classes.get(SUPERVISOR)
    if supervisor_cls is None:
        from aiswe.domain.errors import ConfigurationError

        msg = "No supervisor agent is registered. The graph would have no hub."
        raise ConfigurationError(msg)

    supervisor = supervisor_cls(  # type: ignore[call-arg]
        registry=registry,
        policy=routing_policy,
        llm=llm,
        event_bus=event_bus,
    )
    registry.register(supervisor)

    logger.info(
        "agents.registered",
        count=len(registry),
        roles=sorted(str(d.role) for d in registry.descriptors()),
    )
    return registry


# --- Factories ---------------------------------------------------------------
# Only agents with non-standard constructor needs appear here.


def _generic_factory(cls: type[BaseAgent], **deps: object) -> BaseAgent:
    """The common case: llm + tools + event_bus."""
    return cls(  # type: ignore[call-arg]
        llm=deps["llm"],
        tools=deps["tools"],
        event_bus=deps["event_bus"],
    )


def _planner_factory(cls: type[BaseAgent], **deps: object) -> BaseAgent:
    """The planner reads the registry to learn which agents it may assign work to.

    This is what lets a plan target the Security Agent the moment it registers,
    with no change to the planner's code.
    """
    return cls(  # type: ignore[call-arg]
        registry=deps["registry"],
        llm=deps["llm"],
        event_bus=deps["event_bus"],
    )


def _memory_factory(cls: type[BaseAgent], **deps: object) -> BaseAgent:
    return cls(  # type: ignore[call-arg]
        store=deps["memory_store"],
        llm=deps["llm"],
        event_bus=deps["event_bus"],
    )


_FACTORIES: dict[AgentRole, Callable[..., BaseAgent]] = {
    PLANNER: _planner_factory,
    MEMORY: _memory_factory,
}
