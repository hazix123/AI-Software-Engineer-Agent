"""Graph builder.

Constructs the LangGraph `StateGraph` from the agent registry. This file, the
runtime, and the checkpointer are the ONLY three that import LangGraph -- so
everything else in the framework is testable without it, and LangGraph itself
stays a swappable detail rather than a load-bearing dependency.

TOPOLOGY: HUB AND SPOKE
-----------------------
    START -> supervisor -> {any registered agent} -> supervisor -> ... -> END

Every specialist returns to the supervisor. There are no direct agent-to-agent
edges, and that is deliberate: a mesh of N agents has N^2 possible edges, an
unbounded number of cycles, and no single place to enforce the budget. A hub has
N edges, one place to check every guard, and one place to answer "why did it do
that?".

THE OPEN/CLOSED PAYOFF
----------------------
The destination map for the supervisor's conditional edge is DERIVED from the
registry:

    {str(a.descriptor.role): str(a.descriptor.role) for a in registry.routable()}

Register the Security Review Agent in M7 and it becomes a node, a legal routing
destination, and a budget participant. Not one line of this file changes. That is
the whole requirement, discharged in one dict comprehension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from aiswe.agents.graph.state import AgentState
from aiswe.domain.errors import ConfigurationError
from aiswe.domain.value_objects.agent_role import FINISH, SUPERVISOR
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from aiswe.domain.ports.agent import AgentRegistry

logger = get_logger(__name__)


class GraphBuilder:
    """Assembles the agent graph from whatever is registered.

    Knows the shape of the graph. Knows nothing about which agents exist.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def build(
        self,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        interrupt_before: list[str] | None = None,
    ) -> CompiledStateGraph[AgentState, Any, Any, Any]:
        """Compile the graph.

        Args:
            checkpointer: Durable state. Without it a worker crash loses the run;
                with it, another worker resumes from the last completed node.
            interrupt_before: Nodes to halt before, for human-in-the-loop gates.
                Populated in M6 with the write-path agents.
        """
        supervisor_role = str(SUPERVISOR)

        if not self._registry.has(SUPERVISOR):
            msg = "No supervisor agent is registered. The graph has no hub."
            raise ConfigurationError(msg)

        routable = self._registry.routable()
        if not routable:
            msg = "No routable agents are registered. The graph has no spokes."
            raise ConfigurationError(msg)

        graph: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)

        # --- Nodes -----------------------------------------------------------
        # Every agent is a node. `BaseAgent.__call__` IS the node function: it
        # takes state and returns a partial update, which is exactly LangGraph's
        # node contract. No adapter, no wrapper.
        graph.add_node(supervisor_role, self._registry.get(SUPERVISOR))
        for agent in routable:
            role = str(agent.descriptor.role)
            graph.add_node(role, agent)
            logger.debug("graph.node_added", role=role)

        # --- Edges -----------------------------------------------------------
        graph.add_edge(START, supervisor_role)

        # The one conditional edge in the system. Its destination map is derived
        # from the registry -- this is the OCP seam.
        destinations: dict[str, str] = {
            str(a.descriptor.role): str(a.descriptor.role) for a in routable
        }
        destinations[str(FINISH)] = END

        graph.add_conditional_edges(
            supervisor_role,
            _route,
            destinations,
        )

        # Every specialist returns to the hub. No agent decides where work goes
        # next; it only reports what it did.
        for agent in routable:
            graph.add_edge(str(agent.descriptor.role), supervisor_role)

        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or [],
        )

        logger.info(
            "graph.compiled",
            nodes=len(routable) + 1,
            routable=sorted(destinations.keys() - {str(FINISH)}),
            checkpointed=checkpointer is not None,
        )
        return compiled

    def mermaid(self) -> str:
        """Render the topology. Useful in docs and when debugging routing."""
        lines = ["graph TD", "    START --> supervisor"]
        for agent in self._registry.routable():
            role = str(agent.descriptor.role)
            lines.append(f"    supervisor -->|routes| {role}")
            lines.append(f"    {role} --> supervisor")
        lines.append("    supervisor -->|finish| END")
        return "\n".join(lines)


def _route(state: AgentState) -> str:
    """The conditional-edge function. Reads the decision; never makes one.

    The supervisor already decided and wrote `next_agent` into state. This
    function exists only because LangGraph needs a callable on the edge -- so it
    stays a pure projection of state. Putting routing logic here would split the
    decision across two files and make it untestable without a compiled graph.
    """
    if state.done or state.next_agent is None:
        return str(FINISH)
    return state.next_agent
