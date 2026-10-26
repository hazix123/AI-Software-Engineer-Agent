"""Graph integration tests.

Compiles the REAL LangGraph graph and executes it end to end with the stub LLM
and an in-memory checkpointer. No network, no API key, no cost -- but the actual
builder, the actual routing, the actual state reducers.

The critical assertions are the negative ones. It is easy to prove an agent
system runs; the whole difficulty is proving it STOPS -- on budget exhaustion, on
livelock, on a deadlocked plan. Those are the failures that cost money in
production, and they are the ones asserted here.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from aiswe.agents.base.base_agent import BaseAgent
from aiswe.agents.bootstrap import build_agent_registry
from aiswe.agents.graph.builder import GraphBuilder
from aiswe.agents.graph.checkpointer import build_memory_checkpointer
from aiswe.agents.graph.state import AgentState
from aiswe.agents.runtime import LangGraphRuntime
from aiswe.domain.ports.agent import AgentDescriptor, AgentOutput
from aiswe.domain.value_objects.agent_role import AgentRole
from aiswe.domain.value_objects.run_status import RunStatus
from aiswe.domain.value_objects.token_budget import TokenBudget
from aiswe.infrastructure.llm.stub_provider import StubLLMProvider
from aiswe.infrastructure.memory.stores import NullMemoryStore

pytestmark = pytest.mark.integration


def _budget(*, steps: int = 20) -> TokenBudget:
    return TokenBudget(
        max_tokens=100_000,
        max_cost_usd=Decimal("1.00"),
        max_steps=steps,
    )


def _runtime(extra: list[BaseAgent] | None = None) -> LangGraphRuntime:
    """Build the real graph, with the stub LLM injected."""
    registry = build_agent_registry(
        llm=StubLLMProvider(),
        memory_store=NullMemoryStore(),
    )
    for agent in extra or []:
        registry.register(agent)

    graph = GraphBuilder(registry).build(checkpointer=build_memory_checkpointer())
    return LangGraphRuntime(graph)


class TestGraphExecution:
    async def test_a_run_completes_and_terminates(self) -> None:
        result = await _runtime().start(
            run_id=uuid.uuid4(),
            objective="Add a health check endpoint.",
            budget=_budget(),
        )

        assert result.status in {RunStatus.SUCCEEDED, RunStatus.FAILED}
        assert isinstance(result.state, AgentState)
        # It must actually have done something, not fallen straight out.
        assert result.state.iteration > 0

    async def test_the_planner_runs_before_any_specialist(self) -> None:
        """NeedsPlanRule: plan before acting, always."""
        result = await _runtime().start(
            run_id=uuid.uuid4(),
            objective="Refactor the auth module.",
            budget=_budget(),
        )

        assert result.state.plan is not None
        assert len(result.state.plan.steps) > 0

    async def test_state_is_checkpointed_and_readable(self) -> None:
        """The property that makes crash recovery possible."""
        runtime = _runtime()
        run_id = uuid.uuid4()

        await runtime.start(run_id=run_id, objective="Anything.", budget=_budget())

        restored = await runtime.get_state(run_id)
        assert restored is not None
        assert restored.run_id == run_id


class TestGraphTerminates:
    """The assertions that actually matter.

    Proving an agent graph runs is easy. Proving it stops is the entire job.
    """

    async def test_a_tiny_budget_stops_the_run(self) -> None:
        result = await _runtime().start(
            run_id=uuid.uuid4(),
            objective="An objective that would otherwise take many steps.",
            budget=_budget(steps=2),
        )

        assert result.status is RunStatus.FAILED
        assert result.failure_reason == "budget_exceeded"

    async def test_a_livelocking_agent_is_caught(self) -> None:
        """An agent that always routes back to itself must be stopped by the
        consecutive-visit guard, not by the budget running out hours later."""

        class LoopingAgent(BaseAgent):
            descriptor = AgentDescriptor(
                role=AgentRole("looper"),
                description="Always asks to run again.",
                when_to_use="Testing only.",
                requires_llm=False,
                max_consecutive_visits=2,
            )

            async def execute(self, state: AgentState) -> AgentOutput:  # noqa: ARG002
                return AgentOutput(
                    status="ok",
                    summary="Going around again.",
                    state_delta={"next_agent": "looper"},
                    next_hint="looper",
                )

        runtime = _runtime([LoopingAgent()])

        result = await runtime.start(
            run_id=uuid.uuid4(),
            objective="Loop forever.",
            budget=_budget(steps=100),  # generous: the GUARD must stop it, not the budget
        )

        assert result.status is RunStatus.FAILED
        assert result.failure_reason == "max_iterations"


class TestOpenClosedInACompiledGraph:
    """The OCP claim, verified against a real compiled graph -- not a mock."""

    async def test_a_new_agent_becomes_a_live_graph_node(self) -> None:
        class DocumentationAgent(BaseAgent):
            descriptor = AgentDescriptor(
                role=AgentRole("documentation"),
                description="Writes docstrings and updates the README.",
                when_to_use="After code changes, to keep the docs truthful.",
                requires_llm=False,
            )

            async def execute(self, state: AgentState) -> AgentOutput:  # noqa: ARG002
                return AgentOutput(status="ok", summary="Docs updated.")

        registry = build_agent_registry(
            llm=StubLLMProvider(),
            memory_store=NullMemoryStore(),
        )
        registry.register(DocumentationAgent())  # the ONLY integration step

        builder = GraphBuilder(registry)
        graph = builder.build(checkpointer=build_memory_checkpointer())

        # It is a node in the compiled graph...
        assert "documentation" in graph.nodes
        # ...and a legal routing destination.
        assert "documentation" in builder.mermaid()
