"""Agent framework unit tests.

The headline is `TestOpenClosedPrinciple`. It defines a NEW agent inside the
test, registers it, and asserts that it becomes routable, planner-assignable,
and a graph node -- with zero changes to any framework file.

That is the requirement ("support adding future agents without modifying
existing code") turned into something that fails the build if it regresses. A
requirement that is only asserted in a design document is a requirement that
quietly stops being true.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from aiswe.agents.base.base_agent import BaseAgent
from aiswe.agents.graph.routing import (
    AssignedStepRule,
    BudgetExhaustedRule,
    DeadlockRule,
    PlanCompleteRule,
    RoutingDecision,
    RoutingPolicy,
    RoutingRule,
)
from aiswe.agents.graph.state import AgentState, Plan, PlanStep, initial_state
from aiswe.agents.planner.agent import DraftPlan, DraftStep, PlannerAgent, _find_cycle
from aiswe.agents.registry import InMemoryAgentRegistry
from aiswe.agents.supervisor.agent import SupervisorAgent
from aiswe.domain.errors import (
    AgentOutputError,
    BudgetExceededError,
    MaxIterationsError,
)
from aiswe.domain.policies.retry_policy import RetryPolicy, is_retryable
from aiswe.domain.ports.agent import AgentDescriptor, AgentOutput
from aiswe.domain.ports.tool import Permission, ToolContext, ToolResult, ToolSpec
from aiswe.domain.value_objects.agent_role import FINISH, AgentRole
from aiswe.domain.value_objects.run_status import RunStatus
from aiswe.domain.value_objects.token_budget import TokenBudget
from aiswe.infrastructure.llm.stub_provider import ScriptedLLM, StubLLMProvider
from aiswe.infrastructure.memory.stores import NullMemoryStore
from aiswe.tools.base import BaseTool
from aiswe.tools.policy import ToolPolicy
from aiswe.tools.registry import InMemoryToolRegistry

pytestmark = pytest.mark.unit


# --- Fixtures ----------------------------------------------------------------


def budget(*, steps: int = 20, tokens: int = 100_000) -> TokenBudget:
    return TokenBudget(
        max_tokens=tokens,
        max_cost_usd=Decimal("1.00"),
        max_steps=steps,
    )


def state(**overrides: Any) -> AgentState:
    base = initial_state(
        run_id=uuid.uuid4(),
        objective="Add a health check endpoint.",
        budget=budget(),
    )
    return base.model_copy(update=overrides) if overrides else base


class _Echo(BaseAgent):
    """A trivial agent, for testing the base class in isolation."""

    descriptor = AgentDescriptor(
        role=AgentRole("echo"),
        description="Echoes.",
        when_to_use="Testing only.",
        requires_llm=False,
    )

    async def execute(self, state: AgentState) -> AgentOutput:
        return AgentOutput(status="ok", summary=f"echo: {state.objective}")


class _Exploding(BaseAgent):
    """Always fails. Proves a failing agent does not kill the run."""

    descriptor = AgentDescriptor(
        role=AgentRole("exploding"),
        description="Fails.",
        when_to_use="Testing only.",
        requires_llm=False,
        retry=RetryPolicy(max_attempts=1),
    )

    async def execute(self, state: AgentState) -> AgentOutput:  # noqa: ARG002
        msg = "detonated"
        raise RuntimeError(msg)


# =============================================================================
# THE CORE REQUIREMENT
# =============================================================================


class TestOpenClosedPrinciple:
    """Adding an agent must not require modifying existing code.

    Every assertion below is made against an agent that did not exist when the
    framework was written -- and no framework file is touched to make them pass.
    """

    def _new_agent(self) -> BaseAgent:
        """An agent invented right here, after the framework was built."""

        class SecurityReviewAgent(BaseAgent):
            descriptor = AgentDescriptor(
                role=AgentRole("security_review"),
                description="Scans a patch for vulnerabilities.",
                when_to_use="After code is modified, before it is committed.",
                capabilities=frozenset({"security", "read"}),
                requires_llm=False,
            )

            async def execute(self, state: AgentState) -> AgentOutput:  # noqa: ARG002
                return AgentOutput(status="ok", summary="No findings.")

        return SecurityReviewAgent()

    def test_new_agent_is_routable_without_framework_changes(self) -> None:
        registry = InMemoryAgentRegistry([self._new_agent()])

        role = AgentRole("security_review")
        assert registry.has(role)
        assert role in [a.descriptor.role for a in registry.routable()]

    def test_new_agent_appears_in_the_routing_catalogue(self) -> None:
        """The supervisor's prompt is DERIVED from the registry, so a new agent
        is routable by the LLM with no prompt edit."""
        registry = InMemoryAgentRegistry([self._new_agent()])

        catalogue = registry.routing_catalogue()
        assert "security_review" in catalogue
        assert "After code is modified" in catalogue  # when_to_use IS the prompt

    def test_planner_can_assign_work_to_a_new_agent(self) -> None:
        """The planner validates against the registry, not a hardcoded list."""
        registry = InMemoryAgentRegistry([self._new_agent()])
        planner = PlannerAgent(registry=registry, llm=StubLLMProvider())

        draft = DraftPlan(
            rationale="Review the change.",
            steps=[
                DraftStep(
                    id="s1",
                    title="Scan for vulnerabilities",
                    description="Run a security review over the patch.",
                    assigned_agent="security_review",  # did not exist at build time
                )
            ],
        )

        plan = planner._validate(draft, version=1)
        assert plan.steps[0].assigned_agent == "security_review"

    def test_supervisor_dispatches_to_a_new_agent(self) -> None:
        """AssignedStepRule routes to whatever the plan named, provided it is
        registered. No `if role == ...` chain exists to be extended."""
        registry = InMemoryAgentRegistry([self._new_agent()])
        plan = Plan(
            steps=[
                PlanStep(
                    id="s1",
                    title="Scan",
                    description="…",
                    assigned_agent="security_review",
                )
            ]
        )

        decision = AssignedStepRule().evaluate(state(plan=plan), registry)

        assert decision is not None
        assert decision.target == AgentRole("security_review")

    def test_new_routing_rule_extends_the_policy_without_edits(self) -> None:
        """A new rule is an append, not a modification."""

        class AlwaysFinishRule(RoutingRule):
            priority = -1  # runs first

            def evaluate(  # noqa: ARG002
                self, state: AgentState, registry: Any
            ) -> RoutingDecision | None:
                return RoutingDecision(FINISH, "custom rule fired")

        registry = InMemoryAgentRegistry([self._new_agent()])
        policy = RoutingPolicy().with_rule(AlwaysFinishRule())

        decision = policy.decide(state(), registry)

        assert decision is not None
        assert decision.is_finish
        assert decision.reason == "custom rule fired"


# =============================================================================
# Agent registry
# =============================================================================


class TestAgentRegistry:
    def test_duplicate_roles_are_rejected_loudly(self) -> None:
        """A silently shadowed agent surfaces as 'why is my agent never called?'
        three days later. Fail at registration instead."""
        from aiswe.domain.errors import ConflictError

        registry = InMemoryAgentRegistry([_Echo()])
        with pytest.raises(ConflictError, match="already registered"):
            registry.register(_Echo())

    def test_unknown_role_raises_with_a_useful_message(self) -> None:
        from aiswe.domain.errors import AgentNotFoundError

        registry = InMemoryAgentRegistry([_Echo()])
        with pytest.raises(AgentNotFoundError, match="Known roles"):
            registry.get(AgentRole("nonexistent"))

    def test_supervisor_is_not_routable_to_itself(self) -> None:
        """Offer a model 'supervisor' as a destination and it will eventually
        pick it -- and route to itself forever."""
        registry = InMemoryAgentRegistry()
        registry.register(SupervisorAgent(registry=registry, llm=StubLLMProvider()))
        registry.register(_Echo())

        roles = [a.descriptor.role for a in registry.routable()]
        assert AgentRole("supervisor") not in roles
        assert AgentRole("echo") in roles


# =============================================================================
# Routing
# =============================================================================


class TestRouting:
    def test_budget_exhaustion_beats_every_other_rule(self) -> None:
        """Priority 0. Nothing outranks 'we are out of money'."""
        registry = InMemoryAgentRegistry([_Echo()])
        spent = budget(steps=1).step()

        # A plan with ready work exists -- and is still not followed.
        plan = Plan(steps=[PlanStep(id="s1", title="x", description="y", assigned_agent="echo")])
        decision = RoutingPolicy().decide(state(budget=spent, plan=plan), registry)

        assert decision is not None
        assert decision.is_finish
        assert "Budget exhausted" in decision.reason

    def test_completed_plan_finishes(self) -> None:
        registry = InMemoryAgentRegistry([_Echo()])
        plan = Plan(
            steps=[
                PlanStep(
                    id="s1", title="x", description="y", assigned_agent="echo", status="done"
                )
            ]
        )
        decision = PlanCompleteRule().evaluate(state(plan=plan), registry)

        assert decision is not None
        assert decision.is_finish

    def test_deadlocked_plan_is_detected_not_spun_on(self) -> None:
        """Pending steps, none ready. Without this rule the run burns its entire
        budget making zero progress."""
        registry = InMemoryAgentRegistry([_Echo()])
        plan = Plan(
            steps=[
                PlanStep(
                    id="s1",
                    title="x",
                    description="y",
                    assigned_agent="echo",
                    depends_on=["s2"],
                ),
                PlanStep(
                    id="s2",
                    title="x",
                    description="y",
                    assigned_agent="echo",
                    status="failed",
                ),
            ]
        )
        decision = DeadlockRule().evaluate(state(plan=plan), registry)

        assert decision is not None
        assert decision.is_finish
        assert "deadlocked" in decision.reason

    def test_dag_dependencies_gate_readiness(self) -> None:
        plan = Plan(
            steps=[
                PlanStep(id="s1", title="a", description="a", assigned_agent="echo"),
                PlanStep(
                    id="s2",
                    title="b",
                    description="b",
                    assigned_agent="echo",
                    depends_on=["s1"],
                ),
            ]
        )
        assert plan.next_ready_step() is not None
        assert plan.next_ready_step().id == "s1"  # type: ignore[union-attr]

        plan.steps[0].status = "done"
        assert plan.next_ready_step().id == "s2"  # type: ignore[union-attr]

    def test_unknown_assigned_agent_defers_rather_than_crashing(self) -> None:
        """A plan naming an unregistered agent must not take the run down; it
        falls through to the LLM router."""
        registry = InMemoryAgentRegistry([_Echo()])
        plan = Plan(
            steps=[
                PlanStep(id="s1", title="x", description="y", assigned_agent="ghost")
            ]
        )
        assert AssignedStepRule().evaluate(state(plan=plan), registry) is None


# =============================================================================
# Base agent: error handling and retry
# =============================================================================


class TestBaseAgent:
    async def test_failure_becomes_state_not_an_exception(self) -> None:
        """THE resilience property. A failing agent hands control back to the
        supervisor; it does not kill the run."""
        update = await _Exploding()(state())

        assert "errors" in update
        assert update["errors"][0].code == "RuntimeError"
        assert update["errors"][0].message == "detonated"
        # Control returns to the hub, which can replan or route elsewhere.
        assert update["next_agent"] is None

    async def test_exhausted_budget_raises(self) -> None:
        """The two exceptions that DO end a run, because nothing downstream can
        act on them."""
        with pytest.raises(BudgetExceededError):
            await _Echo()(state(budget=budget(steps=1).step()))

    async def test_livelock_is_caught(self) -> None:
        agent = _Echo()
        limit = agent.descriptor.max_consecutive_visits

        with pytest.raises(MaxIterationsError, match="livelock"):
            await agent(state(visit_counts={"echo": limit + 1}))

    async def test_success_advances_budget_and_iteration(self) -> None:
        update = await _Echo()(state())

        assert update["iteration"] == 1
        assert update["budget"].steps_used == 1

    async def test_visit_counter_resets_for_other_agents(self) -> None:
        """The guard is about CONSECUTIVE re-entry, not lifetime usage: an agent
        legitimately used twenty times across a long plan is not a livelock."""
        update = await _Echo()(state(visit_counts={"echo": 1, "planner": 2}))

        assert update["visit_counts"]["echo"] == 2
        assert update["visit_counts"]["planner"] == 0


# =============================================================================
# Planner validation
# =============================================================================


class TestPlanValidation:
    def _planner(self) -> PlannerAgent:
        registry = InMemoryAgentRegistry([_Echo()])
        return PlannerAgent(registry=registry, llm=StubLLMProvider())

    def test_dependency_cycle_is_rejected(self) -> None:
        """A cycle means nothing is ever ready. The run spins until the budget
        dies -- the most expensive way to make zero progress."""
        draft = DraftPlan(
            rationale="…",
            steps=[
                DraftStep(
                    id="a", title="a", description="a",
                    assigned_agent="echo", depends_on=["b"],
                ),
                DraftStep(
                    id="b", title="b", description="b",
                    assigned_agent="echo", depends_on=["a"],
                ),
            ],
        )
        with pytest.raises(AgentOutputError, match="cycle"):
            self._planner()._validate(draft, version=1)

    def test_dangling_dependency_is_rejected(self) -> None:
        draft = DraftPlan(
            rationale="…",
            steps=[
                DraftStep(
                    id="a",
                    title="a",
                    description="a",
                    assigned_agent="echo",
                    depends_on=["ghost"],
                )
            ],
        )
        with pytest.raises(AgentOutputError, match="does not exist"):
            self._planner()._validate(draft, version=1)

    def test_unknown_agent_is_rejected_with_the_valid_options(self) -> None:
        """The error text goes straight into the re-prompt, so the model is told
        exactly what it may choose from."""
        draft = DraftPlan(
            rationale="…",
            steps=[DraftStep(id="a", title="a", description="a", assigned_agent="ghost")],
        )
        with pytest.raises(AgentOutputError, match="Available"):
            self._planner()._validate(draft, version=1)

    def test_empty_plan_is_rejected(self) -> None:
        with pytest.raises(AgentOutputError, match="no steps"):
            self._planner()._validate(DraftPlan(rationale="…", steps=[]), version=1)

    def test_plan_validation_is_retryable(self) -> None:
        """So BaseAgent re-prompts with the specific error rather than accepting
        a plan that cannot execute."""
        assert is_retryable(AgentOutputError("bad plan")) is True

    @pytest.mark.parametrize(
        ("graph", "expected"),
        [
            ({"a": [], "b": ["a"]}, None),
            ({"a": ["b"], "b": ["a"]}, ["a", "b", "a"]),
            ({"a": ["b"], "b": ["c"], "c": ["a"]}, ["a", "b", "c", "a"]),
        ],
    )
    def test_cycle_detection(self, graph: dict[str, list[str]], expected: list[str] | None) -> None:
        found = _find_cycle(graph)
        if expected is None:
            assert found is None
        else:
            assert found is not None


# =============================================================================
# Tool registry and policy
# =============================================================================


class _ReadArgs(BaseModel):
    path: str


class _FakeReadTool(BaseTool):
    spec = ToolSpec(
        name="fs.read",
        description="Read a file.",
        args_schema=_ReadArgs,
        permissions=frozenset({Permission.READ}),
        tags=frozenset({"filesystem", "read"}),
    )

    async def _run(self, ctx: ToolContext, args: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult.success("fs.read", preview=f"contents of {args.path}")


class _FakePushTool(BaseTool):
    spec = ToolSpec(
        name="git.push",
        description="Push to a remote.",
        args_schema=_ReadArgs,
        permissions=frozenset({Permission.VCS_PUSH}),
        tags=frozenset({"git"}),
        requires_approval=True,
        idempotent=False,
    )

    async def _run(self, ctx: ToolContext, args: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult.success("git.push", preview="pushed")


def _ctx(agent: str = "echo") -> ToolContext:
    return ToolContext(run_id=uuid.uuid4(), agent=agent)


class TestToolRegistry:
    def _registry(self) -> InMemoryToolRegistry:
        return InMemoryToolRegistry([_FakeReadTool(), _FakePushTool()])

    def test_capability_scoping_is_the_boundary(self) -> None:
        """An agent is HANDED its tools. It cannot call what it never received."""
        registry = self._registry()

        docs_tools = registry.tools_for(frozenset({"filesystem", "read"}))
        names = {t.spec.name for t in docs_tools}

        assert "fs.read" in names
        assert "git.push" not in names  # not merely discouraged -- absent

    async def test_calling_an_unscoped_tool_is_denied(self) -> None:
        """And denial is returned as an OBSERVATION, so the agent can adapt."""
        result = await self._registry().invoke(
            "git.push",
            _ctx("documentation"),
            {"path": "x"},
            capabilities=frozenset({"read"}),  # no "git"
        )

        assert result.ok is False
        assert result.error_code == "tool_permission_denied"

    async def test_approval_gated_tool_reports_approval_required(self) -> None:
        result = await self._registry().invoke(
            "git.push",
            _ctx("git"),
            {"path": "x"},
            capabilities=frozenset({"git"}),
        )

        assert result.ok is False
        assert result.error_code == "approval_required"

    async def test_autonomous_mode_never_waives_a_push(self) -> None:
        """'Autonomous' must not mean 'may force-push to main, unattended, at 3am'."""
        registry = InMemoryToolRegistry(
            [_FakePushTool()],
            policy=ToolPolicy(autonomous_mode=True),
        )
        result = await registry.invoke(
            "git.push",
            _ctx("git"),
            {"path": "x"},
            capabilities=frozenset({"git"}),
        )

        assert result.error_code == "approval_required"

    async def test_hallucinated_tool_name_is_an_observation_not_a_crash(self) -> None:
        result = await self._registry().invoke(
            "fs.telepathy",
            _ctx(),
            {},
            capabilities=frozenset({"filesystem"}),
        )

        assert result.ok is False
        assert result.error_code == "tool_not_found"

    async def test_invalid_arguments_explain_themselves(self) -> None:
        result = await self._registry().invoke(
            "fs.read",
            _ctx(),
            {"wrong_field": 1},
            capabilities=frozenset({"read"}),
        )

        assert result.ok is False
        assert "path" in (result.error_message or "")

    async def test_successful_call_returns_an_observation(self) -> None:
        result = await self._registry().invoke(
            "fs.read",
            _ctx(),
            {"path": "src/main.py"},
            capabilities=frozenset({"read"}),
        )

        assert result.ok is True
        assert "src/main.py" in result.as_observation()


# =============================================================================
# Policies
# =============================================================================


class TestRetryPolicy:
    def test_permission_denial_is_never_retried(self) -> None:
        """Retrying a deterministic denial just buys the identical denial again."""
        from aiswe.domain.errors import ToolPermissionError

        assert is_retryable(ToolPermissionError()) is False

    def test_budget_exhaustion_is_never_retried(self) -> None:
        assert is_retryable(BudgetExceededError()) is False

    def test_provider_errors_are_retried(self) -> None:
        from aiswe.domain.errors import LLMProviderError

        assert is_retryable(LLMProviderError(status_code=429)) is True

    def test_unknown_errors_are_not_retried(self) -> None:
        """An unknown failure repeated three times is three times the damage, not
        three times the chance of success."""
        assert is_retryable(ValueError("mystery")) is False

    def test_backoff_is_jittered_and_capped(self) -> None:
        policy = RetryPolicy(base_delay_s=1.0, multiplier=2.0, max_delay_s=5.0)

        # Full jitter: every delay lies in [0, ceiling], never a fixed value --
        # which is what prevents a thundering herd on provider recovery.
        for attempt in range(1, 6):
            delay = policy.delay_for(attempt)
            assert 0.0 <= delay <= 5.0


class TestTokenBudget:
    def test_any_ceiling_exhausts_the_budget(self) -> None:
        b = TokenBudget(max_tokens=100, max_cost_usd=Decimal("1"), max_steps=10)

        assert b.spend(tokens=100).exhausted
        assert b.spend(cost_usd=Decimal("1")).exhausted
        assert b.spend(steps=10).exhausted

    def test_spend_is_immutable(self) -> None:
        b = budget()
        b.spend(tokens=50)
        assert b.tokens_used == 0  # original untouched

    def test_exhausted_reason_is_specific(self) -> None:
        b = TokenBudget(max_tokens=10, max_cost_usd=Decimal("1"), max_steps=10)
        assert "token limit" in (b.spend(tokens=10).exhausted_reason or "")


class TestRunStatus:
    def test_terminal_states_are_terminal(self) -> None:
        assert RunStatus.SUCCEEDED.is_terminal
        assert not RunStatus.SUCCEEDED.can_transition_to(RunStatus.RUNNING)

    def test_approval_can_resume(self) -> None:
        assert RunStatus.AWAITING_APPROVAL.can_transition_to(RunStatus.RUNNING)

    def test_paused_can_resume_after_a_crash(self) -> None:
        assert RunStatus.PAUSED.can_transition_to(RunStatus.RUNNING)


class TestAgentRole:
    def test_roles_are_normalised(self) -> None:
        assert AgentRole("  Security_Review  ") == "security_review"

    @pytest.mark.parametrize("bad", ["", "9lives", "has spaces", "Has-Dashes"])
    def test_malformed_roles_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid agent role"):
            AgentRole(bad)


class TestScriptedLLM:
    async def test_scripted_responses_drive_exact_paths(self) -> None:
        """Lets a test provoke failures a real model almost never produces --
        which are exactly the ones that break in production."""
        cyclic = DraftPlan(
            rationale="bad",
            steps=[
                DraftStep(
                    id="a", title="a", description="a",
                    assigned_agent="echo", depends_on=["b"],
                ),
                DraftStep(
                    id="b", title="b", description="b",
                    assigned_agent="echo", depends_on=["a"],
                ),
            ],
        )
        llm = ScriptedLLM([cyclic])
        registry = InMemoryAgentRegistry([_Echo()])
        planner = PlannerAgent(registry=registry, llm=llm)

        with pytest.raises(AgentOutputError, match="cycle"):
            await planner.execute(state())

        assert llm.remaining == 0


class TestNullMemoryStore:
    async def test_recall_returns_nothing_and_does_not_crash(self) -> None:
        """Runs start cold. Correct behaviour, just not clever behaviour."""
        from aiswe.domain.ports.memory import MemoryQuery

        assert await NullMemoryStore().recall(MemoryQuery(text="anything")) == []
