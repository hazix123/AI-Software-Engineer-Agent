"""Routing policy.

A Chain of Responsibility. Rules are evaluated in priority order; the first one
that returns a decision wins.

WHY DETERMINISTIC RULES COME FIRST
----------------------------------
Most routing decisions are FACTS, not judgement calls:

    budget exhausted            -> finish.        Not a matter of opinion.
    plan step assigned to X     -> X.             The planner already decided.
    approval pending            -> halt.          Non-negotiable.
    plan complete               -> finish.

Asking a language model to re-derive these costs money, adds latency, and
introduces a chance -- small, but never zero -- that it gets one wrong and
routes a run with an exhausted budget straight back into an expensive agent.

So the LLM is the LAST rule, not the first. It is consulted only when the plan
genuinely does not determine the next hop. In a well-planned run that is rare,
which is exactly the point.

OPEN/CLOSED
-----------
A new rule is a new class appended to a list. Nothing existing is edited. The
alternative -- one long `if/elif` in the supervisor -- becomes unreadable at
rule six and a merge-conflict magnet at rule ten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiswe.domain.value_objects.agent_role import FINISH, AgentRole
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.agents.graph.state import AgentState
    from aiswe.domain.ports.agent import AgentRegistry

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Where to go next, and -- crucially -- why.

    `reason` is not decoration. When a user asks "why did it stop?", this is the
    answer, and it is the difference between a debuggable system and a magic one.
    """

    target: AgentRole
    reason: str
    deterministic: bool = True

    @property
    def is_finish(self) -> bool:
        return self.target == FINISH


class RoutingRule(ABC):
    """One routing consideration. Returns a decision, or None to defer."""

    #: Lower runs first. Safety rules must precede progress rules.
    priority: int = 100

    @abstractmethod
    def evaluate(self, state: AgentState, registry: AgentRegistry) -> RoutingDecision | None:
        """Return a decision, or None to pass to the next rule."""

    @property
    def name(self) -> str:
        return type(self).__name__


# --- Safety rules (priority 0-19) --------------------------------------------
# These must run before anything that could spend money.


class BudgetExhaustedRule(RoutingRule):
    """Out of tokens, money, or steps. Stop. Not negotiable."""

    priority = 0

    def evaluate(  # noqa: ARG002
        self, state: AgentState, registry: AgentRegistry
    ) -> RoutingDecision | None:
        if state.budget.exhausted:
            reason = state.budget.exhausted_reason or "budget exhausted"
            return RoutingDecision(FINISH, f"Budget exhausted: {reason}")
        return None


class ApprovalPendingRule(RoutingRule):
    """A human gate is open. The graph must halt, not route around it."""

    priority = 1

    def evaluate(  # noqa: ARG002
        self, state: AgentState, registry: AgentRegistry
    ) -> RoutingDecision | None:
        if state.awaiting_approval is not None:
            return RoutingDecision(
                FINISH,
                f"Awaiting human approval: {state.awaiting_approval.kind}",
            )
        return None


class ExplicitDoneRule(RoutingRule):
    """An agent asserted the run is complete."""

    priority = 2

    def evaluate(  # noqa: ARG002
        self, state: AgentState, registry: AgentRegistry
    ) -> RoutingDecision | None:
        if state.done:
            return RoutingDecision(FINISH, "An agent marked the run complete.")
        return None


class FatalErrorRule(RoutingRule):
    """A non-retryable failure with no plan to fall back on.

    If a plan exists, we defer: the planner may be able to route around the
    failure. If there is no plan, there is nothing to replan, so we stop rather
    than loop on a guaranteed failure.
    """

    priority = 3

    def evaluate(  # noqa: ARG002
        self, state: AgentState, registry: AgentRegistry
    ) -> RoutingDecision | None:
        err = state.last_error
        if err and not err.retryable and state.plan is None:
            return RoutingDecision(FINISH, f"Unrecoverable failure in {err.agent}: {err.message}")
        return None


# --- Progress rules (priority 20-79) -----------------------------------------


class NeedsPlanRule(RoutingRule):
    """No plan yet. Plan before acting -- always."""

    priority = 20

    def evaluate(self, state: AgentState, registry: AgentRegistry) -> RoutingDecision | None:
        planner = AgentRole("planner")
        if state.plan is None and registry.has(planner):
            return RoutingDecision(planner, "No plan exists yet.")
        return None


class ReplanOnFailureRule(RoutingRule):
    """A step failed. Replan rather than blindly retrying the same step.

    Retrying an identical failing step with an identical prompt produces an
    identical failure. Replanning is the only move that changes the outcome.
    """

    priority = 30

    def evaluate(self, state: AgentState, registry: AgentRegistry) -> RoutingDecision | None:
        planner = AgentRole("planner")
        if not state.plan or not registry.has(planner):
            return None
        if state.plan.has_failure and state.visits(str(planner)) < 2:
            return RoutingDecision(planner, "A plan step failed; replanning.")
        return None


class PlanCompleteRule(RoutingRule):
    """Every step done. Finish."""

    priority = 40

    def evaluate(  # noqa: ARG002
        self, state: AgentState, registry: AgentRegistry
    ) -> RoutingDecision | None:
        if state.plan and state.plan.is_complete:
            return RoutingDecision(FINISH, "All plan steps are complete.")
        return None


class AssignedStepRule(RoutingRule):
    """The plan named the agent for the next ready step. Obey it.

    This is the rule that keeps most runs off the LLM router entirely: a good
    plan already answered the routing question.
    """

    priority = 50

    def evaluate(self, state: AgentState, registry: AgentRegistry) -> RoutingDecision | None:
        if not state.plan:
            return None

        step = state.plan.next_ready_step()
        if step is None or step.assigned_agent is None:
            return None

        role = AgentRole(step.assigned_agent)
        if not registry.has(role):
            logger.warning(
                "routing.unknown_assigned_agent",
                agent=step.assigned_agent,
                step=step.id,
            )
            return None  # defer to the LLM router rather than crash

        return RoutingDecision(role, f"Plan step {step.id!r} is assigned to {role}.")


class DeadlockRule(RoutingRule):
    """A plan with pending steps but nothing ready to run.

    Means the dependency graph is unsatisfiable -- usually a planner that emitted
    a cycle, or a step depending on one that failed. Stop; do not spin.
    """

    priority = 60

    def evaluate(  # noqa: ARG002
        self, state: AgentState, registry: AgentRegistry
    ) -> RoutingDecision | None:
        if not state.plan or state.plan.is_complete:
            return None
        pending = [s for s in state.plan.steps if s.status == "pending"]
        if pending and state.plan.next_ready_step() is None:
            blocked = ", ".join(s.id for s in pending)
            return RoutingDecision(
                FINISH,
                f"Plan is deadlocked: steps [{blocked}] have unsatisfiable dependencies.",
            )
        return None


# --- The policy --------------------------------------------------------------

DEFAULT_RULES: list[RoutingRule] = [
    BudgetExhaustedRule(),
    ApprovalPendingRule(),
    ExplicitDoneRule(),
    FatalErrorRule(),
    NeedsPlanRule(),
    ReplanOnFailureRule(),
    PlanCompleteRule(),
    AssignedStepRule(),
    DeadlockRule(),
]


class RoutingPolicy:
    """Evaluates rules in priority order. First decision wins.

    Returning None means "no rule could decide" -- and only then does the
    supervisor pay for an LLM call.
    """

    def __init__(self, rules: list[RoutingRule] | None = None) -> None:
        self._rules = sorted(rules or DEFAULT_RULES, key=lambda r: r.priority)

    def decide(self, state: AgentState, registry: AgentRegistry) -> RoutingDecision | None:
        for rule in self._rules:
            decision = rule.evaluate(state, registry)
            if decision is not None:
                logger.debug(
                    "routing.decided",
                    rule=rule.name,
                    target=str(decision.target),
                    reason=decision.reason,
                )
                return decision
        return None

    def with_rule(self, rule: RoutingRule) -> RoutingPolicy:
        """Extend without mutating. New rule, new policy, existing code untouched."""
        return RoutingPolicy([*self._rules, rule])

    @property
    def rules(self) -> list[RoutingRule]:
        return list(self._rules)
