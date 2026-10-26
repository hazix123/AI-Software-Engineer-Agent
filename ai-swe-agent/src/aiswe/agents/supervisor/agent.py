"""Supervisor agent.

The only component with routing authority. Agents may *suggest* a next hop via
`AgentOutput.next_hint`; the supervisor decides. If agents could command each
other, two of them would eventually command each other in a cycle and the run
would burn its budget bouncing.

DECISION ORDER
--------------
1. Deterministic rules (`RoutingPolicy`). Free, instant, auditable.
2. An agent's `next_hint`, but only if the target is actually registered.
3. A constrained LLM call -- the last resort.

The LLM is handed a CLOSED enum of registered roles. It cannot invent
`"code_reviewer"` and strand the graph on a node that does not exist. If it
somehow returns something unroutable anyway, we fall back to the planner rather
than crash: a supervisor that dies on a bad model response is a supervisor that
takes the run with it.

The supervisor is NOT routable (`routable=False`): it can never be a target,
only a source. Otherwise a model offered "supervisor" in its list of
destinations will, sooner or later, choose it -- and route to itself forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from aiswe.agents.base.base_agent import BaseAgent
from aiswe.agents.graph.routing import RoutingDecision, RoutingPolicy
from aiswe.agents.registry import register_agent
from aiswe.domain.errors import RoutingError
from aiswe.domain.ports.agent import AgentDescriptor, AgentOutput
from aiswe.domain.ports.llm import Message as LLMMessage
from aiswe.domain.value_objects.agent_role import FINISH, SUPERVISOR, AgentRole
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.agents.graph.state import AgentState
    from aiswe.domain.ports.agent import AgentRegistry
    from aiswe.domain.ports.event_bus import EventBus
    from aiswe.domain.ports.llm import LLMProvider

logger = get_logger(__name__)


class RoutingChoice(BaseModel):
    """The supervisor's constrained output schema."""

    next_agent: str = Field(description="Exactly one role from the provided list, or 'finish'.")
    reason: str = Field(description="One sentence. Why this agent, now.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


_SYSTEM = """\
You are the Supervisor of a multi-agent software engineering system.

Your ONLY job is to choose which agent acts next. You do not write code, plan \
work, or analyse anything yourself. You route.

Available agents:
{catalogue}

You may also return "finish" when the objective is met, when no further \
progress is possible, or when continuing would waste budget.

Rules:
- Choose exactly one agent from the list above, or "finish".
- Prefer the agent whose "Use when" most precisely matches the current \
situation.
- If the last agent failed, do NOT immediately route back to it. Route to the \
planner to find another path.
- If the objective is already satisfied, return "finish". Do not manufacture \
extra work.
"""

_USER = """\
Objective:
{objective}

Plan status:
{plan_status}

Recent activity:
{recent}

Errors so far:
{errors}

Budget: {steps_left} steps remaining, {tokens_left} tokens remaining.

Which agent should act next?
"""


@register_agent
class SupervisorAgent(BaseAgent):
    """Routes the run. Owns no domain work of its own."""

    descriptor = AgentDescriptor(
        role=SUPERVISOR,
        description="Routes work to the appropriate specialist agent.",
        when_to_use="Never routed to directly; the supervisor is the graph's hub.",
        capabilities=frozenset(),  # holds NO tools. It routes; it does not act.
        routable=False,
        requires_llm=True,
        max_consecutive_visits=50,  # the hub is re-entered constantly; that is normal
    )

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        policy: RoutingPolicy | None = None,
        llm: LLMProvider | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        super().__init__(llm=llm, event_bus=event_bus)
        self._registry = registry
        self._policy = policy or RoutingPolicy()

    async def execute(self, state: AgentState) -> AgentOutput:
        decision = self._policy.decide(state, self._registry)

        if decision is None:
            decision = self._from_hint(state) or await self._ask_llm(state)

        logger.info(
            "supervisor.routed",
            target=str(decision.target),
            reason=decision.reason,
            deterministic=decision.deterministic,
        )

        delta: dict[str, Any] = {"next_agent": str(decision.target)}

        if decision.is_finish:
            delta["done"] = True
            delta["final_summary"] = decision.reason
        elif state.plan:
            # Mark the step we are dispatching as in-progress, so a crash mid-step
            # is distinguishable from a step never started.
            step = state.plan.next_ready_step()
            if step is not None:
                delta["current_step_id"] = step.id

        return AgentOutput(
            status="ok",
            summary=f"-> {decision.target}: {decision.reason}",
            state_delta=delta,
            confidence=1.0 if decision.deterministic else 0.7,
        )

    # --- Tier 2: an agent's hint ---------------------------------------------

    def _from_hint(self, state: AgentState) -> RoutingDecision | None:
        """Honour an agent's suggestion -- but verify it first.

        A hint is advice from a component that just did real work and may know
        something the rules do not. It is still only advice: an unregistered or
        unroutable target is discarded rather than trusted.
        """
        hint = state.next_agent
        if not hint:
            return None

        try:
            role = AgentRole(hint)
        except ValueError:
            logger.warning("supervisor.malformed_hint", hint=hint)
            return None

        if role == FINISH:
            return RoutingDecision(FINISH, "An agent signalled completion.", deterministic=True)

        if not self._registry.has(role):
            logger.warning("supervisor.unknown_hint", hint=hint)
            return None

        return RoutingDecision(role, f"Handoff hint from the previous agent -> {role}.")

    # --- Tier 3: the model ---------------------------------------------------

    async def _ask_llm(self, state: AgentState) -> RoutingDecision:
        """Last resort. Constrained to registered roles."""
        routable = self._registry.routable()
        if not routable:
            msg = "No routable agents are registered."
            raise RoutingError(msg, agent=str(SUPERVISOR))

        catalogue = "\n".join(
            f"- {a.descriptor.role}: {a.descriptor.description}\n"
            f"    Use when: {a.descriptor.when_to_use}"
            for a in routable
        )

        choice = await self.llm.structured(
            [
                LLMMessage(
                    role="user",
                    content=_USER.format(
                        objective=state.objective,
                        plan_status=_render_plan(state),
                        recent=_render_recent(state),
                        errors=_render_errors(state),
                        steps_left=state.budget.steps_remaining,
                        tokens_left=state.budget.tokens_remaining,
                    ),
                )
            ],
            schema=RoutingChoice,
            system=_SYSTEM.format(catalogue=catalogue),
            temperature=0.0,  # routing is not a creative task
            max_tokens=512,
        )

        target_raw = choice.value.next_agent.strip().lower()

        if target_raw == str(FINISH):
            return RoutingDecision(FINISH, choice.value.reason, deterministic=False)

        try:
            role = AgentRole(target_raw)
        except ValueError:
            role = None  # type: ignore[assignment]

        if role is None or not self._registry.has(role):
            # The model picked something that does not exist. Do NOT crash the
            # run over it -- fall back to the planner, which can re-derive a
            # path. A supervisor that dies on a bad completion is a supervisor
            # that takes the whole run down with it.
            logger.warning("supervisor.invalid_llm_route", returned=target_raw)
            planner = AgentRole("planner")
            if self._registry.has(planner):
                return RoutingDecision(
                    planner,
                    f"Router returned an unknown agent ({target_raw!r}); replanning.",
                    deterministic=False,
                )
            msg = f"Router returned unknown agent {target_raw!r} and no planner is registered."
            raise RoutingError(msg, agent=str(SUPERVISOR))

        return RoutingDecision(role, choice.value.reason, deterministic=False)


# --- Prompt rendering --------------------------------------------------------


def _render_plan(state: AgentState) -> str:
    if not state.plan:
        return "No plan yet."
    lines = [f"Plan v{state.plan.version}:"]
    for step in state.plan.steps:
        marker = {
            "done": "[x]",
            "failed": "[!]",
            "in_progress": "[~]",
            "skipped": "[-]",
            "pending": "[ ]",
        }[step.status]
        agent = f" -> {step.assigned_agent}" if step.assigned_agent else ""
        lines.append(f"  {marker} {step.id}: {step.title}{agent}")
    return "\n".join(lines)


def _render_recent(state: AgentState, n: int = 6) -> str:
    msgs = state.recent_messages(n)
    if not msgs:
        return "Nothing yet."
    return "\n".join(f"  [{m.agent or m.role}] {m.content[:200]}" for m in msgs)


def _render_errors(state: AgentState, n: int = 3) -> str:
    if not state.errors:
        return "None."
    return "\n".join(
        f"  {e.agent}: [{e.code}] {e.message[:160]}" for e in state.errors[-n:]
    )


__all__ = ["RoutingChoice", "SupervisorAgent"]
