"""Planner agent.

Decomposes the objective into a DAG of steps, each assigned to an agent.

VALIDATION IS THE POINT
-----------------------
Generating a plan is easy. Generating a plan that cannot deadlock the graph is
the hard part, and it is where this file spends its effort.

A plan is rejected -- and re-prompted with the specific error -- if it:

  * references an agent that is not registered  (routes into a void)
  * contains a dependency cycle                 (nothing is ever ready; the run
                                                 spins until the budget dies)
  * depends on a step that does not exist       (permanently unsatisfiable)
  * is empty                                    (nothing to do)

None of these fail loudly at runtime. They all present as "the agent got stuck
and burned $12", which is far harder to diagnose than a rejected plan. Catching
them here costs one re-prompt.

OPEN/CLOSED: the planner learns the available agents from the REGISTRY. When the
Security Review Agent registers in M7, the planner can immediately assign work to
it. Not one line of this file changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from aiswe.agents.base.base_agent import BaseAgent
from aiswe.agents.graph.state import Plan, PlanStep
from aiswe.agents.registry import register_agent
from aiswe.domain.errors import AgentOutputError
from aiswe.domain.ports.agent import AgentDescriptor, AgentOutput
from aiswe.domain.ports.llm import Message as LLMMessage
from aiswe.domain.value_objects.agent_role import PLANNER, AgentRole
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.agents.graph.state import AgentState
    from aiswe.domain.ports.agent import AgentRegistry
    from aiswe.domain.ports.event_bus import EventBus
    from aiswe.domain.ports.llm import LLMProvider

logger = get_logger(__name__)


class DraftStep(BaseModel):
    """One proposed step. Deliberately close to PlanStep, but untrusted."""

    id: str = Field(description="Short, stable id: 's1', 's2'.")
    title: str = Field(description="Imperative, one line.")
    description: str = Field(description="What must be done, concretely.")
    assigned_agent: str = Field(description="Exactly one role from the agent list.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step ids that must complete first. Empty if independent.",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="Objectively checkable conditions for 'done'.",
    )


class DraftPlan(BaseModel):
    """The planner's raw output. Validated before it becomes a Plan."""

    rationale: str = Field(description="Two sentences: the strategy and why.")
    steps: list[DraftStep] = Field(description="Ordered, minimal, non-overlapping.")


_SYSTEM = """\
You are the Planner of a multi-agent software engineering system.

Decompose the objective into the SMALLEST set of steps that will actually \
achieve it. Fewer, larger, well-scoped steps beat many trivial ones: every step \
costs a model call.

Available agents:
{catalogue}

Rules:
- `assigned_agent` MUST be exactly one role from the list above. Never invent one.
- Use `depends_on` only for real ordering constraints. Steps with no dependency \
between them may run in parallel, which is cheaper and faster.
- Dependencies must form a DAG. A cycle will be rejected.
- Every `depends_on` entry must reference a step id that exists in this plan.
- `acceptance_criteria` must be objectively checkable ("all tests pass", \
"no HIGH severity findings"), never subjective ("code is clean").
- Do not plan work that no available agent can perform.

{replan_context}
"""

_USER = """\
Objective:
{objective}

{memory_block}

{failure_context}

Produce the plan.
"""


@register_agent
class PlannerAgent(BaseAgent):
    """Turns an objective into an executable, validated DAG."""

    descriptor = AgentDescriptor(
        role=PLANNER,
        description="Decomposes an objective into an ordered plan of agent-assigned steps.",
        when_to_use=(
            "At the start of a run, or whenever a step fails and the approach "
            "must be reconsidered."
        ),
        capabilities=frozenset({"memory"}),  # may recall; may not touch code
        requires_llm=True,
        max_consecutive_visits=2,  # replan twice, then stop. Three is thrashing.
    )

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        llm: LLMProvider | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        super().__init__(llm=llm, event_bus=event_bus)
        self._registry = registry

    async def execute(self, state: AgentState) -> AgentOutput:
        is_replan = state.plan is not None
        version = (state.plan.version + 1) if state.plan else 1

        draft = await self._draft(state, is_replan=is_replan)
        plan = self._validate(draft, version=version)

        logger.info("planner.produced", version=version, steps=len(plan.steps), replan=is_replan)

        delta: dict[str, Any] = {
            "plan": plan,
            "current_step_id": None,
            # Hand control straight back to the supervisor, which will apply the
            # deterministic rules and dispatch the first ready step.
            "next_agent": None,
        }

        verb = "Replanned" if is_replan else "Planned"
        return AgentOutput(
            status="ok",
            summary=f"{verb} in {len(plan.steps)} steps: {plan.rationale}",
            state_delta=delta,
        )

    # --- Drafting ------------------------------------------------------------

    async def _draft(self, state: AgentState, *, is_replan: bool) -> DraftPlan:
        catalogue = "\n".join(
            f"- {a.descriptor.role}: {a.descriptor.description}\n"
            f"    Use when: {a.descriptor.when_to_use}"
            for a in self._registry.routable()
        )

        completion = await self.llm.structured(
            [
                LLMMessage(
                    role="user",
                    content=_USER.format(
                        objective=state.objective,
                        memory_block=_render_memories(state),
                        failure_context=_render_failures(state),
                    ),
                )
            ],
            schema=DraftPlan,
            system=_SYSTEM.format(
                catalogue=catalogue,
                replan_context=(
                    "This is a REPLAN. A previous attempt failed. Do not repeat the "
                    "approach that failed -- change strategy."
                    if is_replan
                    else ""
                ),
            ),
            temperature=0.1,
            max_tokens=2_048,
        )
        return completion.value

    # --- Validation ----------------------------------------------------------

    def _validate(self, draft: DraftPlan, *, version: int) -> Plan:
        """Reject any plan that would deadlock or misroute the graph.

        Raises AgentOutputError (retryable), so BaseAgent re-prompts with the
        specific problem rather than accepting a plan that cannot execute.
        """
        if not draft.steps:
            msg = "The plan contains no steps."
            raise AgentOutputError(msg, agent=str(PLANNER))

        ids = [s.id for s in draft.steps]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"Duplicate step ids: {dupes}."
            raise AgentOutputError(msg, agent=str(PLANNER))

        known = set(ids)
        for step in draft.steps:
            # 1. The agent must exist. Otherwise the run routes into a void.
            try:
                role = AgentRole(step.assigned_agent)
            except ValueError as exc:
                msg = f"Step {step.id!r}: {step.assigned_agent!r} is not a valid role name."
                raise AgentOutputError(msg, agent=str(PLANNER)) from exc

            if not self._registry.has(role):
                available = sorted(str(a.descriptor.role) for a in self._registry.routable())
                msg = (
                    f"Step {step.id!r} is assigned to unknown agent "
                    f"{step.assigned_agent!r}. Available: {available}."
                )
                raise AgentOutputError(msg, agent=str(PLANNER))

            # 2. Dependencies must exist. A dangling dep is never satisfiable,
            #    so the step never becomes ready and the run spins to death.
            for dep in step.depends_on:
                if dep not in known:
                    msg = f"Step {step.id!r} depends on {dep!r}, which does not exist."
                    raise AgentOutputError(msg, agent=str(PLANNER))
                if dep == step.id:
                    msg = f"Step {step.id!r} depends on itself."
                    raise AgentOutputError(msg, agent=str(PLANNER))

        # 3. No cycles. This is the failure that is hardest to spot by eye and
        #    most expensive at runtime: nothing is ever ready, and the run
        #    burns its entire budget making zero progress.
        cycle = _find_cycle({s.id: s.depends_on for s in draft.steps})
        if cycle:
            msg = f"The plan has a dependency cycle: {' -> '.join(cycle)}."
            raise AgentOutputError(msg, agent=str(PLANNER))

        return Plan(
            version=version,
            rationale=draft.rationale,
            steps=[
                PlanStep(
                    id=s.id,
                    title=s.title,
                    description=s.description,
                    assigned_agent=str(AgentRole(s.assigned_agent)),
                    depends_on=list(s.depends_on),
                    acceptance_criteria=list(s.acceptance_criteria),
                )
                for s in draft.steps
            ],
        )


# --- Helpers -----------------------------------------------------------------


def _find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    """Depth-first cycle detection. Returns the cycle path, or None.

    Returning the actual path (not just True) matters: it goes straight into the
    re-prompt, so the model is told exactly what to fix.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        colour[node] = GREY
        path.append(node)
        for dep in graph.get(node, []):
            if colour.get(dep) == GREY:
                # Found a back-edge: slice the cycle out of the current path.
                return [*path[path.index(dep) :], dep]
            if colour.get(dep) == WHITE:
                found = visit(dep)
                if found:
                    return found
        path.pop()
        colour[node] = BLACK
        return None

    for node in graph:
        if colour[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return None


def _render_memories(state: AgentState) -> str:
    block = state.memory_block()
    if not block:
        return ""
    return f"What we already know about this project:\n{block}\n"


def _render_failures(state: AgentState) -> str:
    if not state.errors:
        return ""
    recent = state.errors[-3:]
    lines = "\n".join(f"- {e.agent}: [{e.code}] {e.message}" for e in recent)
    return (
        "Previous attempts failed as follows. Plan a DIFFERENT approach; "
        f"do not repeat what failed:\n{lines}\n"
    )


__all__ = ["DraftPlan", "DraftStep", "PlannerAgent"]
