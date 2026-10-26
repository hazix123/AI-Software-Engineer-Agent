"""Stub LLM provider.

Real adapters (Anthropic / OpenAI / Gemini) and the model router are M2. This
exists so the agent framework can be RUN and TESTED now -- an orchestration layer
that has never executed is not a delivered orchestration layer, it is a
hypothesis.

DETERMINISTIC, NOT RANDOM
-------------------------
Given the same schema and the same state, it returns the same answer. That makes
graph tests reproducible: routing, replanning, cycle rejection, budget
exhaustion and livelock detection can all be asserted without an API key, a
network call, or a cent of spend.

`ScriptedLLM` goes further: you hand it the exact responses to return, in order,
so a test can drive the graph down a precise path -- including the failure paths
that are hard to provoke against a real model (a plan with a dependency cycle, a
router that names a nonexistent agent).

NEVER wire this into a deployed container. The container refuses to, in
production.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from aiswe.domain.errors import AgentOutputError
from aiswe.domain.ports.llm import (
    Completion,
    Message,
    StructuredCompletion,
    Usage,
)
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_USAGE = Usage(input_tokens=100, output_tokens=50, cost_usd=Decimal("0.001"))


class StubLLMProvider:
    """Implements `domain.ports.llm.LLMProvider`. Deterministic. No network."""

    def __init__(self, *, model: str = "stub-1") -> None:
        self._model = model

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,  # noqa: ARG002
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        model: str | None = None,  # noqa: ARG002
    ) -> Completion:
        last = messages[-1].content if messages else ""
        return Completion(
            text=f"[stub] acknowledged: {last[:80]}",
            usage=_USAGE,
            model=self._model,
            provider="stub",
        )

    async def structured(
        self,
        messages: list[Message],
        *,
        schema: type[SchemaT],
        system: str | None = None,  # noqa: ARG002
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        model: str | None = None,  # noqa: ARG002
    ) -> StructuredCompletion[SchemaT]:
        """Synthesise a valid instance of `schema`.

        Dispatch is by schema NAME rather than by import, so this module does not
        import the agents -- which would create a cycle (agents -> llm -> agents)
        and, worse, make an infrastructure adapter depend on the orchestration
        layer.
        """
        name = schema.__name__
        payload = _SYNTHESISERS.get(name, _generic)(schema, messages)

        try:
            value = schema.model_validate(payload)
        except ValidationError as exc:  # pragma: no cover - a stub bug, not a model bug
            msg = f"StubLLMProvider produced output invalid for {name}: {exc}"
            raise AgentOutputError(msg) from exc

        return StructuredCompletion(
            value=value,
            usage=_USAGE,
            model=self._model,
            provider="stub",
        )


class ScriptedLLM(StubLLMProvider):
    """Returns pre-scripted responses, in order.

    The test author's tool for driving the graph down an exact path -- including
    the paths a real model almost never takes, which are precisely the ones that
    break in production:

        llm = ScriptedLLM([
            DraftPlan(rationale="...", steps=[...cyclic...]),   # rejected
            DraftPlan(rationale="...", steps=[...valid...]),    # accepted
        ])

    Falls back to the deterministic stub once the script runs dry, so a test does
    not have to script every incidental call.
    """

    def __init__(
        self,
        responses: list[BaseModel] | None = None,
        *,
        model: str = "scripted-1",
    ) -> None:
        super().__init__(model=model)
        self._queue: deque[BaseModel] = deque(responses or [])

    async def structured(
        self,
        messages: list[Message],
        *,
        schema: type[SchemaT],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> StructuredCompletion[SchemaT]:
        if self._queue and isinstance(self._queue[0], schema):
            value = self._queue.popleft()
            return StructuredCompletion(
                value=value,  # type: ignore[arg-type]
                usage=_USAGE,
                model=self._model,
                provider="scripted",
            )
        return await super().structured(
            messages,
            schema=schema,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )

    @property
    def remaining(self) -> int:
        return len(self._queue)


# --- Synthesisers ------------------------------------------------------------
# Keyed by schema name to avoid an agents <- infrastructure import cycle.


def _routing_choice(  # noqa: ARG001
    schema: type[BaseModel], messages: list[Message]
) -> dict[str, Any]:
    """Route deterministically: plan first, then finish.

    Deliberately conservative. The stub must not pretend to do useful routing --
    a test that depends on the stub being clever is a test that will pass for the
    wrong reason.
    """
    text = messages[-1].content if messages else ""
    if "No plan yet" in text:
        return {"next_agent": "planner", "reason": "[stub] no plan exists", "confidence": 1.0}
    return {"next_agent": "finish", "reason": "[stub] nothing further to do", "confidence": 0.5}


def _draft_plan(  # noqa: ARG001
    schema: type[BaseModel], messages: list[Message]
) -> dict[str, Any]:
    """A minimal, always-valid single-step plan assigned to the memory agent.

    The memory agent is chosen because it is guaranteed to be registered and has
    no side effects -- so the stub plan is executable in any configuration.
    """
    return {
        "rationale": "[stub] single-step plan for framework testing.",
        "steps": [
            {
                "id": "s1",
                "title": "Record what was learned",
                "description": "Consolidate the run into long-term memory.",
                "assigned_agent": "memory",
                "depends_on": [],
                "acceptance_criteria": ["memories are persisted"],
            }
        ],
    }


def _consolidation(  # noqa: ARG001
    schema: type[BaseModel], messages: list[Message]
) -> dict[str, Any]:
    """Extract nothing.

    The correct default: a speculative memory is worse than no memory, and a
    stub has nothing real to say.
    """
    return {"memories": []}


def _generic(  # noqa: ARG001
    schema: type[BaseModel], messages: list[Message]
) -> dict[str, Any]:
    """Best-effort: fill required fields from their annotated types."""
    out: dict[str, Any] = {}
    for field_name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        annotation = field.annotation
        if annotation is str:
            out[field_name] = f"[stub] {field_name}"
        elif annotation is int:
            out[field_name] = 0
        elif annotation is float:
            out[field_name] = 0.0
        elif annotation is bool:
            out[field_name] = False
        else:
            out[field_name] = []
    return out


_SYNTHESISERS: dict[str, Any] = {
    "RoutingChoice": _routing_choice,
    "DraftPlan": _draft_plan,
    "Consolidation": _consolidation,
}
