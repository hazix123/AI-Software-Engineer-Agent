"""Agent runtime port.

Owned by the APPLICATION layer, not the domain: orchestration is a use-case
concern, not a business rule. `agents/runtime.py` implements it with LangGraph.

Because use cases and workers depend on this Protocol, "which orchestration
library are we on" is a decision confined to three files.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiswe.agents.graph.state import AgentState
    from aiswe.agents.runtime import RunResult
    from aiswe.domain.value_objects.token_budget import TokenBudget


@runtime_checkable
class AgentRuntime(Protocol):
    """Executes agent runs. The application layer's only view of the graph."""

    async def start(
        self,
        *,
        run_id: uuid.UUID,
        objective: str,
        budget: TokenBudget,
        organization_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        repository_id: uuid.UUID | None = None,
    ) -> RunResult:
        """Execute a run from the beginning."""
        ...

    async def resume(self, run_id: uuid.UUID) -> RunResult:
        """Continue from the last checkpoint.

        Serves two paths that turn out to be the same mechanism: crash recovery,
        and resumption after a human resolves an approval gate.
        """
        ...

    async def approve(self, run_id: uuid.UUID, *, approved: bool, reason: str = "") -> RunResult:
        """Resolve an approval gate, then continue."""
        ...

    async def get_state(self, run_id: uuid.UUID) -> AgentState | None:
        """Read checkpointed state without advancing the graph."""
        ...
