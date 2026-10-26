"""Run lifecycle status.

This one IS a closed enum -- and the asymmetry with `AgentRole` is deliberate.

Adding an agent must be free (open/closed). Adding a *run status* is a change to
the core state machine: it affects the API contract, the UI, the worker's lease
logic, and every transition guard. It SHOULD force a review. Making it an enum
is what forces that review.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Terminal runs are never resumed and never re-queued."""
        return self in _TERMINAL

    @property
    def is_active(self) -> bool:
        """Active runs hold a worker lease and must emit heartbeats."""
        return self in {RunStatus.RUNNING, RunStatus.AWAITING_APPROVAL}

    def can_transition_to(self, target: RunStatus) -> bool:
        """Guard every write. Illegal transitions are bugs, not edge cases."""
        return target in _TRANSITIONS[self]


_TERMINAL: Final[frozenset[RunStatus]] = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)

# The state machine, stated once, in one place.
#
# Note AWAITING_APPROVAL -> RUNNING: this is the human-in-the-loop resume path.
# Note PAUSED -> RUNNING: this is the crash-recovery path, where another worker
# picks up an expired lease and resumes from the last checkpoint.
_TRANSITIONS: Final[dict[RunStatus, frozenset[RunStatus]]] = {
    RunStatus.PENDING: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.AWAITING_APPROVAL,
            RunStatus.PAUSED,
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.AWAITING_APPROVAL: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}
    ),
    RunStatus.PAUSED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}
