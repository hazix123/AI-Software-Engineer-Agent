"""Clock port.

Time is an external dependency like any other. Injecting it means expiry,
TTL, decay and lease logic can be tested deterministically without patching
the stdlib.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Supplies the current instant. Always timezone-aware, always UTC."""

    def now(self) -> datetime:
        """Return the current UTC time."""
        ...


class SystemClock:
    """Production implementation. Lives in the domain because it has no deps."""

    def now(self) -> datetime:
        return datetime.now(UTC)
