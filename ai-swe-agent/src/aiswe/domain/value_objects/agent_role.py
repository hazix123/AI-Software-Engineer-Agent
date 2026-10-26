"""Agent role identifiers.

WHY THIS IS NOT AN ENUM
-----------------------
An Enum is a *closed* set. If `AgentRole` were an Enum, registering the
Security Review Agent in M7 would require editing this file -- and then the
router, and then the graph builder. That is precisely the "modify existing
code to add an agent" failure the framework is required to avoid.

So a role is a validated string, and the AgentRegistry is the single source of
truth about which roles exist. The constants below are conveniences for the
agents that ship today, not an exhaustive list.

The value object still earns its keep: it normalises casing, rejects garbage at
the boundary, and gives us a distinct type so a role can never be confused with
an arbitrary string in a signature.
"""

from __future__ import annotations

import re
from typing import Any, Final, Self

_ROLE_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class AgentRole(str):
    """A registered agent's identifier. Immutable, normalised, validated.

    Subclasses `str` so it can be used directly as a LangGraph node name and as
    a dict key without unwrapping, while still being a nominal type for mypy.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        normalised = value.strip().lower()
        if not _ROLE_PATTERN.match(normalised):
            msg = (
                f"Invalid agent role {value!r}: must be snake_case, "
                "start with a letter, and be 2-64 characters."
            )
            raise ValueError(msg)
        return super().__new__(cls, normalised)

    def __repr__(self) -> str:
        return f"AgentRole({str.__str__(self)!r})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        """Let Pydantic validate and coerce plain strings into AgentRole."""
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


# --- Roles shipping today ----------------------------------------------------
# Conveniences, NOT a closed set. A new agent adds its own constant in its own
# package; nothing here needs to change.

SUPERVISOR: Final = AgentRole("supervisor")
PLANNER: Final = AgentRole("planner")
MEMORY: Final = AgentRole("memory")

# --- Reserved graph terminals ------------------------------------------------
# Not agents: sentinel destinations the router may return.

FINISH: Final = AgentRole("finish")
