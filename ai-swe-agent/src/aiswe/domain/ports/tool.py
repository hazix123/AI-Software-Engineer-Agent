"""Tool port.

TWO LOAD-BEARING DECISIONS
--------------------------
1. **Tool failures are OBSERVATIONS, not exceptions.** `execute()` returns a
   `ToolResult(ok=False, ...)` rather than raising. An agent that is told "file
   not found: src/mian.py" can notice the typo and retry with `src/main.py`. An
   exception just kills the node and burns a retry. Only failures the agent
   genuinely cannot act on are raised.

2. **Permissions are enforced by the REGISTRY at bind time, not by the prompt.**
   A prompt saying "you must not push to git" is a suggestion. A registry that
   never hands the Documentation Agent `git.push` is a wall. Prompts are not a
   security boundary.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class Permission(StrEnum):
    """What a tool is capable of doing. Declared by the tool, granted by policy."""

    READ = "read"  # inspect the workspace / index
    WRITE = "write"  # mutate files
    EXEC = "exec"  # run commands in the sandbox
    NETWORK = "network"  # reach outside the sandbox
    VCS_WRITE = "vcs_write"  # commit / branch
    VCS_PUSH = "vcs_push"  # push / open PR  -- always approval-gated
    MEMORY_WRITE = "memory_write"  # persist learned facts


class CostClass(StrEnum):
    """Rough expense of a call. The supervisor uses it for budget-aware routing."""

    CHEAP = "cheap"  # sub-second, no LLM  (grep, read)
    MODERATE = "moderate"  # seconds          (test run, index query)
    EXPENSIVE = "expensive"  # LLM call or minutes (threat review, e2e suite)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool's declaration. Everything the registry and the LLM need to know."""

    name: str  # stable, dotted: "code.find_references"
    description: str  # LLM-facing. Written as an instruction, not prose.
    args_schema: type[BaseModel]
    permissions: frozenset[Permission] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)

    requires_approval: bool = False
    idempotent: bool = True  # safe to retry?
    cost_class: CostClass = CostClass.CHEAP
    timeout_s: float = 30.0
    max_output_tokens: int = 4_000  # results above this are truncated + stored

    def to_llm_schema(self) -> dict[str, Any]:
        """Provider-neutral tool definition. Adapters translate per provider."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_schema.model_json_schema(),
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Uniform outcome envelope. Every tool returns this -- success or failure.

    `artifact_ref` is how context bloat is prevented: a 40k-line test log is
    written to object storage and the agent sees a summary plus a pointer, not
    the whole thing. Unbounded tool output is the single most common way an
    agent run dies.
    """

    ok: bool
    tool: str
    data: Any = None
    preview: str = ""  # short, LLM-facing rendering
    artifact_ref: str | None = None  # pointer to the full payload
    truncated: bool = False

    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False

    duration_ms: float = 0.0
    invocation_id: uuid.UUID = field(default_factory=uuid.uuid4)

    @classmethod
    def success(
        cls,
        tool: str,
        *,
        data: Any = None,
        preview: str = "",
        artifact_ref: str | None = None,
        truncated: bool = False,
        duration_ms: float = 0.0,
    ) -> ToolResult:
        return cls(
            ok=True,
            tool=tool,
            data=data,
            preview=preview,
            artifact_ref=artifact_ref,
            truncated=truncated,
            duration_ms=duration_ms,
        )

    @classmethod
    def failure(
        cls,
        tool: str,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        duration_ms: float = 0.0,
    ) -> ToolResult:
        return cls(
            ok=False,
            tool=tool,
            error_code=code,
            error_message=message,
            retryable=retryable,
            preview=f"ERROR [{code}]: {message}",
            duration_ms=duration_ms,
        )

    def as_observation(self) -> str:
        """What the agent actually sees on its next turn."""
        if self.ok:
            suffix = " (output truncated; full result stored)" if self.truncated else ""
            return f"{self.tool} -> {self.preview}{suffix}"
        return f"{self.tool} FAILED -> [{self.error_code}] {self.error_message}"


@dataclass(slots=True)
class ToolContext:
    """Ambient context handed to every tool call.

    Passed in rather than reached for: a tool that reads a global to find its
    run_id is a tool that cannot be tested and cannot run concurrently.
    """

    run_id: uuid.UUID
    agent: str
    organization_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    workspace_id: uuid.UUID | None = None  # M3
    approved: bool = False  # set by the approval gate
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """A single capability. One tool, one verb."""

    @property
    def spec(self) -> ToolSpec:
        ...

    async def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        """Run the tool. Returns a ToolResult -- it does not raise on user error."""
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Discovery + capability scoping.

    `tools_for()` is the enforcement point: an agent is handed only the tools
    its declared capabilities permit. It cannot call what it was never given.
    """

    def register(self, tool: Tool) -> None:
        ...

    def get(self, name: str) -> Tool:
        """Raises ToolNotFoundError if absent."""
        ...

    def tools_for(self, capabilities: frozenset[str]) -> list[Tool]:
        """Every tool whose tags intersect the granted capability set."""
        ...

    async def invoke(
        self,
        name: str,
        ctx: ToolContext,
        args: dict[str, Any],
        *,
        capabilities: frozenset[str],
    ) -> ToolResult:
        """Validate, authorise, and execute -- through the middleware chain."""
        ...
