"""Tool registry.

The enforcement point.

`tools_for(capabilities)` returns only the tools an agent's declared capability
set permits. The agent is HANDED its tools; it does not fetch them. It cannot
call `git.push` because it was never given `git.push` -- not because a prompt
asked it nicely not to.

`invoke()` composes the pipeline for every call:

    lookup -> validate args -> policy check -> middleware chain -> execute

Note that a policy denial returns a `ToolResult(ok=False)` rather than raising.
The agent is told "you are not permitted to use that tool", which is information
it can act on: it will pick a different approach, or hand back to the supervisor.
An exception would just kill the node and teach it nothing.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, TypeVar

from aiswe.domain.errors import ConflictError, ToolNotFoundError
from aiswe.domain.ports.tool import Tool, ToolContext, ToolResult
from aiswe.infrastructure.observability.logging import get_logger
from aiswe.tools.middleware import DEFAULT_MIDDLEWARE, ToolHandler, ToolMiddleware
from aiswe.tools.policy import DEFAULT_PREDICATES, ToolPolicy

if TYPE_CHECKING:
    from aiswe.tools.base import BaseTool

logger = get_logger(__name__)

ToolT = TypeVar("ToolT", bound="type[BaseTool]")


# --- Class registry (import-time) --------------------------------------------

_TOOL_CLASSES: dict[str, type[BaseTool]] = {}


def register_tool(cls: ToolT) -> ToolT:
    """Class decorator. Declares a tool to the framework.

    Same OCP mechanism as `@register_agent`: adding `git.commit` in M6 is a new
    class plus this decorator. The registry, the policy engine, and every agent's
    capability scoping pick it up automatically.
    """
    name = cls.spec.name  # type: ignore[attr-defined]
    if name in _TOOL_CLASSES and _TOOL_CLASSES[name] is not cls:
        existing = _TOOL_CLASSES[name].__name__
        msg = f"Tool {name!r} is already registered by {existing}."
        raise ConflictError(msg, code="duplicate_tool_name")
    _TOOL_CLASSES[name] = cls
    logger.debug("tool.class_registered", tool=name)
    return cls


def registered_tool_classes() -> dict[str, type[BaseTool]]:
    return dict(_TOOL_CLASSES)


def clear_tool_classes() -> None:
    """Test-only."""
    _TOOL_CLASSES.clear()


# --- Instance registry (runtime) ---------------------------------------------


class InMemoryToolRegistry:
    """Implements `domain.ports.tool.ToolRegistry`."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        policy: ToolPolicy | None = None,
        middleware: list[ToolMiddleware] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._policy = policy or ToolPolicy(extra_predicates=list(DEFAULT_PREDICATES))
        self._middleware = list(middleware if middleware is not None else DEFAULT_MIDDLEWARE)

        for tool in tools or []:
            self.register(tool)

    # --- Registration --------------------------------------------------------

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            msg = f"Tool {name!r} is already registered."
            raise ConflictError(msg, code="duplicate_tool_name")
        self._tools[name] = tool
        logger.debug("tool.registered", tool=name, tags=sorted(tool.spec.tags))

    # --- Discovery -----------------------------------------------------------

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            msg = f"No tool named {name!r}. Known tools: {sorted(self._tools)}"
            raise ToolNotFoundError(msg) from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def tools_for(self, capabilities: frozenset[str]) -> list[Tool]:
        """THE SCOPING BOUNDARY.

        A tool with no tags is universal. Otherwise the agent must hold at least
        one of its tags. This is what an agent's tool list is built from -- and
        what is NOT in that list, the agent cannot call.
        """
        return [
            tool
            for tool in self._tools.values()
            if not tool.spec.tags or (tool.spec.tags & capabilities)
        ]

    def llm_schemas(self, capabilities: frozenset[str]) -> list[dict[str, Any]]:
        """Tool definitions for the model, already scoped. (Consumed from M3.)"""
        return [t.spec.to_llm_schema() for t in self.tools_for(capabilities)]

    # --- Invocation ----------------------------------------------------------

    async def invoke(
        self,
        name: str,
        ctx: ToolContext,
        args: dict[str, Any],
        *,
        capabilities: frozenset[str],
    ) -> ToolResult:
        """Validate, authorise, and execute through the middleware chain."""
        # 1. Lookup. A missing tool IS returned as a failure, not raised: the
        #    model hallucinated a tool name, and telling it so is more useful
        #    than a stack trace.
        try:
            tool = self.get(name)
        except ToolNotFoundError as exc:
            return ToolResult.failure(name, code="tool_not_found", message=exc.message)

        # 2. Policy. Capability gate, then approval gate.
        decision = self._policy.check(
            tool.spec,
            capabilities=capabilities,
            args=args,
            ctx=ctx,
        )
        if not decision.allowed:
            return ToolResult.failure(
                name,
                code="tool_permission_denied",
                message=decision.reason,
            )
        if decision.requires_approval:
            # Surfaced to the graph, which raises an interrupt. The agent is told
            # to stop and wait rather than being silently blocked.
            return ToolResult.failure(
                name,
                code="approval_required",
                message=decision.reason,
            )

        # 3. Validate arguments against the tool's declared schema.
        try:
            parsed = tool.parse_args(args)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", "tool_args_invalid")
            return ToolResult.failure(name, code=code, message=str(exc))

        # 4. Middleware chain, then execute.
        handler = self._build_chain()
        return await handler(tool, ctx, parsed)

    def _build_chain(self) -> ToolHandler:
        """Fold the middleware list into a single callable.

        Built right-to-left so the FIRST middleware in the list is the OUTERMOST
        wrapper -- i.e. Audit sees the final, post-truncation result, which is
        what actually reached the agent.
        """

        async def terminal(tool: Tool, ctx: ToolContext, args: Any) -> ToolResult:
            return await tool.execute(ctx, args)

        def wrap(next_: ToolHandler, mw: ToolMiddleware) -> ToolHandler:
            async def handler(tool: Tool, ctx: ToolContext, args: Any) -> ToolResult:
                return await mw(tool, ctx, args, next_)

            return handler

        return reduce(wrap, reversed(self._middleware), terminal)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
