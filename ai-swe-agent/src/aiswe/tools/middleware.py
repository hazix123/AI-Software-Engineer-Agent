"""Tool middleware.

A composable chain wrapped around every tool call:

    invoke -> Audit -> Retry -> Truncate -> tool.execute()

Each middleware is a small object with one job (SRP), and the chain is a list --
so adding one is an append, not an edit (OCP).

TRUNCATION IS THE ONE THAT SAVES RUNS
-------------------------------------
A pytest run on a large repo emits 40,000 lines. Handed to the model raw, it:

  * blows the context window, ending the run outright; or
  * survives, but costs several dollars and buries the three lines that matter.

`TruncationMiddleware` replaces the payload with a head+tail excerpt, a summary,
and an artifact reference. The agent sees ~40 useful lines instead of 40,000
useless ones, and can fetch the full log deliberately if it needs to.

Unbounded tool output is the single most common way an agent run dies. This is
not an optimisation; it is load-bearing.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aiswe.domain.policies.retry_policy import DEFAULT_TOOL_RETRY, RetryPolicy
from aiswe.domain.ports.tool import ToolResult
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.domain.ports.tool import Tool, ToolContext

logger = get_logger(__name__)

#: The next link in the chain.
ToolHandler = Callable[["Tool", "ToolContext", Any], Awaitable[ToolResult]]


class ToolMiddleware(ABC):
    """One cross-cutting concern, applied to every tool call."""

    @abstractmethod
    async def __call__(
        self,
        tool: Tool,
        ctx: ToolContext,
        args: Any,
        next_: ToolHandler,
    ) -> ToolResult:
        """Do work, call `next_`, do more work."""


class AuditMiddleware(ToolMiddleware):
    """Log every call. In M3 this also writes a `tool_invocations` row.

    Arguments are previewed, not dumped: a tool argument can contain a file's
    contents or a credential, and a log is the last place either belongs.
    """

    async def __call__(
        self,
        tool: Tool,
        ctx: ToolContext,
        args: Any,
        next_: ToolHandler,
    ) -> ToolResult:
        name = tool.spec.name
        logger.info(
            "tool.called",
            tool=name,
            agent=ctx.agent,
            run_id=str(ctx.run_id),
            args=_preview_args(args),
        )

        result = await next_(tool, ctx, args)

        logger.info(
            "tool.result",
            tool=name,
            agent=ctx.agent,
            ok=result.ok,
            duration_ms=round(result.duration_ms, 2),
            error_code=result.error_code,
        )
        return result


class RetryMiddleware(ToolMiddleware):
    """Retry transient tool failures.

    Only retries when BOTH the result says retryable AND the tool declares
    itself idempotent. Retrying a non-idempotent tool that may have partially
    succeeded (a commit that landed before the connection dropped) produces
    duplicate side effects -- worse than the original failure.
    """

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self._policy = policy or DEFAULT_TOOL_RETRY

    async def __call__(
        self,
        tool: Tool,
        ctx: ToolContext,
        args: Any,
        next_: ToolHandler,
    ) -> ToolResult:
        if not tool.spec.idempotent:
            return await next_(tool, ctx, args)

        result = await next_(tool, ctx, args)

        for attempt in range(1, self._policy.max_attempts):
            if result.ok or not result.retryable:
                return result

            delay = self._policy.delay_for(attempt)
            logger.warning(
                "tool.retrying",
                tool=tool.spec.name,
                attempt=attempt,
                delay_s=round(delay, 2),
                error_code=result.error_code,
            )
            await asyncio.sleep(delay)
            result = await next_(tool, ctx, args)

        return result


class TruncationMiddleware(ToolMiddleware):
    """Cap tool output. The context-window guard.

    Head + tail, because both ends carry signal and the middle rarely does: the
    head has the command and the first failure; the tail has the summary line
    ("3 failed, 412 passed"). The middle is 39,000 lines of passing tests.
    """

    #: ~4 characters per token is a decent cross-tokeniser approximation.
    CHARS_PER_TOKEN = 4

    def __init__(self, head_lines: int = 30, tail_lines: int = 20) -> None:
        self._head = head_lines
        self._tail = tail_lines

    async def __call__(
        self,
        tool: Tool,
        ctx: ToolContext,
        args: Any,
        next_: ToolHandler,
    ) -> ToolResult:
        result = await next_(tool, ctx, args)

        if not result.ok or not result.preview:
            return result

        limit_chars = tool.spec.max_output_tokens * self.CHARS_PER_TOKEN
        if len(result.preview) <= limit_chars:
            return result

        lines = result.preview.splitlines()
        if len(lines) > self._head + self._tail:
            omitted = len(lines) - self._head - self._tail
            excerpt = "\n".join(
                [
                    *lines[: self._head],
                    f"\n... [{omitted} lines omitted; full output stored as an artifact] ...\n",
                    *lines[-self._tail :],
                ]
            )
        else:
            excerpt = result.preview[:limit_chars] + "\n... [truncated] ..."

        logger.info(
            "tool.truncated",
            tool=tool.spec.name,
            original_chars=len(result.preview),
            kept_chars=len(excerpt),
        )

        # M3 writes the full payload to object storage and sets artifact_ref.
        # Until then the agent still gets the useful excerpt rather than a wall.
        return ToolResult(
            ok=result.ok,
            tool=result.tool,
            data=result.data,
            preview=excerpt,
            artifact_ref=result.artifact_ref,
            truncated=True,
            duration_ms=result.duration_ms,
            invocation_id=result.invocation_id,
        )


def _preview_args(args: Any, limit: int = 300) -> str:
    """Render arguments for a log line -- short, and never a secret."""
    try:
        payload = args.model_dump() if hasattr(args, "model_dump") else args
        text = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        text = repr(args)
    return text[:limit] + ("…" if len(text) > limit else "")


DEFAULT_MIDDLEWARE: list[ToolMiddleware] = [
    AuditMiddleware(),
    RetryMiddleware(),
    TruncationMiddleware(),
]
