"""Base tool.

Mirrors BaseAgent: subclasses implement `_run()`; the base class owns argument
validation, timeout, and -- most importantly -- the conversion of exceptions
into `ToolResult.failure(...)`.

WHY EXCEPTIONS BECOME RESULTS
-----------------------------
An agent asks to read `src/mian.py`. The file does not exist.

    Exception  -> the node dies, a retry is burned re-running the whole agent,
                  and the model never learns what went wrong.
    ToolResult -> the agent's next turn contains
                  "fs.read FAILED -> [not_found] no such file: src/mian.py",
                  it spots the typo, and calls `fs.read src/main.py`.

The second is what makes an agent look competent. The first is what makes it
look broken. Only failures the agent genuinely cannot act on should raise.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ValidationError

from aiswe.domain.errors import DomainError
from aiswe.domain.ports.tool import ToolResult, ToolSpec
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.domain.ports.tool import ToolContext

logger = get_logger(__name__)


class BaseTool(ABC):
    """Every tool extends this. One tool, one verb."""

    #: Subclasses MUST override.
    spec: ClassVar[ToolSpec]

    def __init__(self) -> None:
        if not hasattr(self, "spec"):
            msg = f"{type(self).__name__} must declare a `spec` class attribute."
            raise TypeError(msg)

    @abstractmethod
    async def _run(self, ctx: ToolContext, args: Any) -> ToolResult:
        """The tool's actual work. `args` is already validated against args_schema."""

    async def execute(self, ctx: ToolContext, args: BaseModel) -> ToolResult:
        """Run with timeout and failure capture. Does not raise on user error."""
        started = time.perf_counter()
        name = self.spec.name

        try:
            async with asyncio.timeout(self.spec.timeout_s):
                result = await self._run(ctx, args)

        except TimeoutError:
            elapsed = (time.perf_counter() - started) * 1000
            logger.warning("tool.timeout", tool=name, timeout_s=self.spec.timeout_s)
            return ToolResult.failure(
                name,
                code="timeout",
                message=f"{name} exceeded its {self.spec.timeout_s}s limit.",
                # Retryable only if idempotent: re-running a non-idempotent tool
                # that may have half-completed (a commit, a push) is how you get
                # duplicate side effects.
                retryable=self.spec.idempotent,
                duration_ms=elapsed,
            )

        except DomainError as exc:
            elapsed = (time.perf_counter() - started) * 1000
            logger.warning("tool.failed", tool=name, code=exc.code, error=exc.message)
            return ToolResult.failure(
                name,
                code=exc.code,
                message=exc.message,
                retryable=bool(getattr(exc, "retryable", False)),
                duration_ms=elapsed,
            )

        except Exception as exc:  # noqa: BLE001 - deliberate: see module docstring
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception("tool.unexpected_error", tool=name)
            return ToolResult.failure(
                name,
                code="tool_error",
                message=f"{type(exc).__name__}: {exc}",
                retryable=False,
                duration_ms=elapsed,
            )

        # Stamp the duration if the tool did not.
        if result.duration_ms == 0.0:
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(
                ok=result.ok,
                tool=result.tool,
                data=result.data,
                preview=result.preview,
                artifact_ref=result.artifact_ref,
                truncated=result.truncated,
                error_code=result.error_code,
                error_message=result.error_message,
                retryable=result.retryable,
                duration_ms=elapsed,
                invocation_id=result.invocation_id,
            )
        return result

    def parse_args(self, raw: dict[str, Any]) -> BaseModel:
        """Validate raw LLM-supplied arguments against the declared schema.

        Raises:
            DomainError: with the field-level errors, so the agent's next turn
                explains exactly what was wrong rather than "invalid input".
        """
        try:
            return self.spec.args_schema.model_validate(raw)
        except ValidationError as exc:
            details = [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ]
            msg = f"Invalid arguments for {self.spec.name}: {details}"
            raise DomainError(msg, code="tool_args_invalid", details={"errors": details}) from exc

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.spec.name!r}>"
