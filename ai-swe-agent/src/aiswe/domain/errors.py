"""Domain error taxonomy.

These errors are pure: they know nothing about HTTP, FastAPI, or SQLAlchemy.
The API layer (`api/middleware/error_handler.py`) is the ONLY place that
translates them into transport-level responses. This keeps the dependency rule
intact -- the domain never imports a web framework.
"""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """Base class for every expected failure in the system.

    Attributes:
        code: Stable, machine-readable identifier (e.g. ``run_not_found``).
              Clients switch on this, never on the message.
        message: Human-readable description. Safe to show to the caller.
        details: Optional structured context (field errors, ids, limits).
    """

    code: str = "domain_error"
    message: str = "An unexpected domain error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.code = code or self.__class__.code
        self.details: dict[str, Any] = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --- Client-side faults ------------------------------------------------------


class ValidationError(DomainError):
    """Input violated a domain invariant (not merely a schema mismatch)."""

    code = "validation_error"
    message = "The request payload is invalid."


class NotFoundError(DomainError):
    """A referenced aggregate does not exist within the caller's scope."""

    code = "not_found"
    message = "The requested resource was not found."


class ConflictError(DomainError):
    """The operation conflicts with current state (duplicate, version clash)."""

    code = "conflict"
    message = "The resource conflicts with existing state."


class UnauthorizedError(DomainError):
    """Identity could not be established."""

    code = "unauthorized"
    message = "Authentication is required."


class ForbiddenError(DomainError):
    """Identity is known but lacks the required scope."""

    code = "forbidden"
    message = "You do not have permission to perform this action."


class RateLimitError(DomainError):
    """Caller exceeded an allowance."""

    code = "rate_limited"
    message = "Too many requests."


class BudgetExceededError(DomainError):
    """A run exhausted its token / cost / step allowance.

    Distinct from RateLimitError: this is a *business* budget, not a traffic
    control. Enforced by ``domain/policies/budget_policy.py`` from M5.
    """

    code = "budget_exceeded"
    message = "The run exceeded its allotted budget."


# --- Server-side faults ------------------------------------------------------


class InfrastructureError(DomainError):
    """A downstream dependency (db, cache, provider) failed.

    Raised by adapters so that use cases never see driver-specific exceptions.
    """

    code = "infrastructure_error"
    message = "A downstream dependency failed."


class ConfigurationError(DomainError):
    """The system is misconfigured. Almost always fatal at startup."""

    code = "configuration_error"
    message = "The application is misconfigured."


# --- Agent orchestration faults ----------------------------------------------
#
# `retryable` is declared ON the error, not decided at the call site. That is
# what lets RetryPolicy stay declarative (see domain/policies/retry_policy.py)
# instead of degenerating into a pile of bespoke `except` clauses scattered
# across every agent.


class AgentError(DomainError):
    """Base class for failures inside an agent node."""

    code = "agent_error"
    message = "An agent failed to execute."
    retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        agent: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.agent = agent
        if retryable is not None:
            self.retryable = retryable
        if agent:
            self.details.setdefault("agent", agent)


class AgentExecutionError(AgentError):
    """The agent ran but could not produce a valid result."""

    code = "agent_execution_failed"
    message = "The agent failed to produce a valid result."
    retryable = True


class AgentOutputError(AgentError):
    """The agent's output did not satisfy its declared schema.

    Retryable: the usual cause is a malformed LLM response, and re-prompting
    with the validation error appended very often succeeds.
    """

    code = "agent_output_invalid"
    message = "The agent returned output that failed schema validation."
    retryable = True


class AgentNotFoundError(AgentError):
    """A route targeted an agent that is not registered."""

    code = "agent_not_found"
    message = "No agent is registered under that role."
    retryable = False


class RoutingError(AgentError):
    """The supervisor could not determine a next step."""

    code = "routing_failed"
    message = "The supervisor could not route the run."
    retryable = False


class MaxIterationsError(AgentError):
    """The graph exceeded its step ceiling.

    The single most important guardrail in an agent system: without it, a
    two-agent disagreement becomes an unbounded, billable loop.
    """

    code = "max_iterations_exceeded"
    message = "The run exceeded its maximum number of steps."
    retryable = False


class ToolExecutionError(DomainError):
    """A tool failed.

    NOTE: tool failures are normally returned to the agent as an *observation*
    (a ToolResult with ok=False), not raised -- an agent that sees "file not
    found" can adapt, whereas an exception just kills the node. This exception
    is reserved for failures the agent cannot possibly recover from.
    """

    code = "tool_execution_failed"
    message = "The tool failed to execute."
    retryable = True


class ToolNotFoundError(DomainError):
    code = "tool_not_found"
    message = "No tool is registered under that name."


class ToolPermissionError(DomainError):
    """The agent requested a tool outside its declared capability set.

    Enforced by the registry at BIND time, not by prompt instruction. A prompt
    is a suggestion; a registry is a wall.
    """

    code = "tool_permission_denied"
    message = "This agent is not permitted to use that tool."


class ApprovalRequiredError(DomainError):
    """A human approval gate blocked the operation."""

    code = "approval_required"
    message = "This operation requires human approval."


class LLMProviderError(DomainError):
    """An upstream model provider failed."""

    code = "llm_provider_error"
    message = "The model provider failed."
    retryable: bool = True

    def __init__(
        self,
        message: str | None = None,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message, details={"provider": provider, "status_code": status_code})
        self.provider = provider
        self.status_code = status_code
        if retryable is not None:
            self.retryable = retryable
