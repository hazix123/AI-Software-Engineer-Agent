"""Retry policy.

Pure and injectable, so retry behaviour is configured once and obeyed
everywhere. No agent writes its own `for attempt in range(3)` loop -- that is
how you end up with five different backoff strategies and none of them tested.

TWO NON-OBVIOUS DECISIONS
-------------------------
1. **Full jitter, not fixed backoff.** When a provider has an outage, every
   worker fails at once. With fixed backoff they all retry at the same instant
   and knock the provider straight back over the moment it recovers -- a
   thundering herd of your own making. Full jitter spreads the retries out.

2. **Retryability is read off the error**, never guessed here. `AgentOutputError`
   is retryable because re-prompting with the validation message usually works.
   `ToolPermissionError` is not: retrying a permission denial just wastes money
   and produces the identical denial.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from aiswe.domain.errors import DomainError

# Errors that are ALWAYS worth another attempt, regardless of what they declare.
# Transient by nature: a network blip, a 429, a pool timeout.
_ALWAYS_RETRYABLE_CODES: frozenset[str] = frozenset(
    {
        "infrastructure_error",
        "llm_provider_error",
        "db_commit_failed",
        "lock_timeout",
        "rate_limited",
    }
)

# Errors that are NEVER worth another attempt. Retrying these is pure waste:
# the outcome is deterministic and it will be identical next time.
_NEVER_RETRYABLE_CODES: frozenset[str] = frozenset(
    {
        "budget_exceeded",
        "max_iterations_exceeded",
        "tool_permission_denied",
        "agent_not_found",
        "tool_not_found",
        "forbidden",
        "unauthorized",
        "approval_required",
        "configuration_error",
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    delay = random.uniform(0, min(max_delay, base * multiplier ** attempt))
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = "max_attempts must be at least 1."
            raise ValueError(msg)
        if self.base_delay_s < 0 or self.max_delay_s < self.base_delay_s:
            msg = "Invalid delay bounds."
            raise ValueError(msg)

    def should_retry(self, error: Exception, *, attempt: int) -> bool:
        """`attempt` is 1-based: attempt=1 means the first try just failed."""
        if attempt >= self.max_attempts:
            return False
        return is_retryable(error)

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Backoff before attempt N+1. Full jitter over [0, capped_ceiling]."""
        ceiling = min(self.max_delay_s, self.base_delay_s * (self.multiplier ** (attempt - 1)))
        source = rng or random
        return source.uniform(0.0, ceiling)  # noqa: S311 - jitter, not cryptography


def is_retryable(error: Exception) -> bool:
    """Decide, once, whether an error is worth another attempt.

    Precedence: explicit deny-list, then explicit allow-list, then whatever the
    error itself declares. Anything unrecognised is NOT retried -- an unknown
    failure repeated three times is three times the damage, not three times the
    chance of success.
    """
    if isinstance(error, DomainError):
        if error.code in _NEVER_RETRYABLE_CODES:
            return False
        if error.code in _ALWAYS_RETRYABLE_CODES:
            return True
        return bool(getattr(error, "retryable", False))

    # Transport-level failures raised before an adapter could wrap them.
    return isinstance(error, TimeoutError | ConnectionError)


# Sensible defaults, tuned per workload.
DEFAULT_AGENT_RETRY = RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=20.0)

# Tools fail fast: a failing tool is usually returned to the agent as an
# observation so it can adapt, which is more productive than retrying blindly.
DEFAULT_TOOL_RETRY = RetryPolicy(max_attempts=2, base_delay_s=0.5, max_delay_s=5.0)

# Providers get more patience: 429s and 529s genuinely do clear.
DEFAULT_LLM_RETRY = RetryPolicy(max_attempts=4, base_delay_s=2.0, max_delay_s=60.0)
