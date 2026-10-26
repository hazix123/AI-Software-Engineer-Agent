"""Run budget.

An agent loop without a budget is an unbounded bill. Two agents that disagree
will happily ping-pong until someone notices the invoice.

Immutable by design: `spend()` returns a NEW budget rather than mutating. That
means budget state merges cleanly through the LangGraph reducer and cannot be
silently corrupted by a node that forgets to write it back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Self


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Tracks tokens, money, and steps against hard ceilings.

    All three limits matter independently:
      * tokens  -- guards context blow-up
      * cost    -- guards the invoice (a cheap model can still loop forever)
      * steps   -- guards wall-clock time and catches livelock even when a
                   node is cheap enough not to trip the other two
    """

    max_tokens: int
    max_cost_usd: Decimal
    max_steps: int

    tokens_used: int = 0
    cost_usd: Decimal = Decimal("0")
    steps_used: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.max_steps <= 0 or self.max_cost_usd <= 0:
            msg = "Budget ceilings must be positive."
            raise ValueError(msg)

    # --- queries -------------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        """True if ANY ceiling is hit. Checked before every LLM call."""
        return (
            self.tokens_used >= self.max_tokens
            or self.cost_usd >= self.max_cost_usd
            or self.steps_used >= self.max_steps
        )

    @property
    def exhausted_reason(self) -> str | None:
        if self.tokens_used >= self.max_tokens:
            return f"token limit reached ({self.tokens_used}/{self.max_tokens})"
        if self.cost_usd >= self.max_cost_usd:
            return f"cost limit reached (${self.cost_usd}/${self.max_cost_usd})"
        if self.steps_used >= self.max_steps:
            return f"step limit reached ({self.steps_used}/{self.max_steps})"
        return None

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    # --- transitions ---------------------------------------------------------

    def spend(
        self,
        *,
        tokens: int = 0,
        cost_usd: Decimal | None = None,
        steps: int = 0,
    ) -> Self:
        """Return a new budget with the spend applied. Never mutates."""
        if tokens < 0 or steps < 0:
            msg = "Cannot spend a negative amount."
            raise ValueError(msg)
        return replace(
            self,
            tokens_used=self.tokens_used + tokens,
            cost_usd=self.cost_usd + (cost_usd or Decimal("0")),
            steps_used=self.steps_used + steps,
        )

    def step(self) -> Self:
        """Consume exactly one graph step. Called once per node entry."""
        return self.spend(steps=1)
