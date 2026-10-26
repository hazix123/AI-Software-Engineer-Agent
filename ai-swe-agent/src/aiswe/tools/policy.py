"""Tool policy engine.

Two independent gates, and conflating them is a classic mistake:

    CAPABILITY -- may this agent hold this tool at all?
                  Enforced at BIND time. The Documentation Agent never receives
                  `git.push`, so no prompt injection, jailbreak, or clever
                  reasoning can make it call one. It does not have it.

    APPROVAL   -- does a human need to say yes to this specific call?
                  Enforced at CALL time, because it depends on the arguments.
                  `git.push` to a feature branch is routine; `git.push --force`
                  to `main` is not. Same tool, different answer.

A prompt saying "do not push to main" is a suggestion to a probabilistic system.
This is a wall.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiswe.domain.ports.tool import Permission
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from aiswe.domain.ports.tool import ToolContext, ToolSpec

logger = get_logger(__name__)

#: (spec, args) -> True if this specific call needs a human.
ApprovalPredicate = Callable[["ToolSpec", dict[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool = False
    reason: str = ""


# --- Permissions that ALWAYS require a human, whatever the tool claims -------
# A tool author cannot opt out of these by setting requires_approval=False.
# Defence in depth: the policy, not the tool, has the last word.
_ALWAYS_APPROVED: frozenset[Permission] = frozenset(
    {
        Permission.VCS_PUSH,  # code leaving the sandbox for a real remote
        Permission.NETWORK,  # egress: the exfiltration path
    }
)


@dataclass(slots=True)
class ToolPolicy:
    """Decides whether a tool call may proceed, and whether a human must approve.

    `autonomous_mode` is the one dial an operator turns. Even set to True, the
    `_ALWAYS_APPROVED` permissions still gate -- "autonomous" must never mean
    "may push force-with-lease to main at 3am unattended".
    """

    autonomous_mode: bool = False
    extra_predicates: list[ApprovalPredicate] = field(default_factory=list)

    def check(
        self,
        spec: ToolSpec,
        *,
        capabilities: frozenset[str],
        args: dict[str, Any],
        ctx: ToolContext,
    ) -> PolicyDecision:
        # --- Gate 1: capability ---------------------------------------------
        # A tool with no tags is universal (e.g. `mem.recall`). Otherwise the
        # agent must hold at least one of its tags.
        if spec.tags and not (spec.tags & capabilities):
            reason = (
                f"Agent {ctx.agent!r} holds capabilities {sorted(capabilities)} "
                f"but {spec.name!r} requires one of {sorted(spec.tags)}."
            )
            logger.warning("tool.denied", tool=spec.name, agent=ctx.agent, reason=reason)
            return PolicyDecision(allowed=False, reason=reason)

        # --- Gate 2: approval ------------------------------------------------
        if ctx.approved:
            return PolicyDecision(allowed=True, reason="Previously approved by a human.")

        needs_approval = (
            spec.requires_approval
            or bool(spec.permissions & _ALWAYS_APPROVED)
            or any(p(spec, args) for p in self.extra_predicates)
        )

        # Autonomous mode waives *discretionary* approvals only.
        if needs_approval and self.autonomous_mode:
            hard_gated = bool(spec.permissions & _ALWAYS_APPROVED)
            if not hard_gated:
                return PolicyDecision(
                    allowed=True,
                    reason="Approval waived: autonomous mode.",
                )

        if needs_approval:
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                reason=f"{spec.name} requires human approval before executing.",
            )

        return PolicyDecision(allowed=True)


# --- Reusable predicates -----------------------------------------------------
# Registered by later milestones. Each is a pure function of (spec, args), so it
# is trivially unit-testable without a running graph.


def force_push_predicate(spec: ToolSpec, args: dict[str, Any]) -> bool:
    """Force-pushing is destructive and irreversible. Always ask. (M6)"""
    return spec.name.startswith("git.") and bool(args.get("force"))


def protected_branch_predicate(spec: ToolSpec, args: dict[str, Any]) -> bool:
    """Never touch a protected branch without a human. (M6)"""
    protected = {"main", "master", "production", "release"}
    return spec.name.startswith("git.") and str(args.get("branch", "")) in protected


def destructive_shell_predicate(spec: ToolSpec, args: dict[str, Any]) -> bool:
    """A crude but effective backstop on shell commands. (M3)

    Not a security boundary -- the sandbox is. This is a guard against the agent
    doing something catastrophic *inside* a workspace it legitimately owns.
    """
    if spec.name != "shell.run":
        return False
    command = str(args.get("command", "")).lower()
    dangerous = ("rm -rf", "mkfs", "dd if=", ":(){", "chmod -r 777", "> /dev/sd")
    return any(token in command for token in dangerous)


DEFAULT_PREDICATES: list[ApprovalPredicate] = [
    force_push_predicate,
    protected_branch_predicate,
    destructive_shell_predicate,
]
