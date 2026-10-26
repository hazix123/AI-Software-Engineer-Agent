"""Memory ports.

The tiers from section 7 of the design, as interfaces:

    WORKING      AgentState in the graph        (lives in state.py, not here)
    SHORT-TERM   Redis scratchpad, TTL-bound    (ScratchpadStore -- shipped now)
    LONG-TERM    Postgres + pgvector            (MemoryStore    -- M8)

The framework ships working and short-term memory fully implemented. Long-term
recall is M8, so `MemoryStore` is declared now and backed by a null adapter.

That is not a cosmetic choice: the Memory Agent, the retrieval call sites, and
the prompt-assembly code are all written against the real interface today. When
M8 lands, one line changes in the DI container and nothing else moves.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class MemoryType(StrEnum):
    """What kind of thing is being remembered.

    The distinction is not academic -- it governs the WRITE path. A procedural
    memory ("this repo runs tests with `make test`") may only be promoted from
    a run whose tests actually PASSED. Otherwise the agent faithfully learns its
    own mistakes and repeats them with growing confidence.
    """

    SEMANTIC = "semantic"  # "what is true about this codebase"
    PROCEDURAL = "procedural"  # "how we do things here"
    EPISODIC = "episodic"  # "what happened in run #412"
    PREFERENCE = "preference"  # "this user wants small PRs"


class MemoryScope(StrEnum):
    """Blast radius. Enforced in the repository layer AND by row-level security."""

    GLOBAL = "global"
    ORGANIZATION = "organization"
    PROJECT = "project"
    REPOSITORY = "repository"
    RUN = "run"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One remembered fact, with provenance.

    `source_run_id` is not optional in spirit: a bad memory that cannot be
    traced back to the run that produced it cannot be purged with confidence,
    and an un-purgeable bad memory poisons every future run on that project.
    """

    content: str
    type: MemoryType
    scope: MemoryScope

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    title: str = ""
    structured: dict[str, Any] = field(default_factory=dict)

    organization_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    repository_id: uuid.UUID | None = None
    source_run_id: uuid.UUID | None = None

    confidence: float = 0.5
    importance: float = 0.5
    usage_count: int = 0
    last_used_at: datetime | None = None
    expires_at: datetime | None = None

    def as_context(self) -> str:
        """How this memory appears in a prompt."""
        prefix = f"[{self.type}]"
        return f"{prefix} {self.title + ': ' if self.title else ''}{self.content}"


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """A recall request."""

    text: str
    project_id: uuid.UUID | None = None
    repository_id: uuid.UUID | None = None
    types: frozenset[MemoryType] = field(default_factory=frozenset)
    limit: int = 10
    min_confidence: float = 0.3


@runtime_checkable
class ScratchpadStore(Protocol):
    """SHORT-TERM memory: a TTL-bound key/value scratchpad, scoped to a run.

    For things too big for the graph state but too transient for the database:
    a cached tool result, an intermediate analysis, a partial diff. TTL-bound so
    a crashed run cannot leak memory forever.
    """

    async def get(self, run_id: uuid.UUID, key: str) -> str | None:
        ...

    async def set(
        self,
        run_id: uuid.UUID,
        key: str,
        value: str,
        *,
        ttl: int | None = None,
    ) -> None:
        ...

    async def delete(self, run_id: uuid.UUID, key: str) -> None:
        ...

    async def keys(self, run_id: uuid.UUID) -> list[str]:
        ...

    async def clear(self, run_id: uuid.UUID) -> None:
        """Drop the whole scratchpad. Called when a run reaches a terminal state."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """LONG-TERM memory. Backed by a null adapter until M8.

    `remember()` is intentionally the narrow point: all writes funnel through
    one method so dedupe, contradiction resolution, and confidence scoring have
    exactly one place to live.
    """

    async def recall(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Retrieve relevant memories, ranked."""
        ...

    async def remember(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a memory, deduplicating against what is already known."""
        ...

    async def forget(self, memory_id: uuid.UUID) -> None:
        ...

    async def reinforce(self, memory_id: uuid.UUID) -> None:
        """Mark a memory as used. Drives recency/usage ranking and retention."""
        ...
