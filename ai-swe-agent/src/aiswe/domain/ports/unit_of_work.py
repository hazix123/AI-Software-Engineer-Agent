"""Unit of Work and generic Repository ports.

A use case owns exactly one transaction:

    async with uow:
        org = await uow.organizations.get(org_id)
        ...
        await uow.commit()

The use case never sees a SQLAlchemy Session. Repositories are exposed as
attributes on the UoW so that every repository inside a single `async with`
block shares one transaction and one identity map.

Concrete repositories are registered on the UoW implementation in
`infrastructure/db/unit_of_work.py`. As new aggregates land (M1: Project,
Repository; M5: Run, Plan), they are added to `AbstractUnitOfWork` as
properties -- callers stay decoupled from SQLAlchemy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Protocol, TypeVar
from uuid import UUID

EntityT = TypeVar("EntityT")


class Repository(Protocol[EntityT]):
    """Minimal persistence role for a single aggregate type.

    Kept deliberately narrow (ISP). Query-heavy read models get their own
    dedicated read-only ports rather than bloating this interface.
    """

    async def get(self, entity_id: UUID) -> EntityT | None:
        """Fetch by primary key, or None."""
        ...

    async def add(self, entity: EntityT) -> EntityT:
        """Stage a new aggregate for insertion."""
        ...

    async def delete(self, entity_id: UUID) -> None:
        """Remove (or soft-delete) an aggregate."""
        ...


class AbstractUnitOfWork(ABC):
    """Transaction boundary.

    Exiting the context without an explicit `commit()` rolls back. This makes
    "forgot to commit" a safe failure rather than a partial write.
    """

    async def __aenter__(self) -> AbstractUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Rollback is unconditional and idempotent: if commit() already ran,
        # this is a no-op; if it did not, the transaction is discarded.
        await self.rollback()

    @abstractmethod
    async def commit(self) -> None:
        """Flush and commit the transaction, then dispatch collected events."""

    @abstractmethod
    async def rollback(self) -> None:
        """Discard all staged changes."""
