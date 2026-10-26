"""SQLAlchemy Unit of Work.

This is the wall between the application layer and the driver. Two jobs:

1. Own exactly one transaction per use case.
2. Translate every `SQLAlchemyError` into an `InfrastructureError`, so a
   use case never has to import (or catch) anything from SQLAlchemy.

Domain events raised during a transaction are queued and dispatched only
*after* a successful commit. Emitting them before commit is how you end up
publishing an event for a run that was subsequently rolled back.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from aiswe.domain.errors import ConflictError, InfrastructureError
from aiswe.domain.ports.unit_of_work import AbstractUnitOfWork
from aiswe.infrastructure.observability.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)


class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    """Concrete UoW bound to one AsyncSession.

    Repositories are attached lazily as milestones land. At M0 there are none:
    the transaction boundary exists and is tested, but no aggregate is
    persisted through it yet.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._events: list[Any] = []

    # --- context management --------------------------------------------------

    async def __aenter__(self) -> Self:
        self._session = self._session_factory()
        # Repositories are constructed here from M1 onward, e.g.:
        #   self.organizations = SqlAlchemyOrganizationRepository(self._session)
        #   self.projects = SqlAlchemyProjectRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            # Unconditional rollback: a no-op if commit() already succeeded,
            # a safety net if the caller forgot to commit or raised.
            await self.rollback()
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

    # --- session access ------------------------------------------------------

    @property
    def session(self) -> AsyncSession:
        if self._session is None:
            msg = "UnitOfWork used outside of an `async with` block."
            raise InfrastructureError(msg, code="uow_not_started")
        return self._session

    # --- transaction ---------------------------------------------------------

    async def commit(self) -> None:
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.rollback()
            logger.warning("uow.integrity_error", error=str(exc.orig))
            msg = "The operation violates a uniqueness or referential constraint."
            raise ConflictError(msg, details={"cause": str(exc.orig)}) from exc
        except SQLAlchemyError as exc:
            await self.rollback()
            logger.exception("uow.commit_failed")
            msg = "Failed to commit the transaction."
            raise InfrastructureError(msg, code="db_commit_failed") from exc

        # Only now is it safe to tell the world what happened.
        await self._dispatch_events()

    async def rollback(self) -> None:
        if self._session is None:
            return
        try:
            await self._session.rollback()
        except SQLAlchemyError as exc:  # pragma: no cover - defensive
            logger.warning("uow.rollback_failed", error=str(exc))
        finally:
            self._events.clear()

    # --- domain events -------------------------------------------------------

    def add_event(self, event: Any) -> None:
        """Queue a domain event for post-commit dispatch."""
        self._events.append(event)

    async def _dispatch_events(self) -> None:
        """Publish queued events. Wired to the Redis event bus in M5."""
        for event in self._events:
            logger.debug("domain_event.emitted", event=type(event).__name__)
        self._events.clear()
