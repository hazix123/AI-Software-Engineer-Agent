"""SQLAlchemy declarative base and mixins.

IMPORTANT ARCHITECTURAL NOTE
----------------------------
The classes here and in `models/` are **persistence models**, not domain
entities. They live in the infrastructure layer on purpose. Domain entities
(`domain/entities/`) are plain, framework-free objects; repositories translate
between the two.

Mixing them would drag SQLAlchemy's session/lazy-loading semantics into the
business rules and quietly break the dependency rule. Every project that skips
this step regrets it around month four.

The naming convention below is load-bearing: without deterministic constraint
names, Alembic autogenerate emits unnamed constraints that it cannot later
drop, and migrations become un-revertible.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, func, text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Root of every ORM model. Alembic autogenerates against `Base.metadata`."""

    metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        """Shallow column dump. Debugging aid only -- never a serialisation path."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self) -> str:
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk}>"


class UUIDPrimaryKeyMixin:
    """UUID primary keys.

    Chosen over bigserial so ids can be minted client-side and in workers
    without a round-trip, and so they are non-enumerable across tenants.
    Generated in Postgres via `gen_random_uuid()` (pgcrypto) as a safety net.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )


class TimestampMixin:
    """Server-side timestamps. `updated_at` is maintained by a DB trigger.

    Doing it in the database (rather than in Python) means direct SQL writes,
    migrations, and background jobs cannot forget to bump it.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SoftDeleteMixin:
    """Tombstone column for user-visible aggregates.

    Applied only where an audit trail or undo matters (projects, repositories).
    High-volume append-only tables (events, tool invocations) are never
    soft-deleted -- they are partitioned and dropped.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
