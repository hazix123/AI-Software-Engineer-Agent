"""Project and Repository persistence models.

`Project` is the *memory scope boundary* defined in section 7 of the design:
semantic and procedural memories are keyed to a project, so two projects in the
same org never contaminate each other's learned conventions.

`Repository.credentials_ref` holds a POINTER into a secret manager -- never a
token. Nothing in this database is ever a live credential.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aiswe.infrastructure.db.base import (
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from aiswe.infrastructure.db.models.identity import Organization


class VcsProvider(StrEnum):
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    LOCAL = "local"


class IndexStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class Project(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "projects"

    organization_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_branch: Mapped[str] = mapped_column(String(200), nullable=False, default="main")

    # Project conventions injected into every agent prompt from M5:
    # test command, build command, style guide, forbidden paths, PR etiquette.
    conventions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    organization: Mapped[Organization] = relationship(back_populates="projects")
    repositories: Mapped[list[Repository]] = relationship(back_populates="project")

    __table_args__ = (
        Index("uq_projects_organization_id_name", "organization_id", "name", unique=True),
    )


class Repository(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "repositories"

    project_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(20), nullable=False, default=VcsProvider.GITHUB)
    remote_url: Mapped[str] = mapped_column(String(500), nullable=False)

    # Reference into a secret manager (e.g. "vault://aiswe/repo/<id>/token").
    # NEVER the credential itself.
    credentials_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)

    default_branch: Mapped[str] = mapped_column(String(200), nullable=False, default="main")
    languages: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Index state. Consumed by the Repository Indexing Agent in M4; the delta
    # reindex diffs `last_indexed_commit..HEAD`.
    last_indexed_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    index_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=IndexStatus.PENDING
    )

    project: Mapped[Project] = relationship(back_populates="repositories")

    __table_args__ = (
        Index("ix_repositories_project_id", "project_id"),
        Index("ix_repositories_index_status", "index_status"),
        CheckConstraint(
            "provider IN ('github','gitlab','bitbucket','local')",
            name="provider_valid",
        ),
        CheckConstraint(
            "index_status IN ('pending','indexing','ready','failed')",
            name="index_status_valid",
        ),
    )
