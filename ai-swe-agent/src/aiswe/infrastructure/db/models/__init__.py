"""ORM model registry.

Every model MUST be imported here. Alembic autogenerate walks
`Base.metadata`, which is only populated as a side effect of importing the
module that defines the model. A model that is not imported here is invisible
to migrations -- a silent, and very annoying, failure.
"""

from aiswe.infrastructure.db.base import Base
from aiswe.infrastructure.db.models.identity import (
    ApiKey,
    AuditLog,
    Organization,
    Plan,
    RefreshToken,
    User,
    UserRole,
)
from aiswe.infrastructure.db.models.project import (
    IndexStatus,
    Project,
    Repository,
    VcsProvider,
)

__all__ = [
    "ApiKey",
    "AuditLog",
    "Base",
    "IndexStatus",
    "Organization",
    "Plan",
    "Project",
    "RefreshToken",
    "Repository",
    "User",
    "UserRole",
    "VcsProvider",
]
