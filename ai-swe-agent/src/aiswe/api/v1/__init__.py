"""API v1 router aggregation.

Every router is mounted here, so the app factory never changes as the surface
grows.

Health is mounted at the ROOT, not under /v1: probes must stay stable across
API versions. A liveness check that breaks when you ship v2 is worse than no
liveness check.
"""

from fastapi import APIRouter

from aiswe.api.v1 import health

# Unversioned operational endpoints.
root_router = APIRouter()
root_router.include_router(health.router)

# Versioned application endpoints. The /v1 prefix is applied in `api/app.py`.
api_v1_router = APIRouter()

# Mounted as milestones land:
#   M1  auth, projects
#   M4  repositories (index, symbols, search)
#   M5  runs (create, stream, cancel, resume)
#   M6  approvals (human-in-the-loop gates)
#   M8  memories (browse, teach, correct, forget)

__all__ = ["api_v1_router", "root_router"]
