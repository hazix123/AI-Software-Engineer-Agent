"""FastAPI application factory.

`create_app()` takes `Settings` as a parameter instead of reading a global.
That single choice is what lets an integration test spin up the entire app
against a throwaway Postgres/Redis container in one line, with no monkeypatching.

The container's lifecycle is bound to the app's lifespan, so connections are
opened once before the first request and drained on shutdown -- and a process
that cannot reach its database refuses to boot rather than failing later, one
request at a time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from aiswe.api.middleware.error_handler import register_exception_handlers
from aiswe.api.middleware.request_id import RequestIdMiddleware
from aiswe.api.v1 import api_v1_router, root_router
from aiswe.config.constants import API_V1_PREFIX
from aiswe.config.container import Container
from aiswe.config.settings import Settings, get_settings
from aiswe.infrastructure.observability.logging import get_logger

logger = get_logger(__name__)


def _lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Bind the container's lifecycle to the app's.

    Connections open once, before the first request, and drain on shutdown.
    If a dependency is unreachable, `startup()` raises and the process dies --
    which is correct. A pod that cannot reach Postgres must never accept
    traffic.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = Container.build(settings)
        await container.startup()
        app.state.container = container
        try:
            yield
        finally:
            await container.shutdown()

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. The only entrypoint -- uvicorn, tests, and workers
    all go through here, so there is exactly one wiring path."""
    settings = settings or get_settings()

    app = FastAPI(
        title="AI Software Engineer Agent",
        description="Multi-agent autonomous software engineering platform.",
        version=settings.app.version,
        root_path=settings.app.root_path,
        # Interactive docs are useful in dev and an information leak in prod.
        docs_url="/docs" if not settings.app.is_production else None,
        redoc_url="/redoc" if not settings.app.is_production else None,
        openapi_url="/openapi.json" if not settings.app.is_production else None,
        default_response_class=ORJSONResponse,
        lifespan=_lifespan(settings),
    )

    # --- Middleware ----------------------------------------------------------
    # Order matters: Starlette applies these in reverse, so RequestId runs
    # OUTERMOST and every subsequent log line carries the correlation id.
    if settings.app.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.app.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(RequestIdMiddleware)
    # --- Added in M1: IdempotencyMiddleware, RateLimitMiddleware, AuthMiddleware

    # --- Errors --------------------------------------------------------------
    register_exception_handlers(app)

    # --- Routes --------------------------------------------------------------
    app.include_router(root_router)
    app.include_router(api_v1_router, prefix=API_V1_PREFIX)

    logger.info("app.created", env=settings.app.env, version=settings.app.version)
    return app
