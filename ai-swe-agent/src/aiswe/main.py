"""Process entrypoint.

    uvicorn aiswe.main:app

Deliberately thin. All wiring lives in `api/app.py` -> `config/container.py`,
so the worker entrypoint (M5) can reuse the exact same object graph without
importing anything web-related.
"""

from __future__ import annotations

from aiswe.api.app import create_app
from aiswe.config.settings import Environment, get_settings

settings = get_settings()
app = create_app(settings)


def run() -> None:
    """Local development server. Production runs uvicorn/gunicorn directly."""
    import uvicorn

    uvicorn.run(
        "aiswe.main:app",
        host=settings.app.host,
        port=settings.app.port,
        # Never reload outside local: the reloader forks, which breaks
        # connection pools and doubles memory.
        reload=settings.app.env is Environment.LOCAL,
        log_config=None,  # structlog owns logging
        access_log=False,  # RequestIdMiddleware emits access logs instead
    )


if __name__ == "__main__":
    run()
