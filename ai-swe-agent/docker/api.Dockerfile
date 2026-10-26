# syntax=docker/dockerfile:1.9
# =============================================================================
# API image. Multi-stage: the build toolchain never reaches the runtime layer.
#
# The same image runs the API and (from M5) the agent worker -- only the CMD
# differs. One image means one thing to scan, sign, and promote.
# =============================================================================

ARG PYTHON_VERSION=3.12

# --- Stage 1: builder --------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv: an order of magnitude faster than pip, and it resolves deterministically.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /build

# Copy only the manifest first. This layer is cached until dependencies
# actually change, so a source-only edit does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# --- Stage 2: runtime --------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# Only runtime libs. No gcc, no build headers.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libpq5 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root from the start. A container that runs as root is one CVE away from
# being a host compromise -- and this service will eventually sit next to a
# Docker socket.
RUN groupadd --system --gid 1001 aiswe \
    && useradd --system --uid 1001 --gid aiswe --create-home aiswe

WORKDIR /app

COPY --from=builder --chown=aiswe:aiswe /opt/venv /opt/venv
COPY --chown=aiswe:aiswe src/ ./src/
COPY --chown=aiswe:aiswe migrations/ ./migrations/
COPY --chown=aiswe:aiswe alembic.ini pyproject.toml ./

USER aiswe

EXPOSE 8000

# Liveness only. Probing /readyz here would restart a healthy container during
# a transient Redis blip.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# tini reaps zombies and forwards SIGTERM, so graceful shutdown actually works.
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "aiswe.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
