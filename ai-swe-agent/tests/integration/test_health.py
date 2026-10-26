"""Health endpoint integration tests.

This is the M0 exit criterion made executable: the app boots, the DI container
wires, Postgres and Redis are reachable, and the error envelope is correct.

Requires live services (CI provides them; locally: `docker compose up -d`).
"""

from __future__ import annotations

from http import HTTPStatus

import pytest
from httpx import AsyncClient

from aiswe.config.constants import HEADER_REQUEST_ID, PROBLEM_CONTENT_TYPE

pytestmark = pytest.mark.integration


class TestLiveness:
    async def test_healthz_ok(self, client: AsyncClient) -> None:
        response = await client.get("/healthz")

        assert response.status_code == HTTPStatus.OK
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "aiswe-test"

    async def test_request_id_is_echoed(self, client: AsyncClient) -> None:
        """A caller-supplied id must survive the round trip, for cross-service tracing."""
        response = await client.get("/healthz", headers={HEADER_REQUEST_ID: "trace-abc-123"})

        assert response.headers[HEADER_REQUEST_ID] == "trace-abc-123"

    async def test_request_id_is_minted_when_absent(self, client: AsyncClient) -> None:
        response = await client.get("/healthz")

        assert response.headers.get(HEADER_REQUEST_ID)


class TestReadiness:
    async def test_readyz_reports_all_dependencies(self, client: AsyncClient) -> None:
        response = await client.get("/readyz")

        assert response.status_code == HTTPStatus.OK
        body = response.json()
        assert body["status"] == "ready"

        names = {dep["name"] for dep in body["dependencies"]}
        assert names == {"database", "redis"}
        assert all(dep["healthy"] for dep in body["dependencies"])


class TestErrorEnvelope:
    async def test_unknown_route_returns_problem_detail(self, client: AsyncClient) -> None:
        """Even a 404 must use the RFC 9457 shape -- one error format, no exceptions."""
        response = await client.get("/v1/does-not-exist")

        assert response.status_code == HTTPStatus.NOT_FOUND
        assert response.headers["content-type"].startswith(PROBLEM_CONTENT_TYPE)

        body = response.json()
        assert body["status"] == HTTPStatus.NOT_FOUND
        assert body["type"].startswith("https://errors.aiswe.dev/")
        assert body["request_id"]
