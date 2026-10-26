"""Constants. No magic strings anywhere else in the codebase."""

from __future__ import annotations

from typing import Final

# --- HTTP headers ------------------------------------------------------------
HEADER_REQUEST_ID: Final = "X-Request-ID"
HEADER_IDEMPOTENCY_KEY: Final = "Idempotency-Key"
HEADER_RATELIMIT_LIMIT: Final = "RateLimit-Limit"
HEADER_RATELIMIT_REMAINING: Final = "RateLimit-Remaining"
HEADER_RATELIMIT_RESET: Final = "RateLimit-Reset"

# --- RFC 9457 problem-detail type URIs --------------------------------------
PROBLEM_TYPE_BASE: Final = "https://errors.aiswe.dev"
PROBLEM_CONTENT_TYPE: Final = "application/problem+json"

# --- Redis key namespaces ----------------------------------------------------
# Every key is prefixed. This keeps a shared Redis instance debuggable and
# makes targeted flushes possible.
CACHE_NS: Final = "cache"
LOCK_NS: Final = "lock"
RATELIMIT_NS: Final = "rl"
IDEMPOTENCY_NS: Final = "idem"
RUN_EVENTS_NS: Final = "run"  # run:{id}:events   (M5)
QUEUE_NS: Final = "queue"  # queue:runs        (M5)

# --- Structured-log / trace context keys ------------------------------------
CTX_REQUEST_ID: Final = "request_id"
CTX_TRACE_ID: Final = "trace_id"
CTX_USER_ID: Final = "user_id"
CTX_ORG_ID: Final = "organization_id"
CTX_RUN_ID: Final = "run_id"

# --- Pagination --------------------------------------------------------------
DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100

# --- Naming ------------------------------------------------------------------
API_V1_PREFIX: Final = "/v1"
