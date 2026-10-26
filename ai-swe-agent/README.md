# AI Software Engineer Agent

Multi-agent autonomous software engineering platform.

**Status: M0 — Foundations.** The skeleton, the ports, and the wiring. No agents yet.

---

## Quickstart

```bash
cp .env.example .env
make install      # deps + git hooks
make up           # postgres (pgvector) + redis + migrations + api
curl localhost:8000/healthz
curl localhost:8000/readyz
open http://localhost:8000/docs
```

Run everything CI runs, locally:

```bash
make check        # lint + format + typecheck + tests
```

---

## Architecture rules

These are enforced in code review. Violating one is a blocking comment, not a nit.

| Rule | Why |
|---|---|
| `domain/` imports **nothing** external | It is the only layer that survives a framework change. If you need Redis in the domain, you actually need a port. |
| ORM models are **not** domain entities | Otherwise SQLAlchemy's session semantics leak into business rules and the boundary is decorative. |
| Ports live in `domain/ports/`, adapters in `infrastructure/` | Dependency inversion. Adapters depend on the domain, never the reverse. |
| `config/container.py` is the **only** file that binds a port to an adapter | One composition root. No service locator, no globals, no hidden wiring. |
| `api/` maps HTTP to use cases and nothing else | Business logic in a router is business logic you cannot reuse from the CLI or a worker. |
| Domain errors, never `HTTPException` | `api/middleware/error_handler.py` is the single translation point to HTTP. |
| Every function is annotated | `mypy --strict` runs in CI. The boundary is only real if it type-checks. |

Dependency direction:

```
api/ ─┐
      ├─► application/ ─► domain/ ◄─ implements ─ infrastructure/
workers/ ┘                              ▲
                                        └── (M5) agents/, tools/
```

---

## Layout

| Path | Layer | Depends on |
|---|---|---|
| `src/aiswe/domain/` | Entities, value objects, **ports**, policies | nothing |
| `src/aiswe/application/` | Use cases, DTOs, transaction boundaries | domain |
| `src/aiswe/infrastructure/` | Postgres, Redis, and (M2+) LLM, sandbox, git, tree-sitter adapters | domain |
| `src/aiswe/api/` | FastAPI routers, middleware, schemas | application |
| `src/aiswe/workers/` | Background workers (M5) | application |
| `src/aiswe/config/` | Settings + **composition root** | everything (by design) |

---

## Common tasks

```bash
make revision m="add runs table"   # autogenerate a migration
make migrate                       # apply to head
make downgrade                     # roll back one
make test-integration              # needs `make up`
make shell                         # psql
```

---

## What lands next

| Milestone | Scope |
|---|---|
| **M1** | JWT auth, RBAC scopes, API keys, rate limiting, idempotency, RLS |
| **M2** | LLM abstraction: Anthropic / OpenAI / Gemini behind one port, with routing and fallback |
| **M3** | Docker sandbox, workspace manager, tool registry + policy engine |
| **M4** | Repository Indexing Agent: tree-sitter, symbol graph, pgvector hybrid search |
| **M5** | Agent core: Supervisor, Planner, Code Analysis, LangGraph checkpointing, SSE |
| **M6** | Write path: Refactoring, Testing, Git + human approval gates |
| **M7** | Security Review, Documentation, Playwright |
| **M8** | Memory Agent and the learning loop |
| **M9** | Evals, hardening, launch |
