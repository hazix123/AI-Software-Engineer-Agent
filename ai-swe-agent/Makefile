.DEFAULT_GOAL := help
.PHONY: help install up down logs migrate revision downgrade lint format typecheck test test-unit test-integration check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------------
install: ## Install all dependencies and git hooks
	uv sync --all-extras --dev
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

# --- Stack -------------------------------------------------------------------
up: ## Start postgres, redis, migrate, api
	docker compose up -d --build
	@echo "API:  http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

down: ## Stop the stack (volumes preserved)
	docker compose down

reset: ## Stop the stack and DESTROY all data
	docker compose down -v

logs: ## Tail API logs
	docker compose logs -f api

shell: ## Open a psql shell
	docker compose exec postgres psql -U aiswe -d aiswe

# --- Migrations --------------------------------------------------------------
migrate: ## Apply migrations to head
	uv run alembic upgrade head

revision: ## Autogenerate a migration:  make revision m="add runs table"
	uv run alembic revision --autogenerate -m "$(m)"

downgrade: ## Roll back one revision
	uv run alembic downgrade -1

history: ## Show migration history
	uv run alembic history --verbose

# --- Quality -----------------------------------------------------------------
lint: ## Ruff
	uv run ruff check --fix .

format: ## Black
	uv run black .

typecheck: ## mypy strict
	uv run mypy src/aiswe

# --- Tests -------------------------------------------------------------------
test-unit: ## Unit tests (no I/O)
	uv run pytest tests/unit -m unit

test-integration: ## Integration tests (needs `make up`)
	uv run pytest tests/integration -m integration

test: ## All tests with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint format typecheck test ## Everything CI runs, locally

# --- Housekeeping ------------------------------------------------------------
clean: ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
