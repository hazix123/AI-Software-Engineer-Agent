"""Operational CLI.

    aiswe serve
    aiswe check-config
    aiswe check-health

Operational commands only. No business logic lives here -- the CLI is just
another delivery mechanism sitting on top of the same use cases, exactly like
the HTTP API.
"""

from __future__ import annotations

import asyncio

import typer

from aiswe.config.container import Container
from aiswe.config.settings import get_settings

app = typer.Typer(name="aiswe", help="AI Software Engineer Agent — operations CLI.")


@app.command()
def serve() -> None:
    """Run the API server."""
    from aiswe.main import run

    run()


@app.command("check-config")
def check_config() -> None:
    """Validate configuration and print the resolved settings (secrets masked)."""
    settings = get_settings()
    typer.echo(settings.model_dump_json(indent=2))
    typer.secho("configuration OK", fg=typer.colors.GREEN)


@app.command("check-health")
def check_health() -> None:
    """Verify Postgres and Redis connectivity. Exits non-zero on failure."""

    async def _check() -> bool:
        container = Container.build(get_settings())
        results = {
            name: await check.ping() for name, check in container.health_checks().items()
        }
        await container.shutdown()
        for name, healthy in results.items():
            colour = typer.colors.GREEN if healthy else typer.colors.RED
            typer.secho(f"{name}: {'ok' if healthy else 'FAILED'}", fg=colour)
        return all(results.values())

    if not asyncio.run(_check()):
        raise typer.Exit(code=1)


# --- Registered as milestones land -------------------------------------------
# @app.command("index")   -> trigger a repository index          (M4)
# @app.command("worker")  -> run the agent worker pool           (M5)
# @app.command("eval")    -> run the SWE-bench-style eval suite  (M9)


if __name__ == "__main__":
    app()
