"""Architecture tests — the dependency rule, enforced by CI.

An architecture document rots the moment someone is in a hurry. A test does
not. This module walks the AST of every source file and fails the build if a
layer imports something it is forbidden to know about.

This is how Clean Architecture actually dies in practice: not in a design
review, but at 6pm on a Friday when someone imports SQLAlchemy into the domain
because it was faster than adding a port. This test is the thing that stops it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2] / "src" / "aiswe"

# Third-party packages no inner layer may ever touch.
FRAMEWORKS = (
    "fastapi",
    "starlette",
    "sqlalchemy",
    "redis",
    "alembic",
    "docker",
    "git",
    "tree_sitter",
    "playwright",
    "langgraph",
    "anthropic",
    "openai",
)

# layer -> module prefixes it is FORBIDDEN to import.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": (
        *FRAMEWORKS,
        "aiswe.infrastructure",
        "aiswe.api",
        "aiswe.application",
        "aiswe.agents",
        "aiswe.tools",
        "aiswe.config",
    ),
    # The application layer orchestrates use cases. It may depend on the domain
    # and nothing outward -- otherwise a use case cannot be driven from a
    # worker or the CLI without dragging in a web framework.
    "application": (
        "fastapi",
        "starlette",
        "sqlalchemy",
        "redis",
        "aiswe.infrastructure",
        "aiswe.api",
    ),
    # Infrastructure implements ports. It must not reach back into the
    # presentation layer or into use-case internals.
    "infrastructure": ("fastapi", "starlette", "aiswe.api"),
}


def _iter_modules(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    """Every module name imported by this file, including inside TYPE_CHECKING."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_respects_the_dependency_rule(layer: str) -> None:
    banned = FORBIDDEN[layer]
    violations: list[str] = []

    for module in _iter_modules(layer):
        for imported in _imported_modules(module):
            for prefix in banned:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    rel = module.relative_to(SRC)
                    violations.append(f"{layer}/{rel}  imports  {imported}")

    assert not violations, (
        f"\nThe `{layer}` layer violated the dependency rule:\n\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\nIf you need this capability, define a PORT in domain/ports/ and "
        "implement it in infrastructure/ instead.\n"
    )


def test_domain_is_pure() -> None:
    """The domain must be importable with zero third-party packages installed.

    This is the strongest statement of the dependency rule: strip every library
    from the environment and the business core still loads.
    """
    third_party: set[str] = set()

    for module in _iter_modules("domain"):
        for imported in _imported_modules(module):
            root = imported.split(".")[0]
            if root in {"aiswe", "__future__"}:
                continue
            # Pydantic is the one sanctioned exception: it is used purely for
            # declarative validation of value objects and carries no I/O.
            if root == "pydantic":
                continue
            if root not in _STDLIB:
                third_party.add(imported)

    assert not third_party, (
        f"The domain layer imports non-stdlib packages: {sorted(third_party)}"
    )


_STDLIB = frozenset(
    {
        "abc",
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "pathlib",
        "types",
        "typing",
        "uuid",
    }
)
