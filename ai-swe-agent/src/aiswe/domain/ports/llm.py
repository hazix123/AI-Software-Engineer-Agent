"""LLM provider port.

Concrete adapters (Anthropic, OpenAI, Gemini) and the model router land in M2.
The port is declared now so the agent framework can be built and fully tested
against it -- a stub implementation lives in `infrastructure/llm/`.

`structured()` is the method that matters. Agents must return validated Pydantic
models, never free-form prose: a supervisor that "usually" replies with a valid
agent name is a supervisor that takes production down on a Tuesday.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversation turn."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one call. Feeds TokenBudget and the cost ledger."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Completion:
    """A model response, plus what it cost."""

    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    finish_reason: str = "stop"


@dataclass(frozen=True, slots=True)
class StructuredCompletion[SchemaT: BaseModel]:
    """A validated, typed model response."""

    value: SchemaT
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-agnostic model access.

    Liskov matters here more than anywhere else in the system: a caller MUST
    NOT be able to tell Anthropic from Gemini. Same request shape, same error
    taxonomy (LLMProviderError), same usage accounting. The moment a caller
    needs an `if provider == "openai"` branch, the abstraction has failed and
    the router cannot fall back.
    """

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> Completion:
        """Free-form text completion."""
        ...

    async def structured(
        self,
        messages: list[Message],
        *,
        schema: type[SchemaT],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> StructuredCompletion[SchemaT]:
        """Completion validated against a Pydantic schema.

        Raises:
            AgentOutputError: if the response cannot be coerced into `schema`.
                Retryable -- the caller re-prompts with the validation error
                appended, which recovers the large majority of the time.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text -> vector. Consumed by the indexing and memory subsystems (M4/M8)."""

    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        ...

    @property
    def dimensions(self) -> int:
        ...


def render_schema_instruction(schema: type[BaseModel]) -> str:
    """Render a JSON-schema instruction block.

    Used by adapters whose provider lacks native structured output, so the
    behaviour is identical across providers rather than "better on Anthropic".
    """
    import json

    return (
        "Respond with a single JSON object and nothing else. "
        "No prose, no markdown fences, no explanation.\n\n"
        "It must validate against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


def parse_json_response(raw: str) -> Any:
    """Salvage JSON from a model response.

    Models are told not to wrap output in markdown fences. Models sometimes do
    it anyway. Stripping them is one line here versus a retry that costs money
    and latency -- so we strip, and only fail if the payload is genuinely
    unparseable.
    """
    import json

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening ``` (possibly ```json)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)
