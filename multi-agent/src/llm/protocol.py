"""Provider-agnostic LLM client protocol.

Agents call only this interface. The concrete adapter (OpenAI, Anthropic, etc.)
is injected at startup. Structured outputs are first-class — callers pass a
response_schema and get back a validated Pydantic instance with no regex.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    content: str | None
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "length"
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0    # Anthropic prompt cache hit tokens
    cache_write_tokens: int = 0   # Anthropic prompt cache write tokens


@runtime_checkable
class LLMClient(Protocol):
    """Stateless provider-agnostic chat completion client."""

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        max_tokens: int = 512,
    ) -> tuple[BaseModel, LLMResponse]:
        """Structured completion using json_object mode + Pydantic validation.

        The schema is injected into the system prompt. The provider is asked to
        return only valid JSON matching the schema. Response is validated with
        Pydantic before returning — no regex, no json.loads without validation.
        """
        ...
