"""OpenAI SDK adapter — targets Claude via Anthropic's OpenAI-compatible endpoint.

Configure via env:
    OPENAI_API_KEY    = your Anthropic API key
    OPENAI_BASE_URL   = https://api.anthropic.com/v1/

Prompt caching: when OPENAI_BASE_URL points to Anthropic's endpoint, the system
prompt is passed via extra_body as a structured content block with cache_control
set to "ephemeral". This enables Anthropic's prompt cache and cuts input token
costs by ~80-90% on repeated calls with the same long system prompt.

For structured outputs, we use json_object mode (compatible with Anthropic's
OpenAI endpoint) with the JSON schema injected into the system prompt, then
validate the response with Pydantic — zero regex.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.llm.protocol import LLMClient, LLMResponse, ToolCall, ToolDefinition

_SCHEMA_INSTRUCTION = (
    "\n\nYou MUST respond with valid JSON that matches this exact schema "
    "(no markdown, no explanation, just the JSON object):\n{schema}"
)

# Enable prompt caching when pointed at Anthropic's endpoint
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
_USE_ANTHROPIC_CACHING = "anthropic.com" in _BASE_URL


class OpenAILLMClient:
    """Concrete LLMClient using the OpenAI Python SDK.

    Satisfies the LLMClient Protocol — inject this wherever LLMClient is typed.
    """

    def __init__(self, client: AsyncOpenAI, prompt_caching: bool | None = None) -> None:
        self._client = client
        # Default: enable caching when base URL is Anthropic's compat endpoint
        self._caching = prompt_caching if prompt_caching is not None else _USE_ANTHROPIC_CACHING

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        full_messages, extra_body = self._build_messages(system, messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
        response = await self._client.chat.completions.create(**kwargs)
        return _adapt_response(response)

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        response_schema: type[BaseModel],
        max_tokens: int = 512,
    ) -> tuple[BaseModel, LLMResponse]:
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        augmented_system = system + _SCHEMA_INSTRUCTION.format(schema=schema_json)
        full_messages, extra_body = self._build_messages(augmented_system, messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await self._client.chat.completions.create(**kwargs)
        raw = _adapt_response(response)
        parsed = response_schema.model_validate_json(_strip_fences(raw.content or "{}"))
        return parsed, raw

    def _build_messages(
        self, system: str, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Return (messages_list, extra_body).

        When Anthropic prompt caching is enabled, the system prompt is passed
        via extra_body as a structured content block with cache_control, and is
        NOT included in the messages list (to avoid duplication).
        """
        if self._caching:
            extra_body = {
                "system": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            }
            return list(messages), extra_body
        else:
            return [{"role": "system", "content": system}, *messages], None


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap JSON in."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _to_openai_tool(t: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


def _adapt_response(response: Any) -> LLMResponse:
    choice = response.choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    if msg.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in msg.tool_calls
        ]
    finish = choice.finish_reason
    stop = "tool_use" if finish == "tool_calls" else "end_turn" if finish == "stop" else (finish or "end_turn")

    usage = response.usage
    # Anthropic compat endpoint returns cache stats in usage model_extra
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    return LLMResponse(
        content=msg.content,
        tool_calls=tool_calls,
        stop_reason=stop,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


# Runtime check that OpenAILLMClient satisfies the Protocol
assert isinstance(OpenAILLMClient.__new__(OpenAILLMClient), LLMClient)
