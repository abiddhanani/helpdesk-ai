"""BaseSpecialistAgent — shared REASON -> ACT -> OBSERVE loop for all specialist agents.

Key changes vs exercise-3 original:
- LLMClient protocol replaces AsyncAnthropic (provider-agnostic)
- AgentOutput parsed via complete_structured() — no regex, no json.loads
- AgentMemory replaces raw message list (token-based compaction)
- Tool calls within a single response run in parallel via asyncio.gather()
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from mcp.client.session import ClientSession

from src.context import SharedContext
from src.llm.protocol import LLMClient, LLMResponse, ToolDefinition
from src.memory.agent_memory import AgentMemory
from src.messages import AgentMessage, AgentName, AgentOutput, Intent
from src.trace import TraceLogger

MAX_LOOP_ITERATIONS = 20
MAX_RETRIES = 3

_OUTPUT_SYSTEM_SUFFIX = """

When you have finished all tool calls and are ready to give your final answer,
respond with a JSON object (no markdown fences) matching this schema exactly:
{schema}
"""


class BaseSpecialistAgent:
    name: AgentName = "orchestrator"
    allowed_tools: list[str] = []
    SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        session: ClientSession,
        llm_client: LLMClient,
        agent_model: str,
        router_model: str,
        trace_logger: TraceLogger,
        context: SharedContext,
    ) -> None:
        self.session = session
        self._llm = llm_client
        self._model = agent_model
        self.trace = trace_logger
        self.context = context
        self.tools: list[dict[str, Any]] = []
        self._tool_defs: list[ToolDefinition] = []
        self._memory = AgentMemory(self.name, llm_client, router_model)

    async def discover_tools(self) -> None:
        result = await self.session.list_tools()
        self._tool_defs = [
            ToolDefinition(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema,
            )
            for t in result.tools
            if t.name in self.allowed_tools
        ]
        # Keep raw dict form for backward compat with trace logging
        self.tools = [
            {"name": td.name, "description": td.description, "input_schema": td.parameters}
            for td in self._tool_defs
        ]

    async def run(self, task: str, trace_id: str) -> AgentMessage:
        self.trace.hop_start(self.name, task)
        schema_json = AgentOutput.model_json_schema()
        import json as _json
        system = self.SYSTEM_PROMPT + _OUTPUT_SYSTEM_SUFFIX.format(
            schema=_json.dumps(schema_json, indent=2)
        )

        await self._memory.add_message({"role": "user", "content": task})

        for _ in range(MAX_LOOP_ITERATIONS):
            messages = await self._memory.get_messages()
            response: LLMResponse = await self._llm.complete(
                model=self._model,
                system=system,
                messages=messages,
                tools=self._tool_defs if self._tool_defs else None,
                max_tokens=4096,
            )
            self.trace.llm_tokens(
                self.name, self._model, response.input_tokens, response.output_tokens,
                response.cache_read_tokens, response.cache_write_tokens,
            )

            if response.content:
                self.trace.reasoning(self.name, response.content)

            if response.stop_reason == "end_turn":
                content = response.content or ""
                try:
                    from src.llm.openai_client import _strip_fences
                    output = AgentOutput.model_validate_json(_strip_fences(content))
                except Exception:
                    # Fallback: ask the model to produce structured output explicitly
                    try:
                        output, struct_response = await self._llm.complete_structured(
                            model=self._model,
                            system=system,
                            messages=[
                                *messages,
                                {"role": "assistant", "content": content},
                                {
                                    "role": "user",
                                    "content": "Now produce your final structured JSON decision.",
                                },
                            ],
                            response_schema=AgentOutput,
                        )
                        self.trace.llm_tokens(
                            self.name, self._model,
                            struct_response.input_tokens, struct_response.output_tokens,
                        )
                    except Exception as exc:
                        self.trace.error(f"{self.name}: structured output parse failed: {exc}")
                        output = AgentOutput(
                            intent=Intent.ERROR,
                            confidence=0.0,
                            summary="Failed to produce structured response",
                            actions_taken=[],
                            ticket_id=None,
                        )

                msg = AgentMessage(
                    sender=self.name,
                    output=output,
                    timestamp=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                self.trace.hop_end(self.name, output.intent.value, output.confidence, output.summary)
                return msg

            if response.stop_reason == "tool_use" and response.tool_calls:
                # Parallel tool execution within a single agent turn
                import json as _json3
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": _json3.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                await self._memory.add_message(assistant_msg)

                tool_results = await asyncio.gather(
                    *[self._execute_tool(tc.name, tc.arguments, tc.id, trace_id) for tc in response.tool_calls]
                )
                tool_result_messages = [
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                    for tc, result in zip(response.tool_calls, tool_results)
                ]
                for tr_msg in tool_result_messages:
                    await self._memory.add_message(tr_msg)
                continue

        # Loop exhausted without end_turn
        self.trace.error(f"{self.name}: MAX_LOOP_ITERATIONS reached without end_turn")
        error_output = AgentOutput(
            intent=Intent.ERROR,
            confidence=0.0,
            summary="Agent loop exhausted without producing a result",
            actions_taken=[],
            ticket_id=None,
        )
        return AgentMessage(
            sender=self.name,
            output=error_output,
            timestamp=datetime.now(timezone.utc),
            trace_id=trace_id,
        )

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any], tool_id: str, trace_id: str
    ) -> str:
        last_error = ""
        self.trace.tool_call(self.name, tool_name, arguments)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await self.session.call_tool(tool_name, arguments)
                result_text = _extract_text(result.content)
                self.trace.tool_result(self.name, tool_name, result_text[:200], is_error=result.isError)
                if result.isError and attempt < MAX_RETRIES:
                    last_error = result_text
                    continue
                return result_text
            except Exception as exc:
                last_error = str(exc)
                self.trace.error(f"{self.name}: tool {tool_name} attempt {attempt}: {exc}")
                if attempt < MAX_RETRIES:
                    continue
        return f"ERROR: {tool_name} failed after {MAX_RETRIES} attempts: {last_error}"


def _extract_text(content: list) -> str:
    parts = [block.text for block in content if hasattr(block, "text")]
    return "\n".join(parts) if parts else "(empty result)"
