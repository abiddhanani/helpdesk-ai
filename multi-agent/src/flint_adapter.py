"""Flint adapters wrapping BaseSpecialistAgent subclasses and the synthesis fan-in.

Each SpecialistAgentAdapter opens a fresh MCP session per run() call because
the embedded Flint engine runs in its own event loop (asyncio.run() in a thread
pool), so sessions from the outer loop cannot be reused here.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from flint_ai import FlintAdapter, AgentRunResult

from src.context import SharedContext
from src.llm.protocol import LLMClient
from src.messages import OrchestratorResult, SynthesizedResult
from src.trace import TraceLogger

_SYNTHESIS_SYSTEM = """\
You are a synthesis coordinator for a multi-agent IT helpdesk system.
Two specialist agents (Resolution and Escalation) ran in parallel on the same ticket.
Synthesize their outputs into one coherent final decision.

Rules:
- If both agents succeeded, pick the intent that best reflects what actually happened
  (escalated > resolved — escalation is an audit requirement even if the issue was KB-resolved)
- If one agent errored, use the healthy branch's intent
- Write a single clear summary combining the key actions from both agents
- Do not invent information not present in the agent outputs"""


class SpecialistAgentAdapter(FlintAdapter):
    """Wraps a BaseSpecialistAgent subclass as a Flint adapter.

    Opens a fresh MCP session on every run() call so it works correctly inside
    the embedded Flint engine's own asyncio event loop.
    """

    def __init__(
        self,
        name: str,
        agent_class: type,
        server_params: StdioServerParameters,
        llm_client: LLMClient,
        agent_model: str,
        router_model: str,
    ) -> None:
        super().__init__(name=name)
        self._agent_class = agent_class
        self._server_params = server_params
        self._llm_client = llm_client
        self._agent_model = agent_model
        self._router_model = router_model

    async def run(self, input_data: dict[str, Any]) -> AgentRunResult:
        prompt = input_data["prompt"]
        trace_id = input_data.get("task_id", uuid.uuid4().hex[:8])
        trace_logger = TraceLogger(trace_id)
        context = SharedContext()

        async with stdio_client(self._server_params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                agent = self._agent_class(
                    session,
                    self._llm_client,
                    self._agent_model,
                    self._router_model,
                    trace_logger,
                    context,
                )
                await agent.discover_tools()
                msg = await agent.run(prompt, trace_id)

        return AgentRunResult(output=json.dumps(msg.to_dict()))


class SynthesisAdapter(FlintAdapter):
    """Fan-in node: merges resolution + escalation outputs into a single decision."""

    def __init__(self, llm_client: LLMClient, synthesis_model: str) -> None:
        super().__init__(name="synthesis")
        self._llm = llm_client
        self._model = synthesis_model

    async def run(self, input_data: dict[str, Any]) -> AgentRunResult:
        prompt = input_data["prompt"]
        try:
            result, _ = await self._llm.complete_structured(
                model=self._model,
                system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                response_schema=SynthesizedResult,
                max_tokens=256,
            )
            return AgentRunResult(output=result.model_dump_json())  # type: ignore[union-attr]
        except Exception as exc:
            return AgentRunResult(
                output="",
                success=False,
                error=str(exc),
            )
