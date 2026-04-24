"""Orchestrator — parallel fan-out/fan-in with LLM-based routing.

Flow:
  1. Triage runs first (always sequential — entry point + classifier)
  2. LLMRouter decides next step (fast path for high-confidence triage)
  3. If route_to_resolution → Resolution + Escalation run in parallel (fan-out)
     Fan-in merges both outputs into a single OrchestratorResult
  4. If route_to_escalation → Escalation runs directly
  5. Terminal intents (resolved, waiting, escalated, error) → done

Replaces the static ROUTING_TABLE dict with LLMRouter. The router logs its
reasoning in every trace, making routing decisions auditable.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from src.agents.escalation import EscalationAgent
from src.agents.resolution import ResolutionAgent
from src.agents.triage import TriageAgent
from src.context import SharedContext
from src.llm.protocol import LLMClient
from src.messages import AgentMessage, AgentOutput, Intent, OrchestratorResult, RouterDecision, SynthesizedResult
from src.routing.llm_router import LLMRouter
from src.trace import TraceLogger

_TERMINAL_INTENTS = {Intent.RESOLVED, Intent.ESCALATED, Intent.WAITING_ON_CUSTOMER, Intent.ERROR}


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


class Orchestrator:
    def __init__(
        self,
        triage: TriageAgent,
        resolution: ResolutionAgent,
        escalation: EscalationAgent,
        context: SharedContext,
        trace_logger: TraceLogger,
        router: LLMRouter,
        llm_client: LLMClient,
        synthesis_model: str,
    ) -> None:
        self._triage = triage
        self._resolution = resolution
        self._escalation = escalation
        self.context = context
        self.trace = trace_logger
        self._router = router
        self._llm = llm_client
        self._synthesis_model = synthesis_model

    async def run(self, request: str, trace_id: str) -> OrchestratorResult:
        await self.context.set(trace_id, "original_request", request)
        await self.context.set(trace_id, "visited", [])

        # Phase 1: Triage (always sequential)
        triage_task = await self._build_task(request, trace_id, "triage")
        triage_msg = await self._run_agent(self._triage, triage_task, trace_id)
        await self._persist_agent_output("triage", triage_msg, trace_id)

        hops: list[AgentMessage] = [triage_msg]
        routing_decisions: list[RouterDecision] = []

        if triage_msg.intent in _TERMINAL_INTENTS:
            self.trace.final(triage_msg.intent.value, len(hops))
            return OrchestratorResult(
                trace_id=trace_id,
                final_intent=triage_msg.intent,
                final_summary=triage_msg.output.summary,
                hops=hops,
                routing_decisions=routing_decisions,
            )

        # Phase 2: LLM routing decision
        ctx_snapshot = await self.context.get_all(trace_id)
        visited: list[str] = ctx_snapshot.get("visited", [])
        decision, elapsed_ms = await self._router.decide(
            request, triage_msg, visited, ctx_snapshot
        )
        routing_decisions.append(decision)
        self.trace.routing_llm(
            "triage", decision.next_agent, decision.confidence, decision.reasoning, elapsed_ms
        )

        if decision.next_agent == "terminal" or decision.next_agent is None:
            self.trace.final(triage_msg.intent.value, len(hops))
            return OrchestratorResult(
                trace_id=trace_id,
                final_intent=triage_msg.intent,
                final_summary=triage_msg.output.summary,
                hops=hops,
                routing_decisions=routing_decisions,
            )

        if decision.next_agent == "escalation":
            # Direct escalation path (operational/management requests or low triage confidence)
            esc_task = await self._build_task(request, trace_id, "escalation")
            esc_msg = await self._run_agent(self._escalation, esc_task, trace_id)
            hops.append(esc_msg)
            self.trace.final(esc_msg.intent.value, len(hops))
            return OrchestratorResult(
                trace_id=trace_id,
                final_intent=esc_msg.intent,
                final_summary=esc_msg.output.summary,
                hops=hops,
                routing_decisions=routing_decisions,
            )

        # Phase 3: Parallel fan-out — Resolution + Escalation simultaneously
        # Both agents see the same enriched task (triage context already in SharedContext)
        parallel_task = await self._build_task(request, trace_id, "resolution")

        res_result, esc_result = await asyncio.gather(
            self._timed_agent(self._resolution, parallel_task, trace_id),
            self._timed_agent(self._escalation, parallel_task, trace_id),
            return_exceptions=True,
        )

        # Fan-in merge
        res_msg = res_result[0] if not isinstance(res_result, BaseException) else None
        esc_msg = esc_result[0] if not isinstance(esc_result, BaseException) else None
        res_elapsed = res_result[1] if not isinstance(res_result, BaseException) else 0.0
        esc_elapsed = esc_result[1] if not isinstance(esc_result, BaseException) else 0.0

        self.trace.parallel_branch(
            "resolution", res_elapsed,
            res_msg.intent.value if res_msg else "error",
            str(res_result) if isinstance(res_result, BaseException) else None,
        )
        self.trace.parallel_branch(
            "escalation", esc_elapsed,
            esc_msg.intent.value if esc_msg else "error",
            str(esc_result) if isinstance(esc_result, BaseException) else None,
        )

        merged_intent, merged_summary = await self._synthesize(res_msg, esc_msg, request)
        for m in [res_msg, esc_msg]:
            if m is not None:
                hops.append(m)

        self.trace.final(merged_intent.value, len(hops))
        return OrchestratorResult(
            trace_id=trace_id,
            final_intent=merged_intent,
            final_summary=merged_summary,
            hops=hops,
            routing_decisions=routing_decisions,
        )

    async def _timed_agent(
        self, agent: object, task: str, trace_id: str
    ) -> tuple[AgentMessage, float]:
        start = time.monotonic()
        msg = await self._run_agent(agent, task, trace_id)  # type: ignore[arg-type]
        elapsed_ms = (time.monotonic() - start) * 1000
        return msg, elapsed_ms

    async def _run_agent(self, agent: object, task: str, trace_id: str) -> AgentMessage:
        return await agent.run(task, trace_id)  # type: ignore[union-attr]

    async def _persist_agent_output(
        self, agent_name: str, msg: AgentMessage, trace_id: str
    ) -> None:
        await self.context.set(trace_id, f"{agent_name}_summary", msg.output.summary)
        if msg.output.ticket_id is not None:
            await self.context.set(trace_id, "ticket_id", msg.output.ticket_id)
        visited: list[str] = await self.context.get(trace_id, "visited", [])
        visited.append(agent_name)
        await self.context.set(trace_id, "visited", visited)

    async def _synthesize(
        self,
        res_msg: AgentMessage | None,
        esc_msg: AgentMessage | None,
        original_request: str,
    ) -> tuple[Intent, str]:
        """LLM synthesis of parallel branch outputs into one coherent outcome.

        Uses the cheap router model. Falls back to the heuristic if LLM call fails.
        """
        if res_msg is None and esc_msg is None:
            return Intent.ERROR, "Both parallel branches failed"
        if res_msg is None:
            return esc_msg.intent, esc_msg.output.summary  # type: ignore[union-attr]
        if esc_msg is None:
            return res_msg.intent, res_msg.output.summary

        prompt = (
            f"Original request: {original_request}\n\n"
            f"Resolution branch:\n"
            f"  intent: {res_msg.intent.value}\n"
            f"  summary: {res_msg.output.summary}\n"
            f"  actions_taken: {res_msg.output.actions_taken}\n\n"
            f"Escalation branch:\n"
            f"  intent: {esc_msg.intent.value}\n"
            f"  summary: {esc_msg.output.summary}\n"
            f"  actions_taken: {esc_msg.output.actions_taken}"
        )
        try:
            result, _ = await self._llm.complete_structured(
                model=self._synthesis_model,
                system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                response_schema=SynthesizedResult,
                max_tokens=256,
            )
            return result.final_intent, result.final_summary  # type: ignore[union-attr]
        except Exception:
            return _heuristic_merge(res_msg, esc_msg)

    async def _build_task(self, original_request: str, trace_id: str, agent_name: str) -> str:
        if agent_name == "triage":
            return original_request
        ctx = await self.context.get_all(trace_id)
        parts = [f"Original request: {original_request}"]
        if ticket_id := ctx.get("ticket_id"):
            parts.append(f"Ticket ID: {ticket_id}")
        if summary := ctx.get("triage_summary"):
            parts.append(f"Triage assessment: {summary}")
        return "\n\n".join(parts)


def _heuristic_merge(
    res_msg: AgentMessage,
    esc_msg: AgentMessage,
) -> tuple[Intent, str]:
    """Fallback merge when LLM synthesis fails. Prefers escalated > resolved."""
    if esc_msg.intent in (Intent.ESCALATED, Intent.RESOLVED):
        final_intent = esc_msg.intent
    elif res_msg.intent == Intent.RESOLVED:
        final_intent = Intent.RESOLVED
    else:
        final_intent = esc_msg.intent
    summary = f"Resolution: {res_msg.output.summary} | Escalation: {esc_msg.output.summary}"
    return final_intent, summary
