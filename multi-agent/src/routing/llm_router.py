"""LLM-powered router with high-confidence fast path and safe fallback.

Routing decision hierarchy:
  1. triage confidence >= 0.85  → trust triage intent directly (no LLM call)
  2. triage confidence < 0.85   → call cheap router model for structured decision
  3. router fails OR confidence < 0.60 → escalation (safe default)

vs static ROUTING_TABLE: the LLM router handles ambiguous/novel patterns,
considers full SharedContext, and logs its reasoning in the trace.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.messages import AgentMessage, Intent, RouterDecision

if TYPE_CHECKING:
    from src.llm.protocol import LLMClient

TRIAGE_TRUST_THRESHOLD = 0.85
ROUTER_CONFIDENCE_THRESHOLD = 0.60

_INTENT_TO_NEXT: dict[Intent, str | None] = {
    Intent.ROUTE_TO_RESOLUTION: "resolution",
    Intent.ROUTE_TO_ESCALATION: "escalation",
    Intent.RESOLVED: "terminal",
    Intent.WAITING_ON_CUSTOMER: "terminal",
    Intent.ESCALATED: "terminal",
    Intent.ERROR: "escalation",
}

_ROUTER_SYSTEM = """\
You are a routing coordinator for a multi-agent IT helpdesk system.
Given the triage output and request context, decide what happens next.

Available next_agent values:
  "resolution"  — Route to Resolution agent for KB lookup / ticket assignment
  "escalation"  — Route to Escalation agent for SLA management and human assignment
  "terminal"    — Request is fully handled, no further routing needed

Rules:
- Prefer "resolution" when the issue seems solvable via knowledge base
- Prefer "escalation" when SLA is at risk, priority is critical, or agents are unavailable
- Use "terminal" only when the triage already resolved or acknowledged the request
- When genuinely unsure, choose "escalation" (safer default)"""


class LLMRouter:
    def __init__(self, llm_client: LLMClient, router_model: str) -> None:
        self._llm = llm_client
        self._model = router_model

    async def decide(
        self,
        original_request: str,
        triage_message: AgentMessage,
        visited: list[str],
        context_snapshot: dict[str, Any],
    ) -> tuple[RouterDecision, float]:
        """Return (RouterDecision, elapsed_ms). Uses fast path when confidence is high."""
        start = time.monotonic()

        # Fast path: trust high-confidence triage
        if triage_message.confidence >= TRIAGE_TRUST_THRESHOLD:
            decision = _triage_to_decision(triage_message)
            elapsed_ms = (time.monotonic() - start) * 1000
            return decision, elapsed_ms

        # Slow path: ask the cheap router model
        try:
            decision = await self._call_router(
                original_request, triage_message, visited, context_snapshot
            )
            if decision.confidence >= ROUTER_CONFIDENCE_THRESHOLD:
                elapsed_ms = (time.monotonic() - start) * 1000
                return decision, elapsed_ms
        except Exception:
            pass

        # Safe fallback
        elapsed_ms = (time.monotonic() - start) * 1000
        return RouterDecision(
            next_agent="escalation",
            reasoning="Router unavailable or low confidence — defaulting to escalation",
            confidence=0.0,
        ), elapsed_ms

    async def _call_router(
        self,
        request: str,
        triage_msg: AgentMessage,
        visited: list[str],
        ctx: dict[str, Any],
    ) -> RouterDecision:
        prompt = (
            f"Original request: {request}\n\n"
            f"Triage result:\n"
            f"  intent: {triage_msg.intent.value}\n"
            f"  confidence: {triage_msg.confidence:.2f}\n"
            f"  summary: {triage_msg.output.summary}\n\n"
            f"Agents already visited: {', '.join(visited) or 'none'}\n"
            f"Ticket ID from context: {ctx.get('ticket_id', 'unknown')}"
        )
        parsed, _ = await self._llm.complete_structured(
            model=self._model,
            system=_ROUTER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            response_schema=RouterDecision,
            max_tokens=256,
        )
        return parsed  # type: ignore[return-value]


def _triage_to_decision(triage_msg: AgentMessage) -> RouterDecision:
    next_agent = _INTENT_TO_NEXT.get(triage_msg.intent, "escalation")
    return RouterDecision(
        next_agent=next_agent,  # type: ignore[arg-type]
        reasoning=f"High-confidence triage decision: {triage_msg.intent.value}",
        confidence=triage_msg.confidence,
    )
