"""Pydantic models for all inter-agent communication and structured outputs.

Replaces the plain dataclasses from exercise-3. All agent outputs are now
validated Pydantic instances — no regex parsing, no json.loads without schema.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Intent(str, Enum):
    ROUTE_TO_RESOLUTION = "route_to_resolution"
    ROUTE_TO_ESCALATION = "route_to_escalation"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    ERROR = "error"


AgentName = Literal["triage", "resolution", "escalation", "orchestrator"]


class AgentOutput(BaseModel):
    """Strict structured output returned by every specialist agent.

    This is the response_schema passed to LLMClient.complete_structured().
    Pydantic validates every field — no manual type coercion needed.
    """

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(description="One sentence: what was done and why")
    actions_taken: list[str]
    ticket_id: int | None


class AgentMessage(BaseModel):
    sender: AgentName
    output: AgentOutput
    timestamp: datetime
    trace_id: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def intent(self) -> Intent:
        return self.output.intent

    @property
    def confidence(self) -> float:
        return self.output.confidence

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RouterDecision(BaseModel):
    """Structured output for the LLM router call."""

    next_agent: Literal["resolution", "escalation", "terminal"] | None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryFact(BaseModel):
    """A summarized fact stored in long-term agent memory."""

    content: str
    source_agent: str
    token_count: int
    created_at: datetime


class SynthesizedResult(BaseModel):
    """Output schema for the fan-in LLM synthesis call."""

    final_intent: Intent
    final_summary: str


class OrchestratorResult(BaseModel):
    trace_id: str
    final_intent: Intent
    final_summary: str
    hops: list[AgentMessage] = Field(default_factory=list)
    routing_decisions: list[RouterDecision] = Field(default_factory=list)

    @property
    def total_hops(self) -> int:
        return len(self.hops)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
