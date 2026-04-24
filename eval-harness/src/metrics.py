"""Metrics computation for eval scenarios.

All functions are pure — no I/O. Input is a scenario dict (from YAML),
an OrchestratorResult, and the raw trace event list from TraceLogger.dump().
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScenarioMetrics:
    # Tool accuracy
    tool_recall: float          # fraction of expected_tools that were actually called
    tool_precision: float       # 1 - (forbidden_tools_called / total_tool_calls)
    tool_f1: float              # harmonic mean of recall and precision

    # Task completion (0.0 / 0.5 / 1.0) — finalized after judge scores are known
    task_completion: float

    # Efficiency
    hop_efficiency: float       # 0.0–1.0 relative to max_hops
    actual_hops: int
    actual_tool_calls: int

    # Error recovery — None if no tool errors occurred
    error_recovery: float | None

    # Intent
    intent_match: bool

    # Routing quality (new in exercise-5)
    routing_used_llm: bool = False      # True if LLM router was invoked (not fast-path)
    routing_confidence: float | None = None  # RouterDecision.confidence if LLM was used
    routing_correct: bool = True        # actual next_agent == expected_routing in scenario
    parallel_speedup_ms: float | None = None  # wall-clock savings from fan-out (esc_ms saved)

    def to_dict(self) -> dict:
        d = {
            "tool_recall": round(self.tool_recall, 3),
            "tool_precision": round(self.tool_precision, 3),
            "tool_f1": round(self.tool_f1, 3),
            "task_completion": self.task_completion,
            "hop_efficiency": round(self.hop_efficiency, 3),
            "actual_hops": self.actual_hops,
            "actual_tool_calls": self.actual_tool_calls,
            "error_recovery": self.error_recovery,
            "intent_match": self.intent_match,
            "routing_used_llm": self.routing_used_llm,
            "routing_confidence": self.routing_confidence,
            "routing_correct": self.routing_correct,
            "parallel_speedup_ms": self.parallel_speedup_ms,
        }
        return d


def compute_metrics(scenario: dict, final_intent: str, total_hops: int, trace: list[dict]) -> ScenarioMetrics:
    """Derive all metrics from the scenario spec and run outputs.

    Args:
        scenario:     scenario dict loaded from scenarios.yaml
        final_intent: OrchestratorResult.final_intent
        total_hops:   OrchestratorResult.total_hops
        trace:        TraceLogger.dump() — list of trace event dicts
    """
    # Extract tool call and tool result events from trace
    tool_call_events = [e for e in trace if e.get("event_type") == "tool_call"]
    tool_result_events = [e for e in trace if e.get("event_type") == "tool_result"]

    tools_called = [e["tool"] for e in tool_call_events]
    total_tool_calls = len(tools_called)

    # --- Tool accuracy ---
    expected_tools: list[str] = scenario.get("expected_tools", [])
    forbidden_tools: list[str] = scenario.get("forbidden_tools", [])

    if expected_tools:
        hits = sum(1 for t in expected_tools if t in tools_called)
        tool_recall = hits / len(expected_tools)
    else:
        tool_recall = 1.0  # no expectation = full recall

    if total_tool_calls > 0 and forbidden_tools:
        forbidden_hit_count = sum(1 for t in tools_called if t in forbidden_tools)
        tool_precision = 1.0 - (forbidden_hit_count / total_tool_calls)
    else:
        tool_precision = 1.0

    if tool_recall + tool_precision > 0:
        tool_f1 = 2 * tool_recall * tool_precision / (tool_recall + tool_precision)
    else:
        tool_f1 = 0.0

    # --- Intent match ---
    expected_intent: str = scenario.get("expected_intent", "")
    intent_match = final_intent == expected_intent if expected_intent else True

    # --- Efficiency ---
    max_hops: int = scenario.get("max_hops", 5)
    expected_agents: list[str] = scenario.get("expected_agents", [])
    min_hops = max(1, len(expected_agents))
    if total_hops <= min_hops:
        hop_efficiency = 1.0
    elif total_hops >= max_hops:
        hop_efficiency = 0.0
    else:
        hop_efficiency = 1.0 - (total_hops - min_hops) / (max_hops - min_hops)

    # --- Error recovery ---
    error_results = [e for e in tool_result_events if e.get("is_error")]
    if error_results:
        recovered = final_intent not in ("error",)
        error_recovery: float | None = 1.0 if recovered else 0.0
    else:
        error_recovery = None

    # --- Routing quality (exercise-5) ---
    routing_events = [e for e in trace if e.get("event_type") == "routing_llm"]
    parallel_events = [e for e in trace if e.get("event_type") == "parallel_branch"]

    routing_used_llm = len(routing_events) > 0
    routing_confidence: float | None = None
    if routing_events:
        routing_confidence = routing_events[-1].get("confidence")

    expected_routing: str | None = scenario.get("expected_routing")
    routing_correct = True
    if expected_routing and routing_events:
        actual_next = routing_events[-1].get("next_agent")
        routing_correct = actual_next == expected_routing

    parallel_speedup_ms: float | None = None
    if len(parallel_events) >= 2:
        elapsed_values = [e.get("elapsed_ms", 0) for e in parallel_events]
        parallel_speedup_ms = max(elapsed_values) - min(elapsed_values)

    return ScenarioMetrics(
        tool_recall=tool_recall,
        tool_precision=tool_precision,
        tool_f1=tool_f1,
        task_completion=0.0,  # finalized in harness after judge scoring
        hop_efficiency=hop_efficiency,
        actual_hops=total_hops,
        actual_tool_calls=total_tool_calls,
        error_recovery=error_recovery,
        intent_match=intent_match,
        routing_used_llm=routing_used_llm,
        routing_confidence=routing_confidence,
        routing_correct=routing_correct,
        parallel_speedup_ms=parallel_speedup_ms,
    )
